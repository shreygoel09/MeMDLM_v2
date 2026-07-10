import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.guidance.oligo.oligo_module import OligomerClassifier
from src.sampling.unconditional_sampler import UnconditionalSampler
from src.lm.memdlm.diffusion_module import MembraneDiffusion
from src.utils.model_utils import _print
from src.utils.config_utils import repo_path



class NOSSampler:
    def __init__(self, config, device, mdlm, esm, tokenizer):
        self.config = config
        self.device = device

        self.diffusion = mdlm
        self.memdlm_lm = self.diffusion.model.lm_head
        self.tokenizer = self.diffusion.tokenizer

        ckpt_path = str(repo_path("checkpoints", config.wandb.name, "best_model.ckpt"))
        self.classifier_model = OligomerClassifier(config).eval().to(self.device)
        state_dict = self.classifier_model.get_state_dict(ckpt_path)
        self.classifier_model.load_state_dict(state_dict)

        self.SPECIAL_TOKEN_IDS = {0, 1, 2, 3, 29}


    def sample_from_categorical(self, logits):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        logits += gumbel_noise
        log_probs = F.log_softmax(logits, dim=-1)
        _, tokens = log_probs.max(dim=-1)
        return tokens, log_probs
    

    def get_clf_preds(self, hidden_states, attention_mask):
        """Obtain diffusion model logits and classifier predictions from hidden states"""
        batch = {"embeds": hidden_states.squeeze(), "attention_mask": attention_mask.squeeze()}
        preds = self.classifier_model(batch)
        return self.memdlm_lm(hidden_states), preds


    def embed_and_run_clf(self, input_ids, attention_masks):
        """Get sequence embeddings and classifier model predictions"""
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_masks)
        sequence_embeddings = outputs.last_hidden_state.squeeze(0)

        batch = {"embeds": sequence_embeddings, "attention_mask": attention_masks}
        preds = self.classifier_model(batch)
        
        return {
            "clf_preds": preds.requires_grad_(True), # Enable gradients for backprop
            "embeds": sequence_embeddings 
        }


    def compute_saliency(self, embeddings, attention_masks):
        """
        Compute a saliency map using gradients as defined in LaMBO-2 (https://arxiv.org/pdf/2305.20009)
        """
        embeddings = embeddings.detach().requires_grad_(True)
        batch = {
            "embeds": embeddings.squeeze(),
            "attention_mask": attention_masks.squeeze()
        }
        out = self.classifier_model(batch).sum() 
        out.backward(retain_graph=True)

        # Creating the saliency map (Eq.5 in LaMBO-2 paper)
        grads = embeddings.grad.abs().sum(dim=-1)  # Summation across hidden dim. Abs value for mangitude only
        saliency = grads.pow(1.0 / self.config.olig_guidance.temperature).clamp(min=self.config.olig_guidance.eps)
        return saliency.squeeze()
    

    def determine_edit_positions(self, saliency_map, preds):
        """
        Create a one-hot mask that indicates the top-k low-value residue positions.
        We defind low-value positions as those with high saliency scores and
        thus a high edit probability.
        """
        probabilities = saliency_map.masked_fill(preds >= self.config.olig_guidance.residue_thresh, 0.0) # exclude high-value tokens
        probabilities = probabilities / probabilities.sum()
        
        nonzero = probabilities.count_nonzero().item()
        seq_len = probabilities.shape[0]
        topk = max(1, int(seq_len * self.config.olig_guidance.topk_frac))
        _, topk_edit_pos = torch.topk(probabilities, min(topk, nonzero))
        
        mask = torch.zeros_like(probabilities).scatter(0, topk_edit_pos, torch.ones_like(probabilities))
        return mask.unsqueeze(-1)


    def update_logits(self, og_hidden, og_logits, hidden_state_mask, attention_mask, optimizer, delta):
        """
        Shift logits distribution towards only high-quality residues by applying the explore-exploit loss.
        """
        lamb = self.config.olig_guidance.reg_strength
        
        h_current = og_hidden + hidden_state_mask * delta
        new_logits, v_ht_prime = self.get_clf_preds(h_current, attention_mask)

        kl = F.kl_div(
            F.log_softmax(new_logits, dim=-1),
            F.softmax(og_logits, dim=-1),
            reduction='sum'
        )
        
        loss = lamb * kl - v_ht_prime.sum()
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        
        return delta
    

    def optimized_sampling(self, og_logits, og_hidden, attention_mask, n_steps):
        """Main entry point to optimize a generated sequence."""
        eta = self.config.olig_guidance.step_size

        # Calculate initial clf predictions
        batch = {"embeds": og_hidden, "attention_mask": attention_mask}
        preds = self.classifier_model(batch)
        
        delta = nn.Parameter(torch.zeros_like(og_hidden), requires_grad=True)
        optimizer = torch.optim.Adagrad([delta], lr=eta)
        optimizer.zero_grad()
        
        with torch.enable_grad():
            for n in range(n_steps):

                # Compute saliency map and edit positions using updated hidden states
                saliency_map = self.compute_saliency(og_hidden + delta.data, attention_mask)
                
                # One-hot mask that encodes the saliency-selected edit positions
                mask = self.determine_edit_positions(saliency_map, preds)

                # Optimize and generate the new sequence
                delta = self.update_logits(
                    og_hidden=og_hidden,
                    og_logits=og_logits,
                    hidden_state_mask=mask,
                    attention_mask=attention_mask,
                    optimizer=optimizer,
                    delta=delta
                )

        h_new = og_hidden + delta.data
        new_logits, _ = self.get_clf_preds(h_new, attention_mask)
        
        return new_logits, h_new
    

    def sample_guidance(self, tokens, guide_steps, diffusion_steps):
        x = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        og_pred = self.classifier_model({"input_ids": x, "attention_mask": attention_mask,})

        for _ in range(diffusion_steps):
            hidden_states = self.diffusion(x, attention_mask, return_hidden=True)
            logits = self.memdlm_lm(hidden_states)
            hidden_states = hidden_states.unsqueeze(0) if hidden_states.ndim != 3 else hidden_states

            logits, hidden_states = self.optimized_sampling(logits, hidden_states, attention_mask, guide_steps)
            logits = self.memdlm_lm(hidden_states)
            x, _ = self.sample_from_categorical(logits)

        seq = x.squeeze()
        # _print(seq)
        # start = 1 if seq[0].item() in self.SPECIAL_TOKEN_IDS else 0
        # end = -1 if seq[-1].item() in self.SPECIAL_TOKEN_IDS else len(seq)
        # seq = seq[start:end]
        # _print(seq)

        _print(seq)
        seq = seq[1:-1]
        _print(seq)


        optim_tokens = self.tokenizer.decode(seq, skip_special_tokens=True).replace(" ", "")
        final_pred = self.classifier_model({"embeds": hidden_states, "attention_mask": attention_mask,})

        return optim_tokens, F.sigmoid(og_pred).item(), F.sigmoid(final_pred).item()