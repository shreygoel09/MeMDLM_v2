import os
import math
import torch
import torch.nn.functional as F

from src.utils.model_utils import _print
from src.guidance.solubility_module import SolubilityClassifier
from src.sampling.unconditional_sampler import UnconditionalSampler


class GuidedSampler:
    def __init__(self, config, esm_model, tokenizer, diffusion, device):
        self.config = config
        self.device = device

        self.esm = esm_model
        self.memdlm = diffusion
        self.tokenizer = tokenizer
        self.uncond_generator = UnconditionalSampler(self.tokenizer, self.memdlm)

        ckpt_path = os.path.join(f"/home/a03-sgoel/MeMDLM_v2/checkpoints/{config.wandb.name}/best_model.ckpt")
        self.classifier_model = SolubilityClassifier(config)
        state_dict = self.classifier_model.get_state_dict(ckpt_path)
        self.classifier_model.load_state_dict(state_dict)
        self.classifier_model.eval().to(self.device)

        self.top_p = self.config.guidance.top_p
        self.alpha = self.config.guidance.alpha
        self.gamma = self.config.guidance.gamma
        self.saliency_eps = self.config.guidance.saliency_eps
        self.saliency_t = self.config.guidance.saliency_t
        self.sampling_t = self.config.guidance.sampling_t
        self.boltzmann_t = self.config.guidance.boltzmann_t
    

    def embed_sequence(self, input_ids, attention_masks):
        with torch.no_grad():
            outs = self.esm(
                input_ids=input_ids,
                attention_mask=attention_masks,
                output_hidden_states=True,
                output_attentions=True
            )
            embeds = outs.hidden_states[-1]
            attn_matrix = outs.attentions
        return embeds, attn_matrix


    def sample_from_categorical(self, logits, temperature, noise_scale=1.0):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        logits = (logits / temperature) + (noise_scale * gumbel_noise)
        
        log_probs = F.log_softmax(logits, dim=-1)
        _, tokens = log_probs.max(dim=-1)
        
        return tokens, log_probs
    

    def denoise_sequence(self, input_ids, attn_masks):
        """
        Compute the current and prior sequences' log prob distribution.
        """
        has_masks = (input_ids == self.tokenizer.mask_token_id).any()

        # Denosie the sequence if needed
        if has_masks:
            xt_prior, logits_prior = self.uncond_generator.sample_unconditional(
                xt=input_ids,
                num_steps=self.config.guidance.n_steps,
                tau=self.sampling_t, 
                return_logits=True
            )
        else:
            xt_prior = input_ids
            logits_prior = self.memdlm(input_ids=input_ids, attention_mask=attn_masks)

        # Take the final sampling step
        _, logits = self.uncond_generator.sample_unconditional(
            xt=xt_prior,
            num_steps=1, # Only need 1 sampling step
            tau=self.sampling_t,
            return_logits=True
        )

        # Get final sequence log probs (always needed)
        x0, logp_lm = self.sample_from_categorical(logits, temperature=self.sampling_t)

        return x0.squeeze(), logp_lm.squeeze(), logits_prior


    def get_prior(self, logits_prior, solubility_logits):
        if self.config.guidance.prior == "boltzmann":
            hydrophilic = ["D","E","K","R","N","Q","H","S","T","Y"]
            hydrophobic = ["L","I","V","F","W","M","A","C","G","P"]
            amino_acids = hydrophilic + hydrophobic
            
            tokens = list(self.tokenizer.get_vocab().keys())
            other = [tok for tok in tokens if tok not in amino_acids]

            hydrophilic_idxs = [self.tokenizer.convert_tokens_to_ids(aa) for aa in hydrophilic]
            hydrophobic_idxs = [self.tokenizer.convert_tokens_to_ids(aa) for aa in hydrophobic]
            other_idxs = [self.tokenizer.convert_tokens_to_ids(tok) for tok in other]

            bias = torch.zeros(len(tokens), device=self.device)
            bias[hydrophilic_idxs] = 1.0
            bias[hydrophobic_idxs] = -1.0
            bias[other_idxs] = 0.0

            sol_scores = torch.sigmoid(solubility_logits)
            token_bias = sol_scores.unsqueeze(-1) * bias

            lm_probs = F.softmax(logits_prior / self.sampling_t, dim=-1)
            boltz_weight = torch.exp(token_bias / self.boltzmann_t)

            p_prior = lm_probs * boltz_weight
            p_prior = p_prior / p_prior.sum(dim=-1, keepdim=True)
            logp_prior = torch.log(p_prior)

        elif self.config.guidance.prior == "lm_probs":
            _, logp_prior = self.sample_from_categorical(logits_prior, temperature=self.sampling_t)

        return logp_prior.squeeze()


    def compute_saliency_map(self, embeds, solubility_logits):
        """
        Compute a saliency map as in LaMBO-2 (https://arxiv.org/abs/2305.20009) Eq. 5
        """
        # Gradient tracking is already enabled for the embeddings
        solubility_logits.sum().backward(retain_graph=True) # Clf gradients wrt hidden states
        grads = embeds.grad.abs().sum(dim=-1) # Aggergate across hidden dim. Abs value for mangitude only.
        saliency = grads.pow(1.0 / self.saliency_t).clamp(min=self.saliency_eps).to(self.device)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
        return saliency.squeeze()


    def determine_edit_positions(self, saliency_map, soluble_indices, solubility_logits):
        """
        Fix the insoluble residues and additional TM residues to
        maintain membrane-like protein structure.
        """
        seq_len = saliency_map.shape[0]

        # Initialize a mask to store the editable token positions
        edit_mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)

        # Check for any provided soluble residues, otherwise use classifier preds
        if len(soluble_indices) > 0:
            edit_mask[soluble_indices] = False
        elif soluble_indices is None or len(soluble_indices) == 0:
            solubility_preds = F.sigmoid(solubility_logits)
            edit_mask[solubility_preds > 0.5] = False

        # Find additional TM residues
        num_conserved = max(1, int(0.1 * edit_mask.sum()))
        _, topk_idxs = torch.topk(saliency_map, num_conserved)
        edit_mask[topk_idxs] = False

        edit_idxs = edit_mask.nonzero(as_tuple=True)[0]
        return edit_idxs


    def create_neighborhood(self, edit_pos, attn_matrix, top_p):
        """
        Select a dynamic "neighborhood" of tokens for edit position via top-p sampling.
        Attention scores find relevant tokens, avoding blind updates of the individual token
        """
        # Get the attention scores for the current edit position
        row = attn_matrix[edit_pos].clone().squeeze()
        row = row.index_fill(
            dim=0,
            index=torch.tensor([0, edit_pos, row.size(0)-1], device=row.device),
            value=float('-inf')
        )
        
        # Top-p (nucleus) sampling of tokens via normed attention scores
        temp = 1.0 / math.log(row.size(0)) # scale temp with seq len to balance
        attn_probs = F.softmax(row / temp, dim=0)
        sorted_probs, sorted_idxs = torch.sort(attn_probs, descending=True)
        cum_probs = sorted_probs.cumsum(dim=0)
        cutoff = (cum_probs <= top_p).nonzero(as_tuple=True)[0]
        
        # Ensure neighborhoods will always have 1 token
        final_idx = cutoff[-1].item() + 1 if cutoff.numel() > 0 else 1
        neighborhood = sorted_idxs[:final_idx]
        return neighborhood
    

    def compute_saliency_weight(self, edit_pos, attn_mat, saliency_map, neighborhood):
        """
        Blend the saliency of the neighborhood's tokens and the token at the edit position.
        """
        neighborhood_attns = attn_mat[edit_pos, neighborhood]
        neighborhood_attns /= neighborhood_attns.sum()

        neighborhood_saliencies = saliency_map[neighborhood]
        
        neighborhood_weight = torch.sum(neighborhood_attns * neighborhood_saliencies)
        ctxt_aware_saliency = saliency_map[edit_pos] + (self.gamma * neighborhood_weight)

        return ctxt_aware_saliency


    def compute_guidance_dist(self, logp_lm, logp_prior, saliency_weight):
        """
        Define a guidance distribution between a prior and the current LM probs.
        Compute the log probs of the "new" (optimized) token.
        """
        w = torch.sigmoid(saliency_weight * self.alpha)  # Between [0, 1] to ensure valid probs
        p_lm = torch.exp(logp_lm)
        p_prior = torch.exp(logp_prior)
        mixed_probs = (1 - w) * p_lm + w * p_prior
        guidance_dist = torch.log(mixed_probs + 1e-12)
        return guidance_dist
    

    def check_scaffold(self, seq1, seq2, idxs):
        changed = (seq1[idxs] != seq2[idxs])
        if changed.any():
            _print('soluble residues changed')
        else:
            _print('no soluble residue changes')


    def optimize_sequence(self, input_ids, attn_masks, soluble_indices):
        _print(f'soluble idx: {soluble_indices}')

        # Initialize token ids, logits, and log probs of sequence
        x0, logp_lm, logits_prior = self.denoise_sequence(input_ids, attn_masks)
        _print(f'og tokens: {x0}')
        _print(f'og tokens: {x0.shape}')
        _print(f'og log probs: {logp_lm.shape}')
        
        # Embeddings and attention matrix of current sequence
        embeds, attn_mats = self.embed_sequence(x0.unsqueeze(0), attn_masks)
        embeds = embeds.detach().clone().requires_grad_(True) # enable grad tracking for saliency map
        attn_matrix = attn_mats[-1].mean(dim=1)[0].squeeze(0)

        # Precompute logits of the classifier to avoid repeated calls
        batch = {"embeds": embeds, "attention_mask": attn_masks}
        solubility_logits = self.classifier_model(batch)

        # Create a saliency map to determined optimal edit positions
        saliency_map = self.compute_saliency_map(embeds, solubility_logits)
        _print(f'saliency map: {saliency_map}')
        edit_positions = self.determine_edit_positions(saliency_map, soluble_indices, solubility_logits)
        _print(f'edit positions: {edit_positions}')

        # Compute the log probs of the prior dist
        logp_prior = self.get_prior(logits_prior, solubility_logits)
        _print(f'prior log probs: {logp_prior.shape}')
        
        # Optimize the insoluble residues
        for edit_pos in edit_positions.tolist():
            neighborhood = self.create_neighborhood(
                edit_pos,
                attn_matrix,
                self.top_p
            )
            _print(f'neighborhood: {neighborhood}')
            
            ctxt_aware_saliency = self.compute_saliency_weight(
                edit_pos,
                attn_matrix,
                saliency_map,
                neighborhood
            )
            _print(f'ctx aware saliency: {ctxt_aware_saliency}')
            
            logp_lm_prime = self.compute_guidance_dist(
                logp_lm[edit_pos], 
                logp_prior[edit_pos],
                ctxt_aware_saliency
            )
            logp_lm[edit_pos] = logp_lm_prime

            tot = torch.exp(logp_lm_prime).sum()
            one = torch.tensor(1.0, dtype=tot.dtype, device=tot.device)
            assert torch.isclose(tot, one, atol=1e-4), f"Invalid prob distribution. Sum = {tot:5f}"

        # Sample new tokens        
        x0_prime = torch.distributions.Categorical(logits=logp_lm).sample()
    
        # Check if any soluble residues have been changed
        self.check_scaffold(x0, x0_prime, soluble_indices)

        # Preserve the initial sequence scaffold by copying over the soluble tokens
        x0_prime[soluble_indices] = x0[soluble_indices]
        self.check_scaffold(x0, x0_prime, soluble_indices)

        return x0_prime