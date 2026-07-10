import os
import torch
from torch import nn
import torch.nn.functional as F

from src.guidance.multipass.multipass_module import MultipassClassifier
from src.utils.model_utils import _print
from src.utils.config_utils import repo_path



class MultipassSampler:
    def __init__(self, config, device, mdlm, tokenizer):
        self.config = config
        self.device = device

        self.diffusion = mdlm
        self.memdlm_lm = self.diffusion.model.lm_head
        self.tokenizer = self.diffusion.tokenizer

        ckpt_path = str(repo_path("checkpoints", config.wandb.name, "best_model.ckpt"))
        self.classifier_model = MultipassClassifier(config, self.diffusion).eval().to(self.device)
        state_dict = self.classifier_model.get_state_dict(ckpt_path)
        self.classifier_model.load_state_dict(state_dict)

        self.SPECIAL_TOKEN_IDS = {0, 1, 2, 3, 29}


    def stochastic_sample_from_categorical(self, logits, temperature, noise_scale=1.0):
        """
        Sample from a categorical distribution with optional temperature scaling and Gumbel noise.
        Returns the sampled tokens and their log-probabilities (used as confidence scores).
        """
        logits = logits.double()
        if temperature != 0:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits = (logits / temperature) + (noise_scale * gumbel_noise)
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
        return tokens, scores


    def topk_lowest_masking(self, scores, cutoff_len):
        """
        scores: [b, n]; cutoff_len: [b, 1]
        Returns a [b, n] bool mask, True at the cutoff_len lowest-scoring positions.
        """
        sorted_index = scores.sort(-1)[0]
        cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
        return scores < cutoff
    

    def classifier_score(self, hidden_states, attention_mask):
        return self.classifier_model(x_t=None, attn_mask=attention_mask, with_hidden=True, embeds=hidden_states)


    def guidance_loss(self, og_hidden, og_logits, attention_mask, delta):
        """
        Implementation of explore-exploi  guidance as in LaMBO-2 (https://arxiv.org/pdf/2305.20009).
        Disregarding use of saliency map as we have one score per sequence.
        """
        lamb = self.config.guidance.reg_strength

        h_current = og_hidden + delta
        new_logits = self.memdlm_lm(h_current)
        score = self.classifier_score(h_current, attention_mask)

        kl = F.kl_div(
            F.log_softmax(new_logits, dim=-1),
            F.softmax(og_logits, dim=-1),
            reduction='sum'
        )

        loss = lamb * kl - score.sum()
        return loss


    def optimized_sampling(self, og_logits, og_hidden, attention_mask, n_steps):
        """
        At each diffusion timestep, take n_steps of gradient-based guidance on the hidden states
        """
        eta = self.config.guidance.step_size
        og_logits = og_logits.detach()

        delta = nn.Parameter(torch.zeros_like(og_hidden), requires_grad=True)
        optimizer = torch.optim.Adagrad([delta], lr=eta)

        with torch.enable_grad():
            for _ in range(n_steps):
                optimizer.zero_grad()
                loss = self.guidance_loss(og_hidden, og_logits, attention_mask, delta)
                loss.backward()
                optimizer.step()

        h_new = og_hidden + delta.data
        new_logits = self.memdlm_lm(h_new)
        return new_logits, h_new
    

    def guided_logits(self, xt, attention_mask, guide_steps):
        """
        Run the diffusion model on the current sequence and apply classifier guidance to the
        hidden states, returning the guided LM-head logits used for scoring/sampling this step.
        """
        with torch.no_grad():
            hidden_states = self.diffusion(xt, attention_mask, return_hidden=True)
            hidden_states = hidden_states.unsqueeze(0) if hidden_states.ndim != 3 else hidden_states
            base_logits = self.memdlm_lm(hidden_states)

        logits, _ = self.optimized_sampling(base_logits, hidden_states, attention_mask, guide_steps)
        return logits


    def sample_guidance(self, tokens, guide_steps, diffusion_steps,
                        kappa_fn=lambda t: t, eta=1.0, alpha=1.0):
        """
        Confidence-based progressive-unmasking denoising (as in the unconditional sampler),
        but each step uses classifier-guided logits instead of the raw model logits.

        Args:
            kappa_fn: unmasking schedule, kappa(t) in [0,1]; fraction of positions committed by step t.
            eta:      re-masking scale applied to already-unmasked candidate positions.
            alpha:    blends token log-prob (alpha=1) and negative entropy (alpha=0) in the score.
        """
        tau = self.config.guidance.sampling_temperature

        xt = tokens['input_ids'].to(self.device)
        attention_mask = torch.ones_like(xt).to(self.device)

        dt = 1 / diffusion_steps
        fix_mask = (xt != self.tokenizer.mask_token_id) # would be none for de novo setting

        x0 = xt
        for i in range(1, diffusion_steps + 1):
            kappa_t = kappa_fn(i * dt)

            logits = self.guided_logits(xt, attention_mask, guide_steps)

            with torch.no_grad():
                last_mask = xt == self.tokenizer.mask_token_id          
                unmask_t = ~last_mask & ~fix_mask                       

                x0, logp = self.stochastic_sample_from_categorical(logits, tau)

                entropy = torch.distributions.Categorical(logits=logits).entropy()
                score = alpha * logp + (1 - alpha) * -entropy
                score = score.masked_fill(fix_mask, float('inf'))  # never remask fixed tokens
                score[unmask_t] = score[unmask_t] * eta

                num_to_mask = ((~fix_mask).sum(1, keepdim=True).float() * (1 - kappa_t)).long()
                lowest_k_mask = self.topk_lowest_masking(score, num_to_mask)

                xt[lowest_k_mask] = self.tokenizer.mask_token_id  # remask lowest-confidence
                mask_2_x0 = last_mask & ~lowest_k_mask  # commit newly-confident positions
                xt[mask_2_x0] = x0[mask_2_x0]

        xt[xt == self.tokenizer.mask_token_id] = x0[xt == self.tokenizer.mask_token_id]  # commit remaining

        seq = xt.squeeze()[1:-1]
        optim_tokens = self.tokenizer.decode(seq, skip_special_tokens=True).replace(" ", "")

        with torch.no_grad():
            final_hidden = self.diffusion(xt, attention_mask, return_hidden=True)
            final_hidden = final_hidden.unsqueeze(0) if final_hidden.ndim != 3 else final_hidden
            final_preds = self.classifier_score(final_hidden, attention_mask)

        return optim_tokens, torch.sigmoid(final_preds).item()