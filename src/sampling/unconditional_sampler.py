import sys
import torch
import random
import numpy as np
from tqdm import tqdm
from src.utils.model_utils import _print

class UnconditionalSampler:
    def __init__(self, tokenizer, model):
        self.model = model
        self.tokenizer = tokenizer

        self.device = self.model.device
        self.mask_id = self.tokenizer.mask_token_id
        self.seed_everything(seed=42)

    @torch.inference_mode()
    def sample_unconditional(self, xt, num_steps, tau=1.0, kappa_fn=lambda t: t, eta=1, alpha=1., banned_token_ids=None, return_logits=None): 
        """
        Stochastic remasking sampling method for iterative refinement of sequences.

        Args:
            xt (Tensor): Initial token tensor.
            num_steps (int): Number of refinement steps.
            tau (float): Temperature parameter for softmax sampling.
            kappa_fn (callable): Function controlling the unmasking schedule.
            eta (float): Scaling factor for score adjustments.
            alpha (float): Weighting for confidence-based scoring.

        Returns:
            Tensor: Final sampled sequence tensor.
        """
        
        dt = 1 / num_steps
        fix_mask = xt != self.mask_id # tokens to retain
        attention_mask = torch.ones_like(xt).to(self.device)

        for i in range(1, num_steps + 1):
            kappa_t = kappa_fn(i * dt)
            logits = self.model(input_ids=xt, attention_mask=attention_mask)
            last_mask = xt == self.mask_id # tokens currently masked
            unmask_t = ~last_mask & ~fix_mask # unmasked and not fixed tokens - candidates for masking

            x0, logp = self.stochastic_sample_from_categorical(logits, tau, banned_token_ids=banned_token_ids) # tokens, logprobs

            # Confidence-based scoring
            entropy = torch.distributions.Categorical(logits=logits).entropy()
            score = alpha * logp + (1 - alpha) * -entropy # alpha = 1 --> score = logp
            score = score.masked_fill(fix_mask, float('inf'))

            score[unmask_t] = score[unmask_t] * eta

            num_to_mask = ((~fix_mask).sum(1, keepdim=True).float() * (1 - kappa_t)).long()
            lowest_k_mask = self.topk_lowest_masking(score, num_to_mask)

            xt[lowest_k_mask] = self.mask_id
            mask_2_x0 = last_mask & ~lowest_k_mask
            xt[mask_2_x0] = x0[mask_2_x0]

            # print(f"Step {i}/{num_steps} | eta: {eta}, alpha: {alpha}, Stochastic remask: \n", xt[0])

        xt[xt == self.mask_id] = x0[xt == self.mask_id]
        return xt, logits if return_logits else xt

    def stochastic_sample_from_categorical(self, logits, temperature, noise_scale=1.0, banned_token_ids=None):
        """
        Sample from a categorical distribution with optional temperature scaling and Gumbel noise.
        """
        logits = logits.double()

        if banned_token_ids is not None:
            banned_token_mask = torch.zeros_like(logits, device=logits.device).bool()
            for token_id in banned_token_ids:
                banned_token_mask[..., token_id] = True
            logits = logits.masked_fill(banned_token_mask, float('-inf'))

        if temperature != 0:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            logits = logits / temperature + noise_scale * gumbel_noise
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)

        return tokens, scores

    def topk_lowest_masking(self, scores, cutoff_len):
        """
        scores: [b, n]
        cutoff_len: [b, 1]
        returns:
            mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
        """
        sorted_index = scores.sort(-1)[0]
        cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
        return scores < cutoff

    def seed_everything(self, seed):
        """
        Set the seed for reproducibility across various libraries.
        """
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False