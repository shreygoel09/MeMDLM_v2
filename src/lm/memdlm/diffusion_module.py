import os
import gc
import torch

import torch.nn.functional as F
import lightning as pl

from typing import Optional
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.utils.model_utils import _print
from src.utils.optimizer_utils import get_optimizer, get_scheduler


class MembraneDiffusion(pl.LightningModule):
    def __init__(self, config):
        """
        Args:
            config (OmegaConf): config.yaml file with all training parameters
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(logger=True)

        self.model = AutoModelForMaskedLM.from_pretrained(config.lm.pretrained_evoflow, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm.pretrained_evoflow)

        self.mask_id = self.tokenizer.mask_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def forward(self, input_ids, attention_mask, guidance: Optional[bool] = False):
        """
        Forward pass through language model.

        Args:
            - input_ids (torch.Tensor): [B, L], token ids
            - attention_mask (torch.Tensor): [B, L], pad/non-pad binary mask
        Returns:
            - logits (torch.Tensor): [B, L, V], unnormalized model outputs
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    # -------# Diffusion #-------- #
    def step(self, batch):
        labels = batch['input_ids'] 

        # Forward diffusion
        t1 = self.sample_t(labels) # Sample timestep
        xt, _ = self.noise_x0(labels, t1, maskable_mask=self.is_maskable(labels)) # Noise sequence
        logits = self.forward(input_ids=xt, attention_mask=batch['attention_mask']) # Model logits

        # Loss computation
        weight = self.get_weight(t1, weight_type=self.config.lm.weight_type)  # RDM uses a weighted cross entropy loss
        loss_out = self.compute_loss(logits, labels, weight) # Compute loss and ppl

        self.cleanup()
        return loss_out['loss'], loss_out['ppl']
    
    def sample_t(self, labels, rdm_coupling=False):
        """
        Sample diffusion timesteps. Non-coupling RDM only uses one timestep (t1).
        """
        timesteps = torch.randint(
            1,
            self.config.lm.num_diffusion_timesteps + 1,
            (2 if rdm_coupling else 1) * (labels.size(0),),
            device=labels.device
        )

        if rdm_coupling:
            return timesteps.chunk(2)
        return timesteps

    def noise_x0(self, x0, t1, maskable_mask):
        """
        Apply noise to the initial sequence x0.
        """
        u = torch.rand_like(x0, dtype=torch.float) 
        t1_mask = (u < (t1 / self.config.lm.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x0.masked_fill(t1_mask, self.mask_id)
        x_t1 = x_t1.masked_fill(t1_mask, self.mask_id)
        return x_t1, t1_mask

    def get_weight(self, t, weight_type):
        """
        Compute the weighting factor for the RDM-derived loss (weighted cross-entropy).
        """
        num_timesteps = self.config.lm.num_diffusion_timesteps
        weight = {
            "linear": (num_timesteps - (t - 1)),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(t),
        }[weight_type][:, None].float() / num_timesteps
        return weight.squeeze()

    def compute_loss(self, logits, labels, weight):
        """
        Compute the cross entropy loss per sample.
        First, compute the per-token loss (with no reduction), then reduce over the sequence length for each sample.
        Finally, average over the batch.
        
        Args:
            logits (torch.Tensor): [B, L, vocab_size], unnormalized model outputs
            labels (torch.Tensor): [B, L], target labels (with padding tokens as -100)
            weight (torch.Tensor): [B, 1], per-sample weight for loss calculation
        Returns:
            loss (torch.Tensor): Averaged loss over the batch
            logging_output (torch.Tensor): Dictionary of values for logging
        """

        loss_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='none',
            ignore_index=self.pad_id,
        )
        
        loss_token = loss_token.view(labels.size(0), labels.size(1)) # Reshape to [B, L]
        valid_mask = (labels != self.pad_id)
        
        sample_loss = (loss_token * valid_mask.float()).sum(dim=1) / valid_mask.float().sum(dim=1).clamp(min=1)
        sample_loss *= weight # RDM weighting
        ppl = torch.exp(sample_loss)

        return {'ppl': ppl.mean(), 'loss': sample_loss.mean()}
    

    # -------# Training / Evaluation #-------- #
    def training_step(self, batch):
        loss, ppl = self.step(batch)
        self.log("train/loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/ppl", ppl.item(), on_step=True, on_epoch=False, prog_bar=False)
        return loss
    
    def validation_step(self, batch):
        loss, ppl = self.step(batch)
        self.cleanup()
        self.log("val/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ppl", ppl.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def test_step(self, batch):
        loss, ppl = self.step(batch)
        self.cleanup()
        self.log('test/loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/ppl", ppl.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss


    # -------# Helper methods #-------- #
    def is_maskable(self, input_ids: torch.Tensor):
        return (
            (input_ids != self.tokenizer.pad_token_id) 
            & (input_ids != self.tokenizer.cls_token_id)
            & (input_ids != self.tokenizer.eos_token_id)
        )

    def configure_optimizers(self):
        """
        Choosing which optimizer and lr scheduler to use.
        """
        optimizer = get_optimizer(self.config, self.model.parameters())
        lr_scheduler, extra_kwargs = get_scheduler(self.config, optimizer) # Polynomial scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, **extra_kwargs},
        }

    def validate_config(self):
        assert os.path.isdir(self.config.checkpointing.save_dir), "invalid checkpointing path"
        assert self.config.training.mode in ["train", "test", "resume_from_checkpoint"], "invalid mode"

    def get_state_dict(self, ckpt_path):
        def remove_model_prefix(state_dict):
            for k, v in state_dict.items():
                if "model." in k:
                    k.replace('model.', '')
            return state_dict  

        checkpoint = torch.load(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = checkpoint.get("state_dict", checkpoint)

        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = remove_model_prefix(state_dict)
        
        return state_dict

    def cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()