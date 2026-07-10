import gc
import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F

from omegaconf import OmegaConf

from src.utils.model_utils import _print
from src.guidance.solubility.utils import CosineWarmup

from sklearn.metrics import roc_auc_score, accuracy_score


config = OmegaConf.load("/scratch/pranamlab/sgoel/MeMDLM_v2/src/configs/multipass.yaml")


class MultipassClassifier(pl.LightningModule):
    def __init__(self, config, diffusion_model):
        super().__init__()
        self.config = config
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.all_preds = []
        self.all_labels = []

        self.diffusion_model = diffusion_model
        for p in self.diffusion_model.model.parameters():
            p.requires_grad = False
        self.diffusion_model.eval()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.d_model,
            nhead=config.model.num_heads,
            dropout=config.model.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.model.num_layers)
        self.layer_norm = nn.LayerNorm(config.model.d_model)
        self.dropout = nn.Dropout(config.model.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(config.model.d_model, config.model.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.d_model // 2, 1),
        )


    # -------# Classifier step #-------- #
    def forward(self, x_t, attn_mask, embeds=None, with_hidden=None):
        if embeds is None:
            with torch.no_grad():
                embeds = self.diffusion_model.forward(
                    input_ids=x_t,
                    attention_mask=attn_mask,
                    return_hidden=True
                )
        else:
            assert with_hidden is not None

        encodings = self.encoder(embeds, src_key_padding_mask=(attn_mask== 0))
        encodings = self.dropout(self.layer_norm(encodings))
        mask = attn_mask.unsqueeze(-1)
        pooled = (encodings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.mlp(pooled).squeeze(-1)
        return logits

    def step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        t1 = self.diffusion_model.sample_t(input_ids)
        maskable = self.diffusion_model.is_maskable(input_ids)

        x_t, _ = self.diffusion_model.noise_x0(input_ids, t1, maskable_mask=maskable)
        
        logits = self.forward(x_t, attention_mask)
        loss = self.compute_loss(logits, labels)
        
        return loss, logits

    # -------# Training / Evaluation #-------- #
    def training_step(self, batch, batch_idx):
        train_loss, _ = self.step(batch)
        self.log(name="train/loss", value=train_loss.item(), on_step=True, on_epoch=False, logger=True, sync_dist=True)
        self.save_ckpt()
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss, _ = self.step(batch)
        self.log(name="val/loss", value=val_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return val_loss

    def test_step(self, batch):
        test_loss, logits = self.step(batch)
        preds = F.sigmoid(logits)
        self.all_preds.append(preds.detach().cpu())
        self.all_labels.append(batch['labels'].detach().cpu())
        self.log(name="test/loss", value=test_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return test_loss

    def on_test_epoch_start(self):
        self.all_preds = []
        self.all_labels = []

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.all_preds).numpy()
        all_labels = torch.cat(self.all_labels).numpy()
        
        auroc = roc_auc_score(all_labels, all_preds)
        binaries = (all_preds > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, binaries)

        self.log(name="test/AUROC", value=auroc, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log(name="test/accuracy", value=accuracy, on_step=False, on_epoch=True, logger=True, sync_dist=True)
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        path = self.config.training
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.optim.lr)
        lr_scheduler = CosineWarmup(
            optimizer,
            warmup_steps=path.warmup_steps,
            total_steps=path.max_steps,
        )
        scheduler_dict = {
            "scheduler": lr_scheduler,
            "interval": 'step',
            'frequency': 1,
            'monitor': 'val/loss',
            'name': 'learning_rate'
        }
        return [optimizer], [scheduler_dict]
    
    def save_ckpt(self):
        curr_step = self.global_step
        save_every = self.config.training.val_check_interval
        if curr_step % save_every == 0 and curr_step > 0:  # Save every 250 steps
            ckpt_path = f"{self.config.checkpointing.save_dir}/step={curr_step}.ckpt"
            self.trainer.save_checkpoint(ckpt_path)
    
    # -------# Loss and Test Set Metrics #-------- #
    def compute_loss(self, logits, labels):
        """Helper method to handle loss calculation"""
        loss = self.loss_fn(logits, labels.float()).mean()
        return loss

    # -------# Helper Functions #-------- #
    def get_state_dict(self, ckpt_path):
        """Helper method to load and process a trained model's state dict from saved checkpoint"""
        def remove_model_prefix(state_dict):
            for k in state_dict.keys():
                if "model." in k:
                    k.replace('model.', '')
            return state_dict  

        checkpoint = torch.load(ckpt_path, weights_only=False)#, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = checkpoint.get("state_dict", checkpoint)

        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = remove_model_prefix(state_dict)
        
        return state_dict