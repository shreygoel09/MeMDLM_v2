import gc
import torch
import torch.nn as nn
import lightning.pytorch as pl

from omegaconf import OmegaConf
from transformers import AutoModel
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy

from src.utils.model_utils import _print
from src.guidance.solubility.utils import CosineWarmup


config = OmegaConf.load("/scratch/pranamlab/sgoel/MeMDLM_v2/src/configs/solubility.yaml")

class SolubilityClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.auroc = BinaryAUROC()
        self.accuracy = BinaryAccuracy()

        self.esm_model = AutoModel.from_pretrained(self.config.lm.pretrained_esm)
        for p in self.esm_model.parameters():
            p.requires_grad = False

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
    def forward(self, batch):
        if 'input_ids' in batch:
            esm_embeds = self.get_esm_embeddings(batch['input_ids'], batch['attention_mask'])
        elif 'embeds' in batch:
            esm_embeds = batch['embeds']
        encodings = self.encoder(esm_embeds, src_key_padding_mask=(batch['attention_mask'] == 0))
        encodings = self.dropout(self.layer_norm(encodings))
        logits = self.mlp(encodings).squeeze(-1)
        return logits

    
    # -------# Training / Evaluation #-------- #
    def training_step(self, batch, batch_idx):
        train_loss, _ = self.compute_loss(batch)
        self.log(name="train/loss", value=train_loss.item(), on_step=True, on_epoch=False, logger=True, sync_dist=True)
        self.save_ckpt()
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss, _ = self.compute_loss(batch)
        self.log(name="val/loss", value=val_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return val_loss

    def test_step(self, batch):
        test_loss, preds = self.compute_loss(batch)
        auroc, accuracy = self.get_metrics(batch, preds)
        self.log(name="test/loss", value=test_loss.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log(name="test/AUROC", value=auroc.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log(name="test/accuracy", value=accuracy.item(), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return test_loss

    def on_test_epoch_end(self):
        self.auroc.reset()
        self.accuracy.reset()
    
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
    @torch.no_grad
    def get_esm_embeddings(self, input_ids, attention_mask):
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        return embeddings

    def compute_loss(self, batch):
        """Helper method to handle loss calculation"""
        labels = batch['labels']
        preds = self.forward(batch)
        loss = self.loss_fn(preds, labels)
        loss_mask = (labels != self.config.model.label_pad_value) # only calculate loss over non-pad tokens
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        return loss, preds

    def get_metrics(self, batch, preds):
        """Helper method to compute metrics"""
        labels = batch['labels']

        valid_mask = (labels != self.config.model.label_pad_value)
        labels = labels[valid_mask]
        preds = preds[valid_mask]

        _print(f"labels {labels.shape}")
        _print(f"preds {preds.shape}")

        auroc = self.auroc.forward(preds, labels)
        accuracy = self.accuracy.forward(preds, labels)
        return auroc, accuracy

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