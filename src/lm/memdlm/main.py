import os
import gc
import sys
import torch
import wandb
import torch.nn as nn
import lightning.pytorch as pl

from omegaconf import OmegaConf
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.lm.memdlm.diffusion_module import MembraneDiffusion
from src.lm.memdlm.dataloader import MembraneDataModule, get_datasets
from src.utils.model_utils import apply_rdm_freezing
from src.utils.config_utils import load_config

wandb.login()


# Load yaml config
config = load_config("lm.yaml")

# Get datasets
datasets = get_datasets(config)
data_module = MembraneDataModule(
    config=config,
    train_dataset=datasets['train'],
    val_dataset=datasets['val'],
    test_dataset=datasets['test'],
)

# Initialize WandB for logging
wandb.init(project=config.wandb.project, name=config.wandb.name)
wandb_logger = WandbLogger(**config.wandb)

# PL checkpoints
lr_monitor = LearningRateMonitor(logging_interval="step")
checkpoint_callback = ModelCheckpoint(
    monitor="val/loss",
    save_top_k=1,
    mode="min",
    dirpath=config.checkpointing.save_dir,
    filename="best_model",
    every_n_train_steps=config.checkpointing.save_every_n_steps
)

# PL trainer
trainer = pl.Trainer(
    max_steps=config.training.max_steps,
    max_epochs=None,  # Ensure training is based on num steps
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
    devices=config.training.devices if config.training.mode=='train' else [0],
    strategy=DDPStrategy(find_unused_parameters=True),
    callbacks=[checkpoint_callback, lr_monitor],
    logger=wandb_logger,
    log_every_n_steps=config.training.log_every_n_steps
)


# Folder to save checkpoints
ckpt_path = config.checkpointing.save_dir
try: os.makedirs(ckpt_path, exist_ok=False)
except FileExistsError: pass 

# PL Model for training
diffusion = MembraneDiffusion(config)
diffusion.validate_config()

# Start/resume training or evaluate the model
model_type = "evoflow"
if config.training.mode == "train":
    apply_rdm_freezing(diffusion.model, config.training.n_layers, model_type)
    trainer.fit(diffusion, datamodule=data_module)

elif config.training.mode == "test":
    state_dict = diffusion.get_state_dict(config.checkpointing.best_ckpt_path)
    diffusion.load_state_dict(state_dict)
    trainer.test(diffusion, datamodule=data_module, ckpt_path=config.checkpointing.best_ckpt_path)

elif config.training.mode == "resume_from_checkpoint":
    state_dict = diffusion.get_state_dict(config.training.resume_ckpt_path)
    diffusion.load_state_dict(state_dict)
    apply_rdm_freezing(diffusion.model, config.training.n_layers, model_type)
    trainer.fit(diffusion, datamodule=data_module, ckpt_path=ckpt_path)

wandb.finish()

