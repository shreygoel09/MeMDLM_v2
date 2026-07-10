#!/usr/bin/env python3

import os
import wandb
import lightning.pytorch as pl

from omegaconf import OmegaConf
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.utils.model_utils import _print
from src.utils.config_utils import load_config
from src.guidance.solubility.solubility_module import SolubilityClassifier
from src.guidance.solubility.dataloader import MembraneDataModule, get_datasets


config = load_config("solubility.yaml")
wandb.login()

# data
datasets = get_datasets(config)
data_module = MembraneDataModule(
    config=config,
    train_dataset=datasets['train'],
    val_dataset=datasets['val'],
    test_dataset=datasets['test'],
)

# wandb logging
#wandb.init(project=config.wandb.project, name=config.wandb.name)
wandb_logger = WandbLogger(**config.wandb)

# lightning checkpoints
lr_monitor = LearningRateMonitor(logging_interval="step")
checkpoint_callback = ModelCheckpoint(
    monitor="val/loss",
    save_top_k=1,
    mode="min",
    dirpath=config.checkpointing.save_dir,
    filename="best_model",
)

# lightning trainer
trainer = pl.Trainer(
    max_steps=config.training.max_steps,
    accelerator="cuda",
    devices=1, #config.training.devices if config.training.mode=='train' else [0],
    #strategy=DDPStrategy(find_unused_parameters=True),
    callbacks=[checkpoint_callback, lr_monitor],
    logger=wandb_logger,
    log_every_n_steps=config.training.log_every_n_steps
)

# Folder to save checkpoints
ckpt_dir = config.checkpointing.save_dir
os.makedirs(ckpt_dir, exist_ok=True)

# instantiate model
model = SolubilityClassifier(config)

# train or evalute the model
if config.training.mode == "train":
    trainer.fit(model, datamodule=data_module)

elif config.training.mode == "test":
    ckpt_path = os.path.join(ckpt_dir, "best_model.ckpt")
    state_dict = model.get_state_dict(ckpt_path)
    model.load_state_dict(state_dict)
    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
else:
    raise ValueError(f"{config.training.mode} is invalid. Must be 'train' or 'test'")

wandb.finish()
