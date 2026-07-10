import torch
import pandas as pd
import lightning.pytorch as pl

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class MembraneDataset(Dataset):
    def __init__(self, config, data_path):
        self.config = config
        self.data = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.lm.pretrained_esm)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]["Sequence"]

        tokens = self.tokenizer(
            sequence.upper(),
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.config.data.max_seq_len,
        )

        labels = self.get_labels(sequence)

        return {
            "input_ids": tokens['input_ids'],
            "attention_mask": tokens['attention_mask'],
            "labels": labels
        }

    def get_labels(self, sequence):
        max_len = self.config.data.max_seq_len

        # Create per-residue labels
        labels = torch.tensor([1 if residue.islower() else 0 for residue in sequence], dtype=torch.float)
        
        if len(labels) < max_len: # Padding if sequence shorter than tokenizer truncation length
            padded_labels = torch.cat(
                [labels, torch.full(size=(max_len - len(labels),), fill_value=self.config.model.label_pad_value)]
            )
        else: # Truncation otherwise
            padded_labels = labels[:max_len]
        return padded_labels


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'].squeeze(0) for item in batch])
    masks = torch.stack([item['attention_mask'].squeeze(0) for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': masks,
        'labels': labels
    }


class MembraneDataModule(pl.LightningDataModule):
    def __init__(self, config, train_dataset, val_dataset, test_dataset, collate_fn=collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn
        self.batch_size = config.data.batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=8,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=8,
                          pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=8,
                          pin_memory=True)
    

def get_datasets(config):
    """Helper method to grab datasets to quickly init data module in main.py"""
    esm_model = AutoModel.from_pretrained(config.lm.pretrained_esm)
    tokenizer = AutoTokenizer.from_pretrained(config.lm.pretrained_esm)

    train_dataset = MembraneDataset(config, config.data.train)
    val_dataset = MembraneDataset(config, config.data.val)
    test_dataset = MembraneDataset(config, config.data.test)
    
    return  {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }