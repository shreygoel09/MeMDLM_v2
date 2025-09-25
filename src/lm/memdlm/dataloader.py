import torch
import pandas as pd
import lightning.pytorch as pl

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class MembraneDataset(Dataset):
    def __init__(self, config, data_path):
        self.config = config
        self.data = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm.pretrained_evoflow)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]["Sequence"]

        tokens = self.tokenizer(
            sequence.upper(),
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.config.data.max_seq_len
        )

        #return {"input_ids": tokens['input_ids'], "attention_mask": tokens['attention_mask']}

        return {
            "input_ids": tokens['input_ids'].squeeze(0),
            "attention_mask": tokens['attention_mask'].squeeze(0)
        }


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])#.squeeze()
    masks = torch.stack([item['attention_mask'] for item in batch])#.squeeze()

    return {'input_ids': input_ids, 'attention_mask': masks}


class MembraneDataModule(pl.LightningDataModule):
    def __init__(self, config, train_dataset, val_dataset, test_dataset, collate_fn=collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn
        self.batch_size = config.data.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm.pretrained_evoflow)

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
                          shuffle=False,
                          pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=8,
                          shuffle=False,
                          pin_memory=True)
    

def get_datasets(config):
    """Helper method to grab datasets to quickly init data module in main.py"""
    train_dataset = MembraneDataset(config, config.data.train)
    test_dataset = MembraneDataset(config, config.data.test)
    val_dataset = MembraneDataset(config, config.data.val)
    
    return  {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }