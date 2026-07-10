
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler



class HelixRoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        pos = torch.arange(config.data.max_seq_len).float()
        thetas = (2 * torch.pi / 3.6) * pos 
        self.register_buffer("thetas", thetas)

    def forward(self, x, mask):
        B, L, D = x.shape
        assert D % 2 == 0

        thetas = self.thetas[:L]
        cos = torch.cos(thetas).unsqueeze(0).unsqueeze(-1)
        sin = torch.sin(thetas).unsqueeze(0).unsqueeze(-1)

        x_double = x.view(B, L, D//2, 2)
        x1 = x_double[..., 0]
        x2 = x_double[..., 1]

        r1 = cos * x1 - sin * x2
        r2 = sin * x1 + cos * x2

        ropes = torch.stack([r1, r2], dim=-1).view(B, L, D)
        return ropes * mask  # attention mask to ignore pad tokens


class CosineWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_ratio=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_ratio = eta_ratio  # The ratio of minimum to maximum learning rate
        super(CosineWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        decayed_lr = (1 - self.eta_ratio) * cosine_decay + self.eta_ratio

        return [decayed_lr * base_lr for base_lr in self.base_lrs]