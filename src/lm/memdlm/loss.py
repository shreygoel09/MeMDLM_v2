import torch
import torch.nn as nn
import torch.nn.functional as F

# Ignore file

class RDMCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index):
        self.ignore_index = ignore_index

    def forward(self,
                scores: torch.Tensor,
                target: torch.Tensor,
                label_mask,
                weights,
                ) -> torch.Tensor:
        """
        Computes the RDM-derived loss (weighted cross-entropy).
        """

        sample_size = target.ne(self.ignore_index).float().sum()

        lprobs = F.log_softmax(scores, dim=-1)

        loss = lprobs * weights
        fullseq_loss = loss.sum() / sample_size

        # use coord masked loss for model training,
        # ignoring those position with missing coords (as nan)
        label_mask = label_mask.float()
        sample_size = label_mask.sum()  # sample size should be set to valid coordinates
        loss = (loss * label_mask).sum() / sample_size

        ppl = torch.exp(loss)
        
        logging_output = {
            'ppl': ppl.data,
            'fullseq_loss': fullseq_loss.data,
            'weight_diff_loss': loss.data
        }

        return logging_output