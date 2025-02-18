import torch
from torch import nn

class StyleDiversityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, styled_batch_1:torch.Tensor, styled_batch_2:torch.Tensor)->dict:
        return {
            'loss': self.loss(styled_batch_1, styled_batch_2)
        }