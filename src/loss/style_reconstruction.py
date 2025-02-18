import torch
from torch import nn

class StyleRecontructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, style_code:torch.Tensor, reconstructed_style_code:torch.Tensor)->dict:
        return {
            'loss': self.loss(style_code, reconstructed_style_code)
            }