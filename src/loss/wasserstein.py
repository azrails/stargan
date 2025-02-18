import torch
from torch import nn

class WassersteinAdversarialLoss(nn.Module):
    """
    Wasserstein gan base adversarial loss.
    For more details https://en.wikipedia.org/wiki/Wasserstein_GAN
    """
    def __init__(self):
        super().__init__()

    def forward(self, fake_logits: torch.Tensor, real_logits:torch.Tensor|None = None):
        return {
            'loss': \
                torch.mean(0 if real_logits is None else real_logits) - torch.mean(fake_logits)
            }