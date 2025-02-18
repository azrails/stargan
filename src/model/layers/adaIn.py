import torch
from torch import nn

class AdaIn(nn.Module):
    def __init__(
            self,
            in_channels: int,
            embedding_dim: int = 128
            ):
        super().__init__()
        self.mean_variance_projection = nn.Linear(embedding_dim, 2 * in_channels)
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)
    
    def forward(self, x, s):
        mean_variance = self.mean_variance_projection(s)
        mean_variance = torch.view(mean_variance.size(0), mean_variance.size(1), 1, 1)
        mean, variance = torch.chunk(mean_variance, chunks=2, dim=1)

        out = self.norm(x)
        out = mean + variance * out

        return out