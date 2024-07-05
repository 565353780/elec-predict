import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.cat([i for i in x if i is not None], self.dim)
