import torch.nn as nn


class Empty(nn.Module):

    def __init__(self, size):
        self.size = size
        super().__init__()

    def forward(self, x):
        return x

    def extra_repr(self):
        return f"{self.size}"


