import torch
from torch import nn


class Embeddings(nn.Module):

    """
    References:
      embedding weight initialize https://arxiv.org/pdf/1711.09160.pdf
    """

    def __init__(self, embeds_size=None, seq_last=False):
        super().__init__()
        self.embeds_size = embeds_size
        self.embeddings = nn.ModuleList([nn.Embedding(i, o) for i, o in embeds_size]) if embeds_size else None
        self.seq_last = seq_last
        self.output_size = sum([i for _, i in embeds_size]) if embeds_size is not None else 0

    def forward(self, inputs=None):
        if inputs is None:
            return None
        if self.seq_last:
            embed = torch.cat(
                [self.embeddings[d](inputs[:, d]).transpose(1, 2) for d in range(inputs.shape[1])], dim=1)
        else:
            embed = torch.cat(
                [self.embeddings[d](inputs[:, :, d]) for d in range(inputs.shape[2])], dim=2)
        return embed

    def reset_parameters(self):
        if self.embeds_size:
            for layer in self.embeddings:
                nn.init.xavier_normal_(layer.weight)


