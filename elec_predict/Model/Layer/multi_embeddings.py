import torch
import torch.nn as nn


class MultiEmbeddings(nn.Module):

    def __init__(self, *variable_params):
        # example: *[(name, num_embeddings, embedding_dim), ... ]
        super().__init__()
        self.params = variable_params
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(s, e) for (name, s, e) in variable_params
        })

    def forward(self, input):
        return torch.cat([self.embeddings[name](input[name]) for (name, _, _) in self.params], dim=2)


