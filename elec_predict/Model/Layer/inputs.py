import torch
import torch.nn as nn

from elec_predict.Model.Layer.multi_embeddings import MultiEmbeddings
from elec_predict.Model.Layer.empty import Empty

class Inputs(nn.Module):

    def __init__(self, inputs_config=None):
        super().__init__()
        self.inputs_config = inputs_config
        if inputs_config is not None:
            self.numerical = inputs_config.get("numerical")
            self.categorical = inputs_config.get("categorical")
            self.output_size = 0
            if self.categorical is not None:
                self.categorical_inputs = MultiEmbeddings(*self.categorical)
                self.output_size += sum([i[2] for i in self.categorical])

            if self.numerical is not None:
                self.numerical_inputs = nn.ModuleDict({name: Empty(size) for (name, size) in self.numerical})
                self.output_size += sum([i[1] for i in self.numerical])
        else:
            self.output_size = 0

    def forward(self, feed_dict):
        # batch, seq, N
        if self.inputs_config is not None:
            outputs = []
            if self.categorical is not None:
                outputs.append(self.categorical_inputs(feed_dict))
            if self.numerical is not None:
                for (name, _) in self.numerical:
                    outputs.append(self.numerical_inputs[name](feed_dict[name]))
            return torch.cat(outputs, dim=2)
        else:
            return None
