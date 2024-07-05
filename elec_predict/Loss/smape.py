import torch


class SMAPE:

    def __call__(self, input, target, weight=None):
        mae = torch.abs(input - target)
        divide = input + target
        divide[divide == 0.] = 1e-6
        smape = mae / divide
        if weight is not None:
            smape *= weight
        return torch.mean(smape)


