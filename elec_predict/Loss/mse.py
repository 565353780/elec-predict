import torch
from torch.nn import functional as F

class MSE:

    def __call__(self, input, target, weight=None):
        if weight is None:
            loss = F.mse_loss(input, target, reduction='mean')
        else:
            ret = F.mse_loss(input, target, reduction='none')
            loss = torch.mean(ret * weight)
        return loss


