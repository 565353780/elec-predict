import torch
from torch.nn import functional as F


class RMSE:

    def __call__(self, input, target, weight=None):
        if weight is None:
            ret = F.mse_loss(input, target, reduction='mean')
        else:
            ret = F.mse_loss(input, target, reduction='none') * weight
        ret[ret == 0.] = 1e-6
        loss = torch.sqrt(torch.mean(ret))
        return loss



