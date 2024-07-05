import torch


class RNNActivationLoss:

    """
    RNN outputs -> loss
    """

    def __init__(self, beta=1e-5):
        self.beta = beta

    def __call__(self, rnn_output):
        if self.beta == .0:
            return .0
        return torch.sum(torch.norm(rnn_output)) * self.beta
