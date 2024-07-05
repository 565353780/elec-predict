import torch


class RNNStabilityLoss:
    """

    RNN outputs -> loss

    References:
        https://arxiv.org/pdf/1511.08400.pdf
    """

    def __init__(self, beta=1e-5):
        self.beta = beta

    def __call__(self, rnn_output):
        if self.beta == .0:
            return .0
        l2 = torch.sqrt(torch.sum(torch.pow(rnn_output, 2), dim=-1))
        l2 = self.beta * torch.mean(torch.pow(l2[:, 1:] - l2[:, :-1], 2))
        return l2



