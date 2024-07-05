import torch


class WaveQueue:

    def __init__(self, dilation):
        self.max_len = dilation + 1  # kernel size is 2
        self.values = None

    def init(self, x):
        self.values = torch.zeros(x.shape[0], x.shape[1], self.max_len).to(x.device)
        self.values[:, :, -min(self.max_len, x.shape[2]):] = x[:, :, -min(self.max_len, x.shape[2]):]

    def clear_buffer(self):
        self.values = None

    def enqueue(self, x):
        assert x.shape[2] == 1
        self.values = torch.cat([self.values[:, :, 1:], x], dim=2)

    def dequeue(self):
        return self.values[:, :, [-self.max_len, -1]]
