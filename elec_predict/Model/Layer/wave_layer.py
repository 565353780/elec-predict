import torch
from torch import nn
from torch.nn import functional as F

from elec_predict.Model.Layer.causal_conv1d import CausalConv1d


class WaveLayer(nn.Module):

    def __init__(self, residual_channels, skip_channels, dilation):
        super(WaveLayer, self).__init__()
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilation = dilation
        self.conv_dilation = CausalConv1d(residual_channels, residual_channels, dilation=dilation)
        self.conv_filter = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.conv_gate = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        self.conv_residual = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)

    def forward(self, x):
        """
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
        """
        x_dilation = self.conv_dilation(x)
        x_filter = self.conv_filter(x_dilation)
        x_gate = self.conv_gate(x_dilation)
        x_conv = torch.sigmoid(x_gate) * torch.tanh(x_filter)
        x_skip = self.conv_skip(x_conv)
        x_res = self.conv_residual(x_conv) + x_dilation
        return x_res, x_skip

    def last_forward(self, x):
        x_dilation = F.conv1d(x, self.conv_dilation.weight, self.conv_dilation.bias,
                              self.conv_dilation.stride, 0, dilation=1, groups=self.conv_dilation.groups)
        x_filter = self.conv_filter(x_dilation)
        x_gate = self.conv_gate(x_dilation)
        x_conv = torch.sigmoid(x_gate) * torch.tanh(x_filter)
        x_skip = self.conv_skip(x_conv)
        x_res = self.conv_residual(x_conv) + x_dilation
        return x_res, x_skip


