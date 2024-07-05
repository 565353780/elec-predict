from torch import nn


class CausalConv1d(nn.Conv1d):
    """1D Causal Convolution Layer

    Args:
        inputs, Tensor(batch, input_unit(kernel_size), sequence)

    Returns:
        Tensor(batch, output_unit(kernel_size), sequence)
    """

    def __init__(self, in_channels, out_channels, dilation=1, bias=True,
                 padding_mode='zeros'):
        kernel_size = 2
        self.shift = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.shift,
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode=padding_mode)

    def forward(self, inputs):
        return super(CausalConv1d, self).forward(inputs)[:, :, :-self.shift]


