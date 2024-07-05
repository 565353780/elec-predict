from torch import nn

from elec_predict.Model.Layer.wave_states import WaveStates
from elec_predict.Model.Layer.wave_layer import WaveLayer

class WaveNet(nn.Module):

    def __init__(self, input_channels, residual_channels, skip_channels, num_blocks, num_layers, mode="add"):
        super(WaveNet, self).__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.input_conv = nn.Conv1d(input_channels, residual_channels, kernel_size=1)
        self.mode = mode
        self.wave_layers = nn.ModuleList([WaveLayer(residual_channels, skip_channels, 2 ** i)
                                          for _ in range(num_blocks) for i in range(num_layers)])

    def encode(self, x):
        state = WaveStates(self.num_blocks, self.num_layers)
        x = self.input_conv(x)
        skips = 0.
        for i, layer in enumerate(self.wave_layers):
            state.init(i, x)
            x, skip = layer(x)
            skips += skip
        return skips, state

    def decode(self, x, state):
        x = self.input_conv(x)
        skips = 0.
        for i, layer in enumerate(self.wave_layers):
            state.enqueue(i, x)
            x = state.dequeue(i)
            x, skip = layer.last_forward(x)
            skips += skip
        return skips, state
