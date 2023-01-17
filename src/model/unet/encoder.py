import torch.nn as nn

from src.model.base_model import Encoder
from src.model.unet.conv_block import ConvBlock


class DownsamplingBlock(nn.Module):
    def __init__(self, n_blocks, in_channels, out_channels, scale_factor, kernel_size):
        super().__init__()

        self.layer = nn.Sequential(
            ConvBlock(n_blocks, in_channels, out_channels, kernel_size),
            nn.MaxPool2d(scale_factor)
        )

    def forward(self, x):
        return self.layer(x)


class UNetEncoder(Encoder):
    def __init__(self, n_layers, layer_width, input_channels, channels_scale_factor,
                 resolution_scale_factor, start_num_filters, kernel_size=3):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([ConvBlock(layer_width, input_channels, start_num_filters, 1)])

        cur_channels = start_num_filters
        for _ in range(n_layers - 1):
            self.layers.append(
                DownsamplingBlock(layer_width, cur_channels, cur_channels * channels_scale_factor,
                                  resolution_scale_factor, kernel_size)
            )
            cur_channels *= channels_scale_factor

    def forward(self, x):
        residuals = []
        for i in range(self.n_layers):
            x = self.layers[i](x)
            residuals.append(x)
        return x, residuals
