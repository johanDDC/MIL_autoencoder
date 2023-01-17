import torch
import torch.nn as nn

from src.model.base_model import Decoder
from src.model.unet.conv_block import ConvBlock


class UpsamplingBlock(nn.Module):
    def __init__(self, n_blocks, in_channels, out_channels, scale_factor, kernel_size):
        super().__init__()

        self.input_ch, self.output_ch = in_channels, out_channels
        self.upsampler = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=(scale_factor, scale_factor),
                                            stride=(scale_factor, scale_factor))
        self.conv = ConvBlock(n_blocks, in_channels, out_channels, kernel_size)

    def forward(self, x, encoder_state):
        x = self.upsampler(x)
        x = torch.cat([x, encoder_state], dim=1)
        return self.conv(x)


class UNetDecoder(Decoder):
    def __init__(self, n_layers, layer_width, output_channels, channels_scale_factor,
                 resolution_scale_factor, end_num_filters, kernel_size=3):
        """
        UNet encoder layer

        :param n_layers: number of convolutional blocks of encoder
        :param layer_width: number of convolutions of each convolutional block
        :param channels_scale_factor: scale factor for number of channels
        :param resolution_scale_factor: scale factor for feature map resolution
        """
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([nn.Tanh(),
                                     ConvBlock(layer_width, end_num_filters, output_channels, 1)])

        cur_channels = end_num_filters
        for _ in range(n_layers - 1):
            self.layers.append(
                UpsamplingBlock(layer_width, cur_channels, cur_channels * channels_scale_factor,
                                resolution_scale_factor, kernel_size)
            )
            cur_channels *= channels_scale_factor

        self.layers = self.layers[::-1]

    def forward(self, x, *args):
        if len(args) <= 0:
            raise ValueError("UNet decoder didn't get residuals")
        residuals = args[0][::-1]
        for i in range(self.n_layers):
            x = self.layers[i](x, residuals[i])
        return x
