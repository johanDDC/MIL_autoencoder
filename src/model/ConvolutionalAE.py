import math
import torch.nn as nn
import torch.nn.init as init

from src.model.base_model import Autoencoder, Encoder, Decoder
from src.utils.utils import LayerNorm


class CNNEncoder(Encoder):
    def __init__(self, n_layers, upscale_factor, in_channels,
                 start_num_filters, negative_slope, **kwargs):
        super().__init__()
        kernel_sz = kwargs.get("kernel_size", 4)
        stride = kwargs.get("stride", 2)
        pad = kwargs.get("padding", 1)

        layers = nn.ModuleList([nn.Conv2d(in_channels, start_num_filters,
                                          kernel_size=kernel_sz, stride=stride, padding=pad),
                                nn.LeakyReLU(negative_slope, inplace=True)])
        current_dim = start_num_filters
        for _ in range(n_layers - 1):
            layers.extend([nn.Conv2d(current_dim, current_dim * upscale_factor,
                                     kernel_size=kernel_sz, stride=stride, padding=pad),
                           nn.LeakyReLU(negative_slope, inplace=True),
                           LayerNorm(current_dim * upscale_factor, data_format="channels_first")])
            current_dim *= upscale_factor

        self.encoder = nn.Sequential(*layers, nn.Flatten())

    def forward(self, x):
        return self.encoder(x)


class CNNDecoder(Decoder):
    def __init__(self, n_layers, downscale_factor, out_channels, inner_channels, **kwargs):
        super().__init__()
        kernel_sz = kwargs.get("kernel_size", 4)
        stride = kwargs.get("stride", 2)
        pad = kwargs.get("padding", 1)
        self.inner_channels = inner_channels

        layers = nn.ModuleList()
        current_dim = inner_channels
        for _ in range(n_layers - 1):
            layers.extend([nn.ConvTranspose2d(current_dim, current_dim // downscale_factor,
                                     kernel_size=kernel_sz, stride=stride, padding=pad),
                           nn.ReLU(inplace=True),
                           LayerNorm(current_dim // downscale_factor, data_format="channels_first")])
            current_dim //= downscale_factor

        self.decoder = nn.Sequential(*layers,
                                     nn.ConvTranspose2d(current_dim, out_channels,
                                               kernel_size=kernel_sz, stride=stride, padding=pad),
                                     nn.Tanh())

    def forward(self, x, *args):
        x = x.reshape(x.shape[0], self.inner_channels, -1)
        flat_sz = x.shape[2]
        sz = int(math.sqrt(flat_sz))
        x = x.reshape(*x.shape[:2], sz, sz)
        return self.decoder(x)


class CNNAutoencoder(Autoencoder):
    """
        Simple cnn-based autoencoder architecture
    """

    def __init__(self, encoder_n_layers, encoder_scale_factor, in_channels,
                 start_num_filters, decoder_n_layers, decoder_scale_factor, negative_slope=0.2, **kwargs):
        encoder = CNNEncoder(encoder_n_layers, encoder_scale_factor, in_channels,
                             start_num_filters, negative_slope, **kwargs)
        inner_channels = start_num_filters * encoder_scale_factor ** (encoder_n_layers - 1)
        decoder = CNNDecoder(decoder_n_layers, decoder_scale_factor, in_channels, inner_channels, **kwargs)
        super().__init__(encoder, decoder)
        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_normal_(layer.weight)
            init.constant_(layer.bias, 0)
