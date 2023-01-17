import torch.nn as nn
import torch.nn.init as init

from src.model.base_model import Autoencoder, Encoder, Decoder


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
        for _ in range(n_layers):
            layers.extend([nn.Conv2d(current_dim, current_dim * upscale_factor,
                                     kernel_size=kernel_sz, stride=stride, padding=pad),
                           nn.LeakyReLU(negative_slope, inplace=True),
                           nn.LayerNorm(current_dim * upscale_factor)])
            current_dim *= upscale_factor

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class CNNDecoder(Decoder):
    def __init__(self, n_layers, downscale_factor, out_channels, **kwargs):
        super().__init__()
        kernel_sz = kwargs.get("kernel_size", 4)
        stride = kwargs.get("stride", 2)
        pad = kwargs.get("padding", 1)

        layers = nn.ModuleList()
        current_dim = out_channels * downscale_factor
        for _ in range(n_layers - 1):
            layers.extend([nn.LayerNorm(current_dim * downscale_factor),
                           nn.ReLU(inplace=True),
                           nn.ConvTranspose2d(current_dim * downscale_factor, current_dim,
                                     kernel_size=kernel_sz, stride=stride, padding=pad)])
            current_dim *= downscale_factor

        self.decoder = nn.Sequential(*layers[::-1],
                                     nn.Conv2d(current_dim, out_channels,
                                               kernel_size=kernel_sz, stride=stride, padding=pad),
                                     nn.Tanh())

    def forward(self, x, *args):
        return self.decoder(x)


class CNNAutoencoder(Autoencoder):
    """
        Simple cnn-based autoencoder architecture
    """

    def __init__(self, n_layers, scale_factor, in_channels,
                 start_num_filters, negative_slope=0.2, **kwargs):
        encoder = CNNEncoder(n_layers, scale_factor, in_channels,
                             start_num_filters, negative_slope, **kwargs)
        decoder = CNNDecoder(n_layers, scale_factor, in_channels, **kwargs)
        super().__init__(encoder, decoder)
        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_normal(layer.weight)
            init.constant_(layer.bias, 0)
