import torch.nn as nn

from src.model.base_model import Autoencoder, Encoder, Decoder


class FCEncoder(Encoder):
    def __init__(self, n_layers, downscale_factor, negative_slope):
        super().__init__()
        layers = nn.ModuleList()
        current_dim = self.input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(current_dim, current_dim // downscale_factor),
                           nn.LeakyReLU(negative_slope, inplace=True)])
            current_dim //= downscale_factor

        self.encoder = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x):
        return self.encoder(x)


class FCDecoder(Decoder):
    def __init__(self, n_layers, upscale_factor):
        super().__init__()
        layers = nn.ModuleList()
        current_dim = self.output_dim
        for _ in range(n_layers):
            layers.extend([nn.ReLU(inplace=True),
                           nn.Linear(current_dim, current_dim // upscale_factor)])
            current_dim //= upscale_factor

        self.decoder = nn.Sequential(*layers[::-1])

    def forward(self, x, *args):
        return self.decoder(x)

class FCAutoencoder(Autoencoder):
    """
        Simple fully-connected autoencoder architecture
    """

    def __init__(self, n_layers, scale_factor, negative_slope=0.2):
        '''
        param n_components: dim of latent vectors
        '''
        encoder = FCEncoder(n_layers, scale_factor, negative_slope)
        decoder = FCDecoder(n_layers, scale_factor)
        super().__init__(encoder, decoder)
