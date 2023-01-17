import torch.nn as nn
import torch.nn.init as init

from src.model.base_model import Autoencoder
from src.model.unet.decoder import UNetDecoder
from src.model.unet.encoder import UNetEncoder


class UNet(Autoencoder):
    def __init__(self, n_layers, layer_width, in_channels, channels_scale_factor,
                 resolution_scale_factor, start_num_filters, kernel_size):
        """
        UNet-based autoencoder

        :param n_layers: number of convolutional blocks of encoder
        :param layer_width: number of convolutions of each convolutional block
        :param channels_scale_factor: scale factor for number of channels
        :param resolution_scale_factor: scale factor for feature map resolution
        """
        encoder = UNetEncoder(n_layers, layer_width, in_channels, channels_scale_factor,
                              resolution_scale_factor, start_num_filters, kernel_size)
        decoder = UNetDecoder(n_layers, layer_width, in_channels, channels_scale_factor,
                              resolution_scale_factor, start_num_filters, kernel_size)
        super().__init__(encoder, decoder)
        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_normal(layer.weight)
            init.constant_(layer.bias, 0)
