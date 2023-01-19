import torch.nn as nn
import torch.nn.init as init

from src.model.base_model import Autoencoder
from src.model.mae.encoder import MAEEncoder
from src.model.mae.decoder import MAEDecoder


class MAE(Autoencoder):
    def __init__(self, img_size, patch_size, in_channels, inner_dim, hidden_dim, patch_ratio, encoder_n_layers,
                 encoder_n_heads, decoder_n_layers, decoder_n_heads):
        encoder = MAEEncoder(img_size, patch_size, in_channels, inner_dim, hidden_dim, encoder_n_layers,
                             encoder_n_heads, patch_ratio)
        decoder = MAEDecoder(img_size, patch_size, inner_dim, hidden_dim, decoder_n_layers, decoder_n_heads)
        super().__init__(encoder, decoder)
