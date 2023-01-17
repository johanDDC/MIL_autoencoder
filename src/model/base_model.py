import torch.nn as nn


class Encoder(nn.Module):
    """
        Base encoder class
    """
    def __init__(self):
        super().__init__()
        self.input_dim = 32 * 32 * 3

    def forward(self, x):
        """
            Forward pass of the encoder
        """
        pass


class Decoder(nn.Module):
    """
        Base decoder class
    """
    def __init__(self):
        super().__init__()
        self.output_dim = 32 * 32 * 3

    def forward(self, x, *args):
        """
            Forward pass of the decoder
        """
        pass


class Autoencoder(nn.Module):
    '''
        Base autoencoder class
    '''

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        '''
        Encode batch of images
        param x: batch of input images
        return z: batch of latent vectors
        '''
        return self.encoder(x)

    def decode(self, z):
        '''
        Decode batch of latent vectors
        param z: batch of latent vectors
        return x_hat: batch of reconstructed images
        '''
        return self.decoder(z)

    def forward(self, x):
        '''
        Forward pass of the autoencoder
        param x: batch of input images
        return x_hat: batch of reconstructed images
        '''
        return self.decode(self.encode(x))