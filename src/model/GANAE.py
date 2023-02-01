import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from src.model.base_model import Encoder, Decoder, Autoencoder
from src.utils.utils import LayerNorm
from torchvision.models import resnet18


class GANEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.model = resnet18()
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class GANDecoder(Decoder):
    def __init__(self, bw=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=(1, 1), bias=False),
            LayerNorm(256, data_format="channels_first"),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=(3,3), stride=(2, 2), padding=(1, 1), bias=False),
            LayerNorm(128, data_format="channels_first"),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=(2, 2), padding=(1, 1), bias=False),
            LayerNorm(64, data_format="channels_first"),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=(4,4), stride=(2, 2), padding=(1, 1), bias=False),
            LayerNorm(32, data_format="channels_first"),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 24, kernel_size=(4,4), stride=(2, 2), padding=(1, 1), bias=False),
            LayerNorm(24, data_format="channels_first"),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 16, kernel_size=(4,4), stride=(2, 2), padding=(1, 1), bias=False),
            LayerNorm(16, data_format="channels_first"),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3 if not bw else 1, kernel_size=(4,4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Tanh() if not bw else nn.Sigmoid()
        )

    def forward(self, x, *args):
        x = x.view(x.shape[0], -1, 1, 1)
        return self.model(x)


class GANAE(Autoencoder):
    def __init__(self, bw=False):
        encoder = GANEncoder()
        decoder = GANDecoder(bw)
        super().__init__(encoder, decoder)

        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            init.normal_(layer.weight, 0, 0.02)


class Descriminator(nn.Module):
    def __init__(self, bw=False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3 if not bw else 1, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=(4, 4), stride=(2, 2), bias=False),
            nn.Sigmoid()
        )

        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            init.normal_(layer.weight, 0, 0.02)

    def forward(self, x):
        return self.main(x).view(x.shape[0], -1)

