import torch.nn as nn
import torch.nn.functional as F

from src.model.base_model import Encoder, Decoder, Autoencoder
from torchvision.models import resnet18


class SimpleEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.model = resnet18()
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class SimpleDecoder(Decoder):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.model = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, bias=False),
            nn.Tanh()
        )

    def forward(self, x, *args):
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.model(x)


class SimpleAE(Autoencoder):
    def __init__(self):
        encoder = SimpleEncoder()
        decoder = SimpleDecoder()
        super().__init__(encoder, decoder)
