import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, n_blocks, in_channels, out_channels, kernel_size):
        super().__init__()

        layers = nn.ModuleList()
        for i in range(n_blocks):
            if i == 0:
                layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
                )
            else:
                layers.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding="same")
                )
            layers.extend([nn.LayerNorm(out_channels), nn.ReLU])

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block(x)
        return x
