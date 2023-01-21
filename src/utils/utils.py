import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError("This data format is not supported")
        self.normalized_shape = shape

        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)
        else:
            mean = x.mean(1, keepdim=True)
            std = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(std + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

