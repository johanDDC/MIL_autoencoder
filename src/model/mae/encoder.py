import torch
import torch.nn as nn

from transformers import ViTModel, ViTConfig
from torch import Tensor

from src.model.base_model import Encoder
from src.model.mae.patch_embeddings import PatchEmbeddings


class MAEEncoder(Encoder):
    def __init__(self, img_size, patch_size, in_channels, inner_dim, hidden_dim, n_layers, n_heads,
                 patch_ratio):
        """
        hidden_dim: dimension of latent representation
        inner_dim: dimension of inner representation (between attention and add-norm layers)
        """
        super().__init__()
        self.patch_ratio = patch_ratio

        vit_config = ViTConfig(hidden_size=hidden_dim, num_hidden_layers=n_layers,
                               num_attention_heads=n_heads, intermediate_size=inner_dim, layer_norm_eps=1e-7)

        self.patchify = nn.Conv2d(in_channels, hidden_dim, kernel_size=(patch_size, patch_size),
                                  stride=(patch_size, patch_size))
        self.patch_embeddings = PatchEmbeddings(img_size, patch_size, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, inner_dim))

        self.vit = ViTModel(vit_config).encoder

        nn.init.xavier_normal_(self.cls.data)

    def __shuffle(self, patches: Tensor):
        shuffle_ids = torch.stack([torch.randperm(patches.shape[-1])])
        unshuffle_ids = torch.argsort(shuffle_ids, dim=0)

        shuffled_patches = patches.gather(0, shuffle_ids.expand(1, *shuffle_ids.shape).permute(1, 0, 2).repeat(1, 3, 1))
        return shuffled_patches[:int(patches.shape[-1] * (1 - self.patch_ratio))], unshuffle_ids

    def forward(self, x):
        patches = self.patchify(x)
        patches = self.patch_embeddings(patches)
        patches, unshuffle_ids = self.__shuffle(patches)
        patches = torch.cat([self.cls.expand(-1, patches.shape[1], -1), patches], dim=0)
        return self.vit(patches), unshuffle_ids



