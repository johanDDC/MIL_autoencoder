import torch
import torch.nn as nn

from transformers import ViTModel, ViTConfig
from torch import Tensor

from src.model.base_model import Decoder
from src.model.mae.patch_embeddings import PatchEmbeddings


class MAEDecoder(Decoder):
    def __init__(self, img_size, patch_size, inner_dim, hidden_dim, n_layers, n_heads):
        super().__init__()
        vit_config = ViTConfig(hidden_size=hidden_dim, num_hidden_layers=n_layers,
                               num_attention_heads=n_heads, intermediate_size=inner_dim, layer_norm_eps=1e-7)

        self.mask = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.patch_embeddings = PatchEmbeddings(img_size, patch_size, hidden_dim, with_cls=True)

        self.vit = ViTModel(vit_config).encoder
        self.head = torch.nn.Linear(hidden_dim, patch_size ** 2 * 3)

        nn.init.xavier_normal_(self.cls.data)

    def forward(self, x, *args):
        if len(args) <= 0:
            raise ValueError("MAE decoder didn't get unshuffle indexes")

        unshuffle_ids = args[0]
        unshuffle_ids = torch.cat([torch.zeros(1, unshuffle_ids.shape[0]), unshuffle_ids + 1], dim=0)
        x = torch.cat([x, self.mask_token.expand(unshuffle_ids.shape[2] - x.shape[2], x.shape[0], -1)], dim=0)
        x = x.gather(0, unshuffle_ids.expand(1, *unshuffle_ids.shape).permute(1, 0, 2).repeat(1, 3, 1))
        x = self.patch_embeddings(x)
        x = self.vit(x)[1:]

        patches = self.head(x)
        img = None
        return img
