import torch
import torch.nn as nn

from transformers import ViTModel, ViTConfig
from torch import Tensor

from src.model.base_model import Decoder
from src.model.mae.patch_embeddings import PatchEmbeddings


class MAEDecoder(Decoder):
    def __init__(self, img_size, patch_size, out_channels, inner_dim, hidden_dim, n_layers, n_heads):
        super().__init__()
        vit_config = ViTConfig(hidden_size=hidden_dim, num_hidden_layers=n_layers,
                               num_attention_heads=n_heads, intermediate_size=inner_dim, layer_norm_eps=1e-7)
        self.out_channels = out_channels
        self.img_size = img_size

        self.mask = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.patch_embeddings = PatchEmbeddings(img_size, patch_size, hidden_dim, with_cls=True)

        self.vit = ViTModel(vit_config).encoder
        self.head = torch.nn.Linear(hidden_dim, patch_size ** 2 * 3)

        self.tanh = nn.Tanh()

        nn.init.xavier_normal_(self.mask.data)

    def forward(self, x, *args):
        if len(args) <= 0:
            raise ValueError("MAE decoder didn't get unshuffle indexes")

        hw = x.shape[1]
        unshuffle_ids = args[0]
        unshuffle_ids = torch.cat([torch.zeros(unshuffle_ids.shape[0], 1), unshuffle_ids + 1], dim=1)
        x = torch.cat([x, self.mask.expand(x.shape[0], unshuffle_ids.shape[1] - x.shape[1], -1)], dim=1)
        x = x.gather(1, unshuffle_ids.long().unsqueeze(2).repeat(1, 1, 192))
        x = self.patch_embeddings(x)
        x = self.vit(x).last_hidden_state[:,1:]

        patches = self.head(x)
        mask = torch.zeros_like(patches)
        mask[hw:] = 1
        mask = mask.gather(1, (unshuffle_ids[:, 1:] - 1).long().unsqueeze(2).repeat(1, 1, mask.shape[-1]))
        img = patches.view(patches.shape[0], self.out_channels, self.img_size, self.img_size)
        mask = mask.view(patches.shape[0], self.out_channels, self.img_size, self.img_size)
        return self.tanh(img), mask
