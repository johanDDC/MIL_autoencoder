import torch
import torch.nn as nn

from torch import Tensor


class PatchEmbeddings(nn.Module):
    def __init__(self, img_size: int, patch_size: int, emb_size: int, with_cls=False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.with_cls = with_cls

        self.pos_embeddings = nn.Embedding((img_size // patch_size) ** 2 + int(with_cls), emb_size)

    def forward(self, patches: Tensor):
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        positions = torch.arange(int(self.with_cls),
                                 (self.img_size // self.patch_size) ** 2 + int(self.with_cls), device=patches.device)

        return patches + self.pos_embeddings(positions)
