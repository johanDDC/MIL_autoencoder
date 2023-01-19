import torch


def fc_collator(batch):
    return torch.flatten(batch)
