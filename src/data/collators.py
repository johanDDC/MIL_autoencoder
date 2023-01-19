import torch


def fc_collator(batch):
    features = torch.Tensor()
    features = torch.stack([torch.cat([features, elem[0]], dim=0) for elem in batch])
    targets = torch.tensor([elem[1] for elem in batch])
    return torch.flatten(features, start_dim=1), targets
