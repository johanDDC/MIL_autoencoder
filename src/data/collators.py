import torch


def fc_collator(batch):
    features = torch.Tensor()
    features = torch.stack([torch.cat([features, elem[0]], dim=0) for elem in batch])
    targets = torch.tensor([elem[1] for elem in batch])
    return torch.flatten(features, start_dim=1), targets

def pretrain_collator(batch):
    features = torch.Tensor()
    features = torch.stack([torch.cat([features, elem[0]], dim=0) for elem in batch])
    return features, None

def bw_collator(batch, no_targets=False):
    features = [torch.cat([elem[0], torch.zeros_like(elem[0]), torch.zeros_like(elem[0])], dim=0) for elem in batch]
    features = torch.stack(features)
    targets = None
    # features = torch.stack([features, torch.dstack([elem[0], torch.zeros_like(elem[0]), torch.zeros_like(elem[0])]) for elem in batch])
    # print(features.shape)
    if not no_targets:
        if type(batch[0][1]) == int:
            targets = torch.tensor([elem[1] for elem in batch])
        else:
            targets = torch.stack([elem[1] for elem in batch])
    return features, targets
