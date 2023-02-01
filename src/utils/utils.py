import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as T
import torch.nn.functional as F

from torchvision.utils import make_grid
from src.utils.mixup import mixup_criterion



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



class MAEclsExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0][:, 0, :]


@torch.no_grad()
def construct_grid(model, dataset, num_images=10, ids=None, f=None, collator=None, mae=False, device="cuda"):
    if ids is None:
        ids = np.random.randint(0, len(dataset), size=num_images)
    img_batch = torch.Tensor().to(device)
    restored_img_batch = torch.Tensor().to(device)
    masked_batch = torch.Tensor().to(device)
    for idx in ids:
        img, _ = dataset[idx]
        if collator is not None:
            img, _ = collator([(img, _)])
        else:
            img = img.unsqueeze(0)
        img = img.to(device, non_blocking=True)
        restored_img = model(img)
        if mae:
            restored_img, mask = restored_img
            masked_batch = torch.cat([masked_batch, img * (1 - mask)], dim=0)
        if f is not None:
            img = f(img).unsqueeze(0)
            restored_img = f(restored_img).unsqueeze(0)
        img_batch = torch.cat([img_batch, img], dim=0)
        restored_img_batch = torch.cat([restored_img_batch,
                                        restored_img], dim=0)

    img_batch = make_grid(img_batch.cpu(), nrow=num_images, padding=1, normalize=True, value_range=(-1, 1))
    restored_img_batch = make_grid(restored_img_batch.cpu(), nrow=num_images, padding=1, normalize=True,
                                   value_range=(-1, 1))
    if mae:
        masked_batch = make_grid(masked_batch.cpu(), nrow=num_images, padding=1, normalize=True, value_range=(-1, 1))
        grid = torch.cat([img_batch, masked_batch, restored_img_batch], dim=1)
    else:
        grid = torch.cat([img_batch, restored_img_batch], dim=1)
    return grid

def show_grid(grid, img_size=(5, 5)):
    fig, axs = plt.subplots(figsize=(5 * img_size[0], 5 * img_size[1]),
                            squeeze=False)
    grid = grid.detach()
    grid = T.to_pil_image(grid)
    axs[0, 0].imshow(grid)
    axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


class CosineFocalLoss(nn.Module):
    def __init__(self, num_classes, label_smoothing=0, lamb=0, gamma=0):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.lamb = lamb
        self.gamma = gamma

    def forward(self, logits, targets, mixup_lambda=1, train=False):
        if train:
            focal_loss = mixup_lambda * F.cross_entropy(logits, targets[0], reduction="none") + \
                         (1 - mixup_lambda) * F.cross_entropy(logits, targets[1], reduction="none")
            targets = F.one_hot(targets[0], num_classes=self.num_classes) + F.one_hot(targets[1], num_classes=self.num_classes)
        else:
            focal_loss = F.cross_entropy(logits, targets, reduction="none")
            targets = F.one_hot(targets, num_classes=self.num_classes)

        cos_loss = F.cosine_embedding_loss(
            logits,
            targets,
            torch.tensor([1], device=targets.device)
        )
        pt = torch.exp(-focal_loss)
        focal_loss = (1 - pt) ** self.gamma * focal_loss
        return cos_loss + self.lamb * focal_loss.mean()


class Rotation(nn.Module):
    def __init__(self, angles):
        super().__init__()
        self.angles = angles

    def forward(self, img):
        num = torch.randint(0, len(self.angles), (1,))
        return T.rotate(img, self.angles[num])

