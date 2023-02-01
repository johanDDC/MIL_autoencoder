import glob

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from src.utils.utils import Rotation

color_common = T.Compose([
    T.ToTensor(),
    T.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    ),
])

bw_common = T.Compose([
    T.ToTensor()
])

train_trainsform = lambda cmn: T.Compose([
    T.RandAugment(9, 1),
    cmn,
    T.RandomErasing(p=0.25),
    Rotation([0, 90, 180, 270])
])

tta = lambda cmn: T.Compose([
    T.RandAugment(9, 1),
    cmn
])


def get_train_dataset(dataset_path="data", download=False):
    return torchvision.datasets.CIFAR100(dataset_path, train=True, download=download, transform=common)


def get_val_dataset(dataset_path="data", download=False):
    return torchvision.datasets.CIFAR100(dataset_path, train=False, download=download, transform=common)


class SSLDataset(Dataset):
    def __init__(self, path, train=False, labled=False):
        self.labled = labled
        if train:
            self.transforms = train_trainsform(color_common)
        else:
            self.transforms = tta(color_common)
        if labled:
            self.data = ImageFolder(path, transform=T.Compose([T.ToTensor()]))
        else:
            imgs = list(glob.iglob(f"{path}/*.jpg", recursive=True))
            self.data = []
            for img in tqdm(imgs, desc="Dataset loading"):
                self.data.append(io.imread(img))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        if self.labled:
            img, target = img
        else:
            target = color_common(TF.to_pil_image(img.copy()))
        img = self.transforms(TF.to_pil_image(img))
        return img, target


class BlackWhiteDataset(Dataset):
    def __init__(self, path, train=False, labled=False):
        self.labled = labled
        grayscale = T.Grayscale(num_output_channels=1)
        if train:
            self.transforms = train_trainsform(bw_common)
        else:
            self.transforms = tta(bw_common)
        if labled:
            self.data = ImageFolder(path, transform=T.Compose([grayscale]))
        else:
            imgs = list(glob.iglob(f"{path}/*.jpg", recursive=True))
            self.data = []
            for img in tqdm(imgs, desc="Dataset loading"):
                self.data.append(grayscale(Image.open(img)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        if self.labled:
            img, target = img
        else:
            target = bw_common(img.copy())
        img = self.transforms(img)
        return img, target
