import torchvision
import torchvision.transforms as T

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(0.5, 0.5)
])


def get_train_dataset(download=False):
    return torchvision.datasets.CIFAR100("data", train=True, download=download, transform=transforms)


def get_val_dataset(download=False):
    return torchvision.datasets.CIFAR100("data", train=False, download=download, transform=transforms)
