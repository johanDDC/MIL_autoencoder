import torchvision
import torchvision.transforms as T

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(0.5, 0.5)
])


def get_train_dataset(dataset_path="data", download=False):
    return torchvision.datasets.CIFAR100(dataset_path, train=True, download=download, transform=transforms)


def get_val_dataset(dataset_path="data", download=False):
    return torchvision.datasets.CIFAR100(dataset_path, train=False, download=download, transform=transforms)
