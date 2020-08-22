from typing import Tuple
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


def get_dataset(name: str, data_dir: str) -> Tuple[Dataset, Dataset, int, int]:
    if name == 'MNIST':
        train_set = MNIST(root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
        test_set = MNIST(root=data_dir, train=False, transform=transforms.ToTensor())
        nc = 1
        n_classes = 10
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    return train_set, test_set, nc, n_classes
