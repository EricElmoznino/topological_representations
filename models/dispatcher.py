from .simple import *


def get_model(name: str, topological: bool, **kwargs) -> nn.Module:
    if name == 'MNIST':
        model = MNISTTopologicalModel(**kwargs) if topological else MNISTModel(**kwargs)
    elif name == 'CIFAR100':
        model = CIFAR100TopologicalModel(**kwargs) if topological else CIFAR100Model(**kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    return model
