from torch import nn
from .simple import SimpleModel, SimpleTopologicalModel


def get_model(name: str, topological: bool, **kwargs) -> nn.Module:
    if name == 'simple':
        model = SimpleTopologicalModel(**kwargs) if topological else SimpleModel(**kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    return model
