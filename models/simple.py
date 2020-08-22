from torch import nn
from torch import Tensor

"""
These models are designed to map a (28, 28) input (e.g. MNIST images)
down to a spatial dimension of (1, 1), but can be used on larger images.
"""


class MNISTModel(nn.Module):

    def __init__(self, nc: int, n_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(nc, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 784, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes)
        )

    def forward(self, image: Tensor) -> Tensor:
        feats = self.features(image)
        feats = feats.view(feats.size(0), -1)
        preds = self.classifier(feats)
        return preds


class MNISTTopologicalModel(nn.Module):

    def __init__(self, nc: int, n_classes: int, clamp: bool = True):
        super().__init__()

        self.spatial_features = nn.Sequential(
            nn.Conv2d(nc, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 784, 3),
            nn.Sigmoid() if clamp else nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )

        self.topological_features = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes)
        )

    def forward(self, image: Tensor, return_topology: bool = False) -> Tensor:
        topological_representation = self.spatial_features(image)
        topological_representation = topological_representation.view(-1, 1, 28, 28)
        if return_topology:
            return topological_representation
        feats = self.topological_features(topological_representation)
        feats = feats.view(feats.size(0), -1)
        preds = self.classifier(feats)
        return preds


class CIFAR100Model(nn.Module):

    def __init__(self, nc: int, n_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(nc, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_classes)
        )

    def forward(self, image: Tensor) -> Tensor:
        feats = self.features(image)
        feats = feats.view(feats.size(0), -1)
        preds = self.classifier(feats)
        return preds


class CIFAR100TopologicalModel(nn.Module):

    def __init__(self, nc: int, n_classes: int, clamp: bool = True):
        super().__init__()

        self.spatial_features = nn.Sequential(
            nn.Conv2d(nc, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, 3),
            nn.Sigmoid() if clamp else nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )

        self.topological_features = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes)
        )

    def forward(self, image: Tensor, return_topology: bool = False) -> Tensor:
        topological_representation = self.spatial_features(image)
        topological_representation = topological_representation.view(-1, 1, 28, 28)
        if return_topology:
            return topological_representation
        feats = self.topological_features(topological_representation)
        feats = feats.view(feats.size(0), -1)
        preds = self.classifier(feats)
        return preds
