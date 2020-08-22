from typing import Optional, List, Tuple
import math
from torch import Tensor
from torchvision.transforms import functional as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()


def plot_confusion_matrix(predictions: Tensor, truths: Tensor, class_names: Optional[List[str]] = None,
                          title: str = 'Confusion matrix', save_path: Optional[str] = None):
    plt.close()
    cm = confusion_matrix(truths.cpu().numpy(), predictions.cpu().numpy())
    cm = cm / cm.sum(axis=1, keepdims=True)
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    ax = sns.heatmap(cm, vmin=0, vmax=1, xticklabels=class_names, yticklabels=class_names, cmap='YlGnBu')
    ax.set_title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def topology_as_image(topology: Tensor, range: Optional[Tuple[int, int]] = None) -> Image.Image:
    if range is None:
        range = (topology.min(), topology.max())
    topology = (topology - range[0]) / (range[1] - range[0])
    topology = transforms.to_pil_image(topology)
    return topology


def class_topology_grid(class_images: List[List[Image.Image]], class_topologies: List[List[Image.Image]]) -> Image.Image:
    class_grids = []
    for class_image_samples, class_topology_samples in zip(class_images, class_topologies):
        class_image_grid = make_grid(class_image_samples, pad=1)
        class_topology_grid = make_grid(class_topology_samples, pad=1)
        class_grid = make_grid([class_image_grid, class_topology_grid], n_rows=1, pad=4)
        class_grids.append(class_grid)
    class_grids = make_grid(class_grids, pad=8)
    return class_grids


def make_grid(images, n_rows=None, pad=0):
    assert len(images) > 0
    for i in range(len(images)):
        if min([images[i].width, images[i].height]) < 224:
            images[i] = transforms.resize(images[i], 224, interpolation=Image.NEAREST)
    if n_rows is None:
        n_rows = math.ceil(math.sqrt(len(images)))
    n_cols = math.ceil(len(images) / n_rows)
    w, h = images[0].width, images[0].height
    grid = Image.new(images[0].mode, (w * n_cols + pad * (n_cols - 1),
                                      h * n_rows + pad * (n_rows - 1)),
                     color=255 if images[0].mode == 'L' else (255, 255, 255))
    for i, img in enumerate(images):
        row = int(i / n_cols)
        col = i % n_cols
        grid.paste(img, ((w + pad) * col, (h + pad) * row))
    return grid
