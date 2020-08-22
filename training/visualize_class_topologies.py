from argparse import ArgumentParser
import os
import torch

from datasets.dispatcher import get_dataset
from models.dispatcher import get_model
from training.utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = ArgumentParser(description='Train a classification model')
    parser.add_argument('--run_name', required=True, type=str, help='name of the run containing a trained model')
    parser.add_argument('--data_dir', required=True, type=str, help='directory containing dataset')
    parser.add_argument('--model', type=str, default='simple', help='model architecture type to use')
    parser.add_argument('--dataset', type=str, default='MNIST', help='directory containing dataset')
    parser.add_argument('--samples_per_class', type=int, default=16, help='how many topology samples to show per class')
    args = parser.parse_args()

    _, test_set, nc, n_classes = get_dataset('MNIST', args.data_dir)
    model = get_model(args.model, topological=True, nc=nc, n_classes=n_classes).to(device)
    model.eval()

    class_images = [[] for _ in range(n_classes)]
    class_topologies = [[] for _ in range(n_classes)]
    remaining_classes = set([c for c in range(n_classes)])
    min_topology_activation = None
    max_topology_activation = None

    for i in range(len(test_set)):
        image_tensor, target = test_set[i]
        image = transforms.to_pil_image(image_tensor)

        if target not in remaining_classes:
            continue

        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            topology = model(image_tensor.unsqueeze(0), return_topology=True).squeeze(0).cpu()

        if min_topology_activation is None or topology.min() < min_topology_activation:
            min_topology_activation = topology.min()
        if max_topology_activation is None or topology.max() > max_topology_activation:
            max_topology_activation = topology.max()

        if len(class_topologies[target]) == args.samples_per_class:
            remaining_classes.remove(target)
        else:
            class_topologies[target].append(topology)
            class_images[target].append(image)

    class_topologies = [[topology_as_image(t, (min_topology_activation, max_topology_activation))
                         for t in c] for c in class_topologies]

    class_grids = class_topology_grid(class_images, class_topologies)
    class_grids.save(os.path.join('training/saved_runs', args.run_name, 'class_topologies.jpg'))
