from argparse import ArgumentParser
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ignite.metrics import RunningAverage, Average, Accuracy

from training.trainer import train
from training.typing_aliases import *
from datasets.dispatcher import get_dataset
from models.dispatcher import get_model
from training.utils import plot_confusion_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = ArgumentParser(description='Train a classification model')
    parser.add_argument('--run_name', required=True, type=str, help='name of the current run (where runs are saved)')
    parser.add_argument('--data_dir', required=True, type=str, help='directory containing dataset')
    parser.add_argument('--n_iterations', type=float, default=5e3, help='number of iterations to run')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model', type=str, default='MNIST', help='model architecture type to use')
    parser.add_argument('--dataset', type=str, default='MNIST', help='directory containing dataset')
    parser.add_argument('--topological', action='store_true', help='use topological version of the model')
    args = parser.parse_args()

    train_set, test_set, nc, n_classes = get_dataset(args.dataset, args.data_dir)
    model = get_model(args.model, args.topological, nc=nc, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train_step(engine: Engine, batch: Batch) -> StepOutput:
        model.train()
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        loss = F.cross_entropy(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'preds': preds, 'targets': targets, 'cross_entropy': loss.item()}

    def eval_step(engine: Engine, batch: Batch) -> StepOutput:
        model.eval()
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            preds = model(images)
        loss = F.cross_entropy(preds, targets)
        return {'preds': preds, 'targets': targets, 'cross_entropy': loss.item()}

    train_metrics = {
        'Loss': RunningAverage(output_transform=lambda x: x['cross_entropy']),
        'Accuracy': RunningAverage(Accuracy(output_transform=lambda x: (x['preds'], x['targets'])))
    }

    eval_metrics = {
        'Loss': Average(output_transform=lambda x: x['cross_entropy']),
        'Accuracy': Accuracy(output_transform=lambda x: (x['preds'], x['targets']))
    }

    train(args.run_name, model, train_set, test_set,
          train_step, eval_step,
          train_metrics, eval_metrics,
          args.n_iterations, args.batch_size)

    predictions = []
    truths = []
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            predictions.append(model(images).argmax(dim=1))
        truths.append(targets)
    predictions = torch.cat(predictions, dim=0)
    truths = torch.cat(truths, dim=0)
    plot_confusion_matrix(predictions, truths, title=args.dataset + ' Confusion Matrix',
                          save_path=os.path.join('training/saved_runs', args.run_name, 'confusion_matrix.jpg'))


