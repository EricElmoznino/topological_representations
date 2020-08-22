import os
import shutil
import math
import random
import time
from datetime import timedelta
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Metric

from .typing_aliases import *

cuda = torch.cuda.is_available()
random.seed(27)
torch.manual_seed(27)
if cuda: torch.cuda.manual_seed(27)


def train(run_name: str, model: nn.Module,
          train_set: Dataset, test_set: Dataset,
          train_step: StepFunction, eval_step: StepFunction,
          train_metrics: Dict[str, Metric], eval_metrics: Dict[str, Metric],
          n_iterations: int, batch_size: int) -> None:
    assert 'Loss' in eval_metrics

    # Make the run directory
    if not os.path.exists('training/saved_runs'):
        os.mkdir('training/saved_runs')
    save_dir = os.path.join('training/saved_runs', run_name)
    if run_name == 'debug':     # If we're debugging, just remove the old run without returning an error
        shutil.rmtree(save_dir, ignore_errors=True)
    os.mkdir(save_dir)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))      # Tensorboard logging

    # Instantiate model, data loaders, loss function, and optimizer
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Ignite train/evaluation engines
    train_engine = Engine(train_step)
    test_engine = Engine(eval_step)

    # Bind train/test metrics
    for name, metric in train_metrics.items():
        metric.attach(train_engine, name)
    for name, metric in eval_metrics.items():
        metric.attach(test_engine, name)

    # Progress bar displaying training loss and accuracy
    ProgressBar(persist=True).attach(train_engine, metric_names=list(train_metrics.keys()))

    # Model checkpointing (keep model with lowest test loss)
    checkpoint_handler = ModelCheckpoint(os.path.join(save_dir, 'checkpoints'), type(model).__name__,
                                         score_function=lambda eng: -eng.state.metrics['Loss'])
    test_engine.add_event_handler(event_name=Events.COMPLETED, handler=checkpoint_handler, to_save={'model': model})

    # Early stopping if the test set loss does not decrease over 5 epochs
    early_stop_handler = EarlyStopping(patience=5, trainer=train_engine,
                                       score_function=lambda eng: -eng.state.metrics['Loss'])
    test_engine.add_event_handler(Events.COMPLETED, early_stop_handler)

    # Log training metrics to tensorboard every 100 batches
    @train_engine.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_metrics(engine):
        for metric, value in engine.state.metrics.items():
            writer.add_scalar('training/{}'.format(metric), value, engine.state.iteration)

    # Print and log test metrics to tensorboard after every epoch
    @train_engine.on(Events.EPOCH_COMPLETED)
    def log_test_metrics(engine):
        test_engine.run(test_loader)
        results = ['Test Results - Epoch: {}'.format(engine.state.epoch)]
        for metric, value in test_engine.state.metrics.items():
            writer.add_scalar('test/{}'.format(metric), value, engine.state.iteration)
            results.append('{}: {:.2f}'.format(metric, value))
        print(' '.join(results))

    # Gracefully terminate on any exception, and simply end training + save the current model
    # if we are manually stopping with a keyboard interrupt (e.g. model was converging)
    @train_engine.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            checkpoint_handler(engine, {'model_exception': model})
        else:
            raise e

    # Start off training and report total execution time when over
    start_time = time.time()
    n_epochs = math.ceil(n_iterations / (len(train_set) / batch_size))
    train_engine.run(train_loader, n_epochs)
    writer.close()
    end_time = time.time()
    print('Total training time: {}'.format(timedelta(seconds=end_time - start_time)))
