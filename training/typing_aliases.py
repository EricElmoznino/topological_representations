from typing import Callable, Tuple, Dict, Union
from torch import Tensor
from ignite.engine import Engine

Batch = Union[Tensor, Tuple[Tensor, ...], Dict[str, Tensor]]
StepOutput = Dict[str, Union[Tensor, float]]
StepFunction = Callable[[Engine, Batch], StepOutput]
