from dataclasses import dataclass
from ..utils.constants import TYPE_ARRAY


@dataclass
class TrainStep:
    """ Data class containing information about the current training iteration. """

    iteration: int = -1
    score: float = None
    max_steps: int = 0
    max_score: float = None
    best_score: float = None
    best_iteration: int = -1
    running_score: float = None
    X_test: TYPE_ARRAY = None
    y_test: TYPE_ARRAY = None
    X_train: TYPE_ARRAY = None
    y_train: TYPE_ARRAY = None
