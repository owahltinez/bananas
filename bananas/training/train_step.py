from dataclasses import dataclass
from ..utils.constants import TYPE_ARRAY


@dataclass
class TrainStep:
    """ Data class containing information about the current training iteration. """

    idx: int = 0
    step: int = 0
    score: float = 0.0
    max_steps: int = 0
    max_score: float = 0.0
    best_score: float = 0.0
    best_iteration: int = 0
    running_score: float = 0.0
    X_test: TYPE_ARRAY = None
    y_test: TYPE_ARRAY = None
    X_train: TYPE_ARRAY = None
    y_train: TYPE_ARRAY = None
