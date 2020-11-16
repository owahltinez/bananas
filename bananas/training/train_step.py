from dataclasses import dataclass
from ..utils.arrays import ARRAY_LIKE


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
    X_test: ARRAY_LIKE = None
    y_test: ARRAY_LIKE = None
    X_train: ARRAY_LIKE = None
    y_train: ARRAY_LIKE = None
