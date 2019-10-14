from dataclasses import dataclass
from enum import Enum

from ..utils.arrays import ARRAY_LIKE

@dataclass
class TrainStep:
    ''' Data class containing information about the current training iteration. '''

    idx: int = 0
    step: int = 0
    score: float = 0.
    max_steps: int = 0
    max_score: float = 0.
    best_idx: int = 0.
    best_score: float = 0.
    running_score: float = 0.
    X_test: ARRAY_LIKE = None
    y_test: ARRAY_LIKE = None
    X_train: ARRAY_LIKE = None
    y_train: ARRAY_LIKE = None


def crit_target_score(step: TrainStep):
    ''' If current score exceeds target score '''
    return step.running_score >= step.max_score

def crit_worse_score(step: TrainStep):
    ''' If score has gotten significantly worse than best '''
    # TODO: handle negative score case
    return step.idx > 500 and step.best_score * .8 > step.score.running_score

def crit_improve_score(step: TrainStep):
    ''' If score has not reached new best in 500 steps '''
    return step.idx - step.best_idx > 500

def crit_negative_score(step: TrainStep):
    ''' If score is negative after warmup '''
    return step.idx > 500 and step.best_score <= 0.


class HaltCriteria(Enum):
    ''' Enum of possible criteria used to halt training iterations '''
    TARGET_SCORE = crit_target_score
    WORSE_SCORE = crit_worse_score
    IMPROVE_SCORE = crit_improve_score
    NEGATIVE_SCORE = crit_negative_score
