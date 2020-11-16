from enum import Enum
from ..training.train_step import TrainStep

_WARMUP_STEP_COUNT = 1000


def crit_target_score(step: TrainStep):
    """ If current score exceeds target score """
    return step.running_score >= step.max_score


def crit_worse_score(step: TrainStep):
    """ If score has gotten significantly worse than best """
    # TODO: handle negative score case
    return step.iteration > _WARMUP_STEP_COUNT and step.best_score * 0.8 > step.score.running_score


def crit_improve_score(step: TrainStep):
    """ If score has not reached new best in minimum step count """
    return step.iteration - step.best_iteration > _WARMUP_STEP_COUNT



def crit_negative_score(step: TrainStep):
    """ If score is negative after warmup step count """
    return step.iteration > _WARMUP_STEP_COUNT and step.best_score <= 0.0


class HaltCriteria(Enum):
    """ Enum of possible criteria used to halt training iterations """

    TARGET_SCORE = crit_target_score
    WORSE_SCORE = crit_worse_score
    IMPROVE_SCORE = crit_improve_score
    NEGATIVE_SCORE = crit_negative_score
