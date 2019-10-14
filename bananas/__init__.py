''' ..include:: ../README.md '''

from .core.learner import Learner, SupervisedLearner, UnsupervisedLearner
from .core.mixins import BaseClassifier, BaseRegressor, HighDimensionalMixin
from .core.pipeline import Pipeline
