'''
Interfaces and APIs that drive the design of the rest of the modules.

Core is, as the name implies, the most fundamental module of this library. It contains the interfaces
and APIs that drive the design of the rest of the modules in this library. It contains 3
sub-components: `Learner` classes, **mixins**, and the `Pipeline` class.
'''

from .learner import Learner, SupervisedLearner, UnsupervisedLearner
from .mixins import BaseClassifier, BaseRegressor
from .pipeline import Pipeline
