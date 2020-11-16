"""
Mixins and halt criteria definitions for training of models.

Obviously, training is a fundamental component of a fully-fledged ML framework. In this library,
training is implemented with a mixin that introduces a `train` function to inheriting classes.
Unlike `fit`, which takes in samples to be learned, `train` accepts an input function. This is very
similar to how Tensorflow's [train](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#train)
function works.
"""

from .criteria import HaltCriteria
