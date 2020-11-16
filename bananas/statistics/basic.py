"""
These are your basic stats. When available, functions in this submodule will utilize `numpy`.
Otherwise it will fallback to the `statistics` package or, if nothing else is available, a custom
implementation of the function. Implemented functions include `mean`, `median` and `variance`.
"""

import statistics
from ..utils.constants import TYPE_ARRAY
from ..utils import __NUMPY_AVAILABLE__

if __NUMPY_AVAILABLE__:
    import numpy


def almost_zero(x, atol: float = 1e-6):
    """ Returns true if `x` is within a given tolerance parameter """
    return abs(x) < atol


def mean(arr: TYPE_ARRAY):
    """
    Compute the mean of the array using `numpy` if available, else fallback to `statistics` module.
    """
    return numpy.mean(arr) if __NUMPY_AVAILABLE__ else statistics.mean(arr)


def median(arr: TYPE_ARRAY):
    """
    Compute the median of the array using `numpy` if available, else fallback to `statistics`
    module.
    """
    return numpy.median(arr) if __NUMPY_AVAILABLE__ else statistics.median(arr)


def variance(arr: TYPE_ARRAY):
    """
    Compute the variance of the array using `numpy` if available, else fallback to `statistics`
    module.
    """
    return numpy.var(arr, ddof=1) if __NUMPY_AVAILABLE__ else statistics.variance(arr)
