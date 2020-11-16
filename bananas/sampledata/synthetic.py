""" Synthetic datasets module """

import math
import itertools
import collections

import numpy

from ..dataset.dataset import DataSet
from ..statistics.random import RandomState

# Disable variable name nagging since these are mostly math functions
# pylint: disable=invalid-name


def new_labels(samples: int = 1024, random_seed: int = None, key: str = None) -> DataSet:
    """ Random labeled dataset """
    rng = RandomState(random_seed)
    kinds = collections.OrderedDict()
    kinds["fruits"] = ["apple", "pear", "orange", "banana", "lemon", "lime", "strawberry"]
    kinds["animals"] = ["giraffe", "cat", "dog", "horse", "sheep", "cow", "pig", "chicken"]
    kinds["permissions"] = ["".join(map(str, x)) for x in itertools.product([4, 5, 6, 7], repeat=3)]
    if key is None:
        key = rng.choice(list(kinds.keys()))
    num_labels = len(kinds[key])
    X = rng.randint(0, num_labels, samples)  # sampling
    y = numpy.array([kinds[key][i] for i in X], dtype=str)
    return DataSet.from_ndarray(  # keep X in [0, 1]
        numpy.asarray([X / num_labels]), y, random_seed=random_seed, name="labels"
    )


def new_line(samples: int = 1024, random_seed: int = None, noise_scale: float = 0.05) -> DataSet:
    """ ~y = a * ~X + b """
    rng = RandomState(random_seed)
    X = numpy.linspace(-1, 1, samples)  # sampling
    r = (rng.randn(samples)) * noise_scale  # noise
    a = rng.rand() * rng.choice([1, -1])  # slope
    b = rng.rand() * rng.choice([1, -1])  # independent term
    y = a * X + b  # ground truth
    y_ = a * (X + r) + b  # samples with noise
    y_ += (rng.randn(samples) - 0.5) * noise_scale  # noise
    return DataSet.from_ndarray(numpy.asarray([X]), y, random_seed=random_seed, name="line")


def new_poly(
    samples: int = 1024, random_seed: int = None, degree: int = 4, noise_scale: float = 0.05
) -> DataSet:
    """ ~y = a_i * ~X^i + a_(i+1) * ~X^(i+1) ..."""
    rng = RandomState(random_seed)
    X = numpy.linspace(-1, 1, samples)  # sampling
    y = numpy.zeros(samples)  # initialize ground truth
    y_ = numpy.zeros(samples)  # initialize sample with noise
    for i in range(degree + 1):
        r = (rng.randn(samples)) * noise_scale  # noise
        a = rng.rand() * rng.choice([1, -1])  # a_i * X ^ i
        y += a * X ** i  # update ground truth
        y_ += a * (X + r) ** i  # update sample with noise
    y_ += (rng.randn(samples) - 0.5) * noise_scale  # noise
    return DataSet.from_ndarray(numpy.asarray([X]), y, random_seed=random_seed, name="poly")


def new_trig(samples: int = 1024, random_seed: int = None, noise_scale: float = 0.05) -> DataSet:
    """ ~y = a * (sin or cos)(~X) + b """
    rng = RandomState(random_seed)
    t = numpy.pi ** 2 / 2
    X = numpy.linspace(-1, 1, samples)  # sampling
    r = (rng.randn(samples)) * noise_scale  # noise
    a = rng.rand() * rng.choice([1, -1])
    b = rng.rand() * rng.choice([1, -1])
    op = rng.choice([numpy.sin, numpy.cos])
    y = (a * op(X * t) + b) * t  # ground truth
    y_ = (a * op((X + r) * t) + b) * t  # samples with noise
    y_ += (rng.randn(samples) - 0.5) * noise_scale  # noise
    return DataSet.from_ndarray(numpy.asarray([X]), y, random_seed=random_seed, name="trig")


def new_wave(samples: int = 1024, random_seed: int = None, noise_scale: float = 0.05) -> DataSet:
    """ ~y = a * (sin or cos)(~X) + b """
    rng = RandomState(random_seed)
    t = numpy.pi * 3
    X = numpy.linspace(-1, 1, samples)  # sampling
    r = (rng.randn(samples)) * noise_scale  # noise
    a = rng.rand() * rng.choice([1, -1])
    b = rng.rand() * rng.choice([1, -1])
    op = rng.choice([numpy.sin, numpy.cos])
    y = (a * op(X * t) + b) * t  # ground truth
    y_ = (a * op((X + r) * t) + b) * t  # samples with noise
    y_ += (rng.randn(samples) - 0.5) * noise_scale  # noise
    return DataSet.from_ndarray(numpy.asarray([X]), y, random_seed=random_seed, name="wave")


def new_mat9(
    samples: int = 1024, random_seed: int = None, k: int = 27, noise_scale: float = 0.2
) -> DataSet:
    """ Fuzzy 3x3 matrix of dots in a 27x27 grid """
    rng = RandomState(random_seed)
    X = numpy.zeros((samples, k, k))
    r = (rng.rand(samples, k, k)) * noise_scale  # noise
    a = numpy.linspace(0, 0.5, math.ceil(k / 6)).tolist()
    b = numpy.linspace(0.5, 0, math.floor(k / 6)).tolist()
    q = numpy.array([(a + b)] * math.floor(k / 3))
    q += numpy.transpose(q)
    y = rng.randint(0, 9, samples)
    for i, y_ in enumerate(y):
        xi, yi = y_ // 3, y_ % 3
        i1, i2 = int(k * (y_ // 3) / 3), int(k * (y_ % 3) / 3)
        X[i, i1 : i1 + q.shape[0], i2 : i2 + q.shape[1]] += q
    return DataSet.from_ndarray(numpy.asarray([X + r]), y + 1, random_seed=random_seed, name="mat9")
