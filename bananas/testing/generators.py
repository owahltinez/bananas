"""
Used for testing purposes, this module implements a large number of synthetic data generators that
output pseudo-random data of many different types. For example, we can use it to generate a set of
fake images, one-hot vectors, or simply an array of strings. Reference the [source code](
generators.py) for more details.
"""

import numpy
from functools import wraps
from ..statistics.random import RandomState
from ..utils.constants import TYPE_ARRAY

# Global (to this module) random number generator
_RNG = RandomState()


def random_seeded(func):
    """ Decorator that uses the `random_seed` parameter from functions to seed the RNG. """

    @wraps(func)
    def wrapper(*args, random_seed: int = None, **kwargs):
        _RNG.seed(random_seed)
        return func(*args, **kwargs)

    return wrapper


@random_seeded
def generate_array_floats(n: int = 1024, random_seed: int = None) -> TYPE_ARRAY:
    """ Generate an array of size `n` and type `numpy.float64`. """
    return _RNG.rand(n).astype(numpy.float64)


@random_seeded
def generate_array_chars(n: int = 1024, vocab: list = None, random_seed: int = None) -> TYPE_ARRAY:
    """ Generate an array of size `n` and type `str`, each of length 1. """
    vocab = vocab or list("abcdefghijklmnopqrstuvwxyz")
    return _RNG.choice(vocab, n).astype(str)


@random_seeded
def generate_array_strings(
    n: int = 1024, vocab: list = None, word_size: int = 8, random_seed: int = None
) -> TYPE_ARRAY:
    """ Generate an array of size `n` and type `str`, each of length `word_size`. """
    vocab = vocab or list("abcdefghijklmnopqrstuvwxyz")
    return numpy.asarray(["".join(_RNG.choice(vocab, word_size)) for _ in range(n)], dtype=str)


@random_seeded
def generate_array_ints(n: int = 1024, max_int: int = 256, random_seed: int = None) -> TYPE_ARRAY:
    """ Generate an array of size `n` and type `int`. """
    return _RNG.randint(0, max_int, n).astype(int)


@random_seeded
def generate_array_uints(n: int = 1024, max_int: int = 256, random_seed: int = None) -> TYPE_ARRAY:
    """ Generate an array of size `n` and type `numpy.uint8`. """
    return _RNG.randint(0, max_int, n).astype(numpy.uint8)


@random_seeded
def generate_array_booleans(n: int = 1024, random_seed: int = None) -> TYPE_ARRAY:
    """ Generate an array of size `n` and type `bool`. """
    return _RNG.randint(0, 2, n).astype(bool)


@random_seeded
def generate_array_nones(n: int = 1024, random_seed: int = None) -> TYPE_ARRAY:
    """ Generate an array of size `n` and type `O` contianing None objects. """
    return numpy.asarray([None] * n, dtype="O")


@random_seeded
def generate_array_infinities(n: int = 1024) -> TYPE_ARRAY:
    """ Generate an array of size `n` and type `numpy.float64` containing `numpy.inf` objects. """
    return numpy.asarray([numpy.inf] * n, dtype=numpy.float64)


@random_seeded
def generate_array_int_floats(
    n: int = 1024, max_int: int = 256, random_seed: int = None
) -> TYPE_ARRAY:
    """ Generate an array of size `n` and type `float`. """
    return _RNG.randint(0, max_int, n).astype(float)


# Used for testing different feature types


@random_seeded
def generate_images(
    n: int = 128, w: int = 32, h: int = 32, c: int = 1, random_seed: int = None
) -> TYPE_ARRAY:
    """ Generate an array of size `n` containing images that are `w` x `h` with `c` channels. """
    return _RNG.randint(0, 256, n * c * w * h).reshape(n, c, w, h)


@random_seeded
def generate_onehot_matrix(n: int = 1024, ndim: int = 8, random_seed: int = None) -> TYPE_ARRAY:
    """
    Generate an array of size `n` containing vectors of size `ndim` in which a single value is one
    and all other values are zero.
    """
    to_vec = lambda x: [1 if i == x else 0 for i in range(ndim)]
    return numpy.array([to_vec(x) for x in _RNG.randint(0, ndim, n)]).astype(int)

