""" Module with utilities used to deal with arrays and array-like objects """

import math
import numbers
from typing import Any, Iterable, List
from .constants import DTYPE_FLOAT, DTYPE_INT, DTYPE_STR, DTYPE_UINT8
from .constants import ARRAY_LIKE, ALLOWED_DTYPES
from ..utils import __NUMPY_AVAILABLE__

if __NUMPY_AVAILABLE__:
    import numpy


def check_array(
    arr: Iterable,
    min_samples: int = 1,
    min_dimension: int = 0,
    max_dimension: int = None,
    allow_nan: bool = False,
    dtype: numpy.dtype = None,
) -> numpy.array:
    """
    Makes sure that argument is of compatible numpy.array type, converts it to a numpy array if
    necessary. We restrict ourselves to 5 dtypes: bool, str, int, uint8, and float64. Casting will
    be attempted from the closest subdtype, failure will not be caught and exceptions may be raised
    by this function.
    """

    # If the array has an attribute named values (e.g. pandas dataframes), extract it
    if hasattr(arr, "values"):
        arr = getattr(arr, "values")

    # If the array has an attribute named data (e.g. some tensor types), extract it
    if hasattr(arr, "data"):
        arr = getattr(arr, "data")

    # If the array is not a numpy, convert to numpy array
    if not isinstance(arr, numpy.ndarray):
        arr = numpy.asarray(arr, dtype=dtype)

    # At this point, array should be of numpy array type
    assert isinstance(arr, numpy.ndarray), "Unknown type of input. Found %r, expected %r" % (
        type(arr),
        numpy.ndarray,
    )

    # Enforce minimum samples required
    num_samples = len(arr)
    if num_samples < min_samples:
        raise ValueError(
            "Input has less than the minimum required number of samples. Expected at "
            "least %d, found %d" % (min_samples, num_samples)
        )

    # Enforce minimum columns required
    # TODO: this should be min/max dimensions instead of columns
    num_columns = 1 if arr.ndim == 1 else arr.shape[1]
    if num_columns < min_dimension:
        raise ValueError(
            "Input has less than the minimum required number of columns. Expected at "
            "least %d, found %d" % (min_dimension, num_columns)
        )

    # Enforce maximum columns required
    # TODO: this should be min/max dimensions instead of columns
    if max_dimension is not None and num_columns > max_dimension:
        raise ValueError(
            "Input has more than the maximum allowed number of columns. Expected at "
            "most %d, found %d" % (max_dimension, num_columns)
        )

    # Early exit: if a dtype was requested, return input as that type
    if dtype is not None:
        return arr.astype(dtype)

    # Try to convert object type to float first, fallback to string
    if arr.dtype.kind == "O":
        try:
            arr = arr.astype(DTYPE_FLOAT[0])
        except ValueError:
            arr = arr.astype(DTYPE_STR[0])

    # Sanity checks for numeric types
    if (
        not allow_nan
        and numpy.issubdtype(arr.dtype, numpy.number)
        and (not numpy.isfinite(arr.sum()) or not numpy.isfinite(arr).all())
    ):
        raise ValueError("Input contains NaN, infinity or a value too large for %r" % arr.dtype)

    # Convert all floating-like types to numpy.float64
    if arr.dtype != DTYPE_FLOAT[0] and numpy.issubdtype(arr.dtype, DTYPE_FLOAT[1]):
        arr = arr.astype(DTYPE_FLOAT[0])

    # Cast floats into ints if no precission is lost (i.e. [1., 2., 3.] => [1, 2, 3])
    if arr.dtype == DTYPE_FLOAT[0] and numpy.all(arr == arr.astype(DTYPE_INT[0])):
        arr = arr.astype(DTYPE_INT[0])

    # Convert all int-like types to int, leave uint8 alone
    if (
        arr.dtype != DTYPE_INT[0]
        and arr.dtype != DTYPE_UINT8[0]
        and numpy.issubdtype(arr.dtype, DTYPE_INT[1])
    ):
        arr = arr.astype(DTYPE_INT[0])

    # Convert all string-like types to str
    if arr.dtype != DTYPE_STR[0] and numpy.issubdtype(arr.dtype, DTYPE_STR[1]):
        arr = arr.astype(DTYPE_STR[0])

    # What's left must be one of the supported dtypes
    if not any([numpy.issubdtype(arr.dtype, dtype) for dtype in ALLOWED_DTYPES]):
        raise ValueError(
            "Input has non-allowed dtype. Expected one of %r, found %r"
            % (ALLOWED_DTYPES, arr.dtype)
        )

    return arr


def transpose_array(arr: Iterable):
    """
    Transposes the first two dimensions of the provided array, so a column-first array will become
    a sample-first array and vice-versa. This function makes no data type conversions of the
    underlying data and respects the shape of the input. In other words, if the input is a vector it
    will return a vector, and if the input is a matrix or a list of vectors it will return a list of
    vectors. The returned columns could be a list of vectors or a single vector, and the vectors can
    be of `list` or `ndarray` type. All vectors will have consistent type, either all `list` or all
    `ndarray`.
    """
    assert isinstance(
        arr, ARRAY_LIKE
    ), "Unknown array-like type used. Expected one of %r, found " "%r" % (ARRAY_LIKE, type(arr))

    # Data might already be a numpy array, breakout and return columns as list (or single vector)
    if __NUMPY_AVAILABLE__ and isinstance(arr, numpy.ndarray):
        return [arr[:, i] for i in range(arr.shape[1])] if len(arr.shape) > 1 else arr

    # Data might be in a single column, so there's nothing to transpose
    if not isinstance(arr[0], ARRAY_LIKE):
        return arr

    # Otherwise transpose samples iteratively
    return [[x[i] for x in arr] for i in range(len(arr[0]))]


def check_equal_shape(arr1: Iterable, arr2: Iterable):
    """ Returns true of both arrays are of equal shape as per [shape_of_array]. """
    assert shape_of_array(arr1) == shape_of_array(arr2), (
        "Shape of targets differs. Expected "
        "%r, found %r" % (shape_of_array(arr1), shape_of_array(arr2))
    )


def equal_nested(a: Iterable, b: Iterable, atol: float = 1e-6) -> [bool]:
    """ Returns true if both arrays, which can be nested, are equal up to `atol` tolerance. """
    if all([isinstance(x, ARRAY_LIKE) for x in (a, b)]):
        if len(a) != len(b):
            return [False] * max(len(a), len(b))
        return [all(equal_nested(a_, b_)) for a_, b_ in zip(a, b)]
    if all([isinstance(x, numbers.Number) for x in (a, b)]):
        return [abs(max(a, b) - min(a, b)) < atol]
    return [a == b]


def ndim(arr: Iterable) -> int:
    """
    Compute the dimensionality of an array. This function assumes that all elements of the array
    are of equal shape.
    """
    assert isinstance(arr, ARRAY_LIKE)

    dim = 0
    while isinstance(arr, ARRAY_LIKE):
        dim += 1
        arr = arr[0]
    return dim


def shape_of_array(arr: Iterable):
    """
    Computes the shape of the array, including sub-arrays -- looking only at the first item. For
    example, the array `[[1,2,3],[4,5,6],[7,8,9]]` would return shape of `(3,3)`.
    """
    if not isinstance(arr, ARRAY_LIKE):
        return tuple()
    return tuple([len(arr), *(shape_of_array(arr[0]))])


def shape_of_features(cols: Iterable):
    """
    Convenience method used to compute the shape of each of the provided features. Essentially,
    returns the dimensions of each column as per [shape_of_array].
    """
    return {i: shape_of_array(col)[1:] for i, col in enumerate(cols)}


def argwhere(arr: Iterable, value: Any = True) -> List:
    """ Reimplementation of [numpy.argwhere]. """
    # if __NUMPY_AVAILABLE__:
    #     arr = numpy.array(arr)
    #     return numpy.argwhere(arr[arr == value]).flatten().tolist()
    return [idx for idx, val in enumerate(arr) if val == value]


def argmax(arr: Iterable[Iterable]) -> List:
    """ Returns the index of the max value item for each row in a 2D array """
    if __NUMPY_AVAILABLE__:
        return numpy.argmax(arr, axis=1).flatten().tolist()
    row_len = len(arr[0])
    return [max(zip(row, range(row_len)))[1] for row in arr]


def argmin(arr: Iterable[float]) -> float:
    """ Reimplementation of [numpy.argmin]. """
    if __NUMPY_AVAILABLE__:
        return numpy.argmin(arr, axis=1).flatten().tolist()
    row_len = len(arr[0])
    return [min(zip(row, range(row_len)))[1] for row in arr]


def argsort(arr: Iterable) -> list:
    """ Reimplementation of [numpy.argsort]. """
    if __NUMPY_AVAILABLE__:
        return numpy.argsort(arr).tolist()
    arr_sorted = list(sorted(arr))
    raise NotImplementedError()


def concat_arrays(*arrays: Iterable):
    """
    Concatenate multiple arrays into one. This function checks that the inner dimensions of the
    arrays being concatenated match, but the length of the concatenated arrays can be different.
    """
    arr_concat = []
    expected_shape = shape_of_array(arrays[0])[1:]
    for arr in arrays:
        # Input type check
        if not isinstance(arr, ARRAY_LIKE):
            raise TypeError(
                "Unknown array type found. Expected one of %r, found %r" % (ARRAY_LIKE, type(arr))
            )

        # Input shape check
        inner_shape = shape_of_array(arr)[1:]
        if inner_shape != expected_shape:
            raise ValueError(
                "Shape mismatch. Expected all arrays to be of shape %r, found %r"
                % (expected_shape, inner_shape)
            )

        # Convert numpy arrays to list form
        if isinstance(arr, numpy.ndarray):
            arr = arr.tolist()

        # Concatenate to final array
        arr_concat += arr

    # Return all concatenated arrays
    return arr_concat


def take_slice(arr: Iterable, start: int = None, stop: int = None, step: int = None):
    """
    Attempt to take elemenets between [start] and [end] using slicing, fallback to individual item
    addressing.
    """
    if step is None:
        step = 1
    if stop is None:
        stop = len(arr)

    try:
        return arr[start:stop:step]
    except:
        return [arr[i] for i in range(start, stop, step)]


def take_elems(arr: Iterable, elems: list):
    """
    Attempt to take elements using a list and the `__get__` interface, fallback to individual item
    addressing.
    """
    try:
        return arr[elems]
    except:
        return [arr[i] for i in elems]


def take_axis(arr: Iterable, axis: tuple):
    """
    Attempt to take elements given an axis (tuple) and the `__get__` interface, fallback to
    [take_any].
    """
    try:
        return arr[axis]
    except:
        ax = take_any(arr, axis[0])
        if len(axis) > 2:
            return take_axis(ax, axis[1:])
        return take_any(ax, axis[1])


def take_any(arr: Iterable, key: Any):
    """
    Catch-all method to take elements from an array. The behavior will depend on the type of [key].
    """
    try:
        return arr[key]
    except:

        if isinstance(key, tuple):
            if len(key) > 1:
                return take_axis(arr, key)
            else:
                return arr[key[0]]

        if isinstance(key, numpy.ndarray):
            key = key.tolist()

        if isinstance(key, slice):
            return take_slice(arr, key.start, stop=key.stop, step=key.step)

        if isinstance(key, list):
            return take_elems(arr, key)

        raise TypeError(
            "Unknown key type. Expected one of %r, found %r"
            % ((int, slice, list, tuple), type(key))
        )


def array_to_tuple(arr: Iterable):
    """ Recursively convert an array-like object to a tuple """
    if not isinstance(arr, (list, numpy.ndarray)):
        raise TypeError(
            "Unexpected type of input. Expected one of %r, found %r"
            % ((list, numpy.ndarray), type(arr))
        )
    return tuple(
        [array_to_tuple(elem) if isinstance(elem, (list, numpy.ndarray)) else elem for elem in arr]
    )


def unique(arr: Iterable):
    """ Returns all the unique elements in the array. """
    if __NUMPY_AVAILABLE__ and isinstance(arr, numpy.ndarray):
        return numpy.unique(arr).tolist()
    return sorted(list(set(arr)))


def flatten(arr: Iterable):
    """ Returns a 1D list given an array of arbitrary shape by collapsing all its dimensions. """
    if not isinstance(arr, ARRAY_LIKE):
        raise TypeError(
            "Function expected type to be one of %r, found %r" % (ARRAY_LIKE, type(arr))
        )

    # Early exit: use numpy's flatten function if available
    if __NUMPY_AVAILABLE__ and isinstance(arr, numpy.ndarray):
        return numpy.asarray(arr).flatten()

    output = []
    for val in arr:
        if isinstance(val, ARRAY_LIKE):
            output += flatten(val)
        else:
            output.append(val)
    return output


def difference(arr1: Iterable, arr2: Iterable):
    """ Computes the set difference between two arrays. """
    return numpy.setdiff1d(arr1, arr2)


def value_counts(arr: Iterable):
    """ Returns a histogram of all values from the given array. """
    counts = {val: 0 for val in arr}
    for val in arr:
        counts[val] += 1
    counts = reversed(sorted([(val, num) for val, num in counts.items()]))
    return [val for val, num in counts], [num for val, num in counts]


def _is_npy_nan(val):
    if not __NUMPY_AVAILABLE__:
        return False
    try:
        return numpy.isnan(val)
    except TypeError:
        return False


def is_null(arr: Iterable):
    """
    Returns an array of equal length as the input array where each element is True if it is of
    null-like type (i.e. `None`, `numpy.isnan()`, `math.inf` or `float('inf')`). Input array is
    assumed to be 1D.
    """
    assert isinstance(arr, ARRAY_LIKE)
    if __NUMPY_AVAILABLE__:
        try:
            arr = numpy.asarray(arr, dtype=DTYPE_FLOAT[0])
            return ((arr == math.inf) + (arr == float("inf")) + _is_npy_nan(arr)).tolist()
        except Exception as ex:
            pass
    is_inf = lambda val: val == math.inf or val == float("inf")
    return [_is_npy_nan(val) or is_inf(val) or val is None for val in arr]
