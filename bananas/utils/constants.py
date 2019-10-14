''' Module with constants used across a number of utilities, methods and classes '''

from ..utils import __NUMPY_AVAILABLE__
if __NUMPY_AVAILABLE__:
    import numpy

    # Iterable type represents all types that can be considered a vector
    ARRAY_LIKE = (numpy.ndarray, list, tuple)

    # For each of the supported dtypes, define a pair of exact dtype, and the base dtype
    DTYPE_BOOL = (numpy.dtype(bool), numpy.bool_)  # pylint: disable=no-member
    DTYPE_FLOAT = (numpy.dtype(float), numpy.floating)  # pylint: disable=no-member
    DTYPE_INT = (numpy.dtype(int), numpy.integer)  # pylint: disable=no-member
    DTYPE_STR = (numpy.dtype(str), numpy.str_)  # pylint: disable=no-member
    DTYPE_BYTES = (numpy.dtype(bytes), numpy.bytes_)  # pylint: disable=no-member
    DTYPE_UINT8 = (numpy.dtype(numpy.uint8), numpy.integer)  # pylint: disable=no-member

else:
    # Provide alternative constants for systems without numpy
    from numbers import Number

    ARRAY_LIKE = (list, tuple)
    DTYPE_BOOL = (bool, bool)
    DTYPE_FLOAT = (float, Number)
    DTYPE_INT = (int, Number)
    DTYPE_STR = (str, str)
    DTYPE_BYTES = (bytes, bytes)
    DTYPE_UINT8 = (int, Number)


# These are the dtypes that estimators can expect
ALLOWED_DTYPES = [dtype[0] for dtype in
                  [DTYPE_BOOL, DTYPE_FLOAT, DTYPE_INT, DTYPE_STR, DTYPE_UINT8]]

# Used for classifying datasets as well as determining the size of sample drawings
SAMPLE_SIZE_SMALL = int(1E3)
SAMPLE_SIZE_LARGE = int(1E6)

# Used for determining if one-hot encoder is the most efficient way to encode a categorical feature
ONEHOT_MAX_CLASSES = 8
