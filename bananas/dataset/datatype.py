# Standard imports
from enum import Enum, auto

# Third party imports
import numpy

# Relative imports
from ..utils.arrays import unique
from ..utils.constants import DTYPE_BOOL, DTYPE_FLOAT, DTYPE_INT, DTYPE_UINT8, DTYPE_STR


class DataType(Enum):
    ''' Enum of different kinds of data for a given feature '''
    BINARY = auto()
    ONEHOT = auto()
    CATEGORICAL = auto()
    CONTINUOUS = auto()
    HIGH_DIMENSIOAL = auto()
    IMAGE_PATH = auto()
    UNKNOWN = auto()

    @staticmethod
    def is_categorical(data: (numpy.array, 'DataType')) -> bool:
        '''
        Determines whether a specific piece of data or DataType is categorical or a subset thereof

        Parameters
        ----------
        data : (numpy.array, DataType)
            Numpy array or DataType to analyze

        Returns
        -------
        bool
            Whether the input is of categorical DataType or a subset thereof
        '''
        if not isinstance(data, DataType):
            data = DataType.parse(data)
        return data in (DataType.BINARY, DataType.ONEHOT, DataType.CATEGORICAL)

    @staticmethod
    def parse(data: numpy.ndarray) -> 'DataType':
        '''
        Determine the `DataType` of the given numpy array

        Parameters
        ----------
        data : numpy.ndarray
            Numpy array to analyze

        Returns
        -------
        DataType
            Guess for the type of data from the sample provided
        '''

        # This function assumes that data is already in a numpy array
        assert isinstance(data, numpy.ndarray), 'Unknown data dtype. Expected %r, found %r' % \
            (numpy.ndarray, type(data))

        # Special case: onehot encoding
        if data.ndim == 2 and data.shape[1] > 1 and \
                any([numpy.issubdtype(data.dtype, dtype)
                    for dtype in (DTYPE_BOOL[1], DTYPE_INT[1], DTYPE_UINT8[1])]) and \
                numpy.array_equal(data.astype(int).sum(axis=1), numpy.ones(data.shape[0])) and \
                numpy.array_equal(unique(data.flatten()), numpy.array([0, 1])):
            return DataType.ONEHOT

        # Anything with more than 2 dimensions is considered high-dimensional (typically an image)
        # NOTE: Only allow for numeric types when dealing with high-dimensional data
        if data.ndim > 2 and any([numpy.issubdtype(data.dtype, dtype) \
                for dtype in (DTYPE_BOOL[1], DTYPE_FLOAT[1], DTYPE_INT[1], DTYPE_UINT8[1])]):
            return DataType.HIGH_DIMENSIOAL

        # Cast floats into ints if no precission is lost (i.e. [1., 2., 3.] => [1, 2, 3])
        if numpy.issubdtype(data.dtype, DTYPE_FLOAT[1]) and \
                numpy.all(data == data.astype(DTYPE_INT[0])):
            data = data.astype(DTYPE_INT[0])

        # Exact float type means continuous type
        if numpy.issubdtype(data.dtype, DTYPE_FLOAT[1]):
            return DataType.CONTINUOUS

        # Exact bool type means binary type
        if data.dtype == DTYPE_BOOL[0]:
            return DataType.BINARY

        # Types like bool, str, int or uint type can be categorical or, more specifically, binary
        if any([data.dtype == dtype[0] or numpy.issubdtype(data.dtype, dtype[1])
                for dtype in (DTYPE_BOOL, DTYPE_INT, DTYPE_UINT8, DTYPE_STR)]):
            return DataType.BINARY if len(unique(data)) == 2 else DataType.CATEGORICAL

        # Everything else is unknown
        return DataType.UNKNOWN
