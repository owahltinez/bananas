'''
These are constants used to evaluate *loss* from an ML model perspective. The loss functions
themselves are not implemented, it is expected that the ML framework underneath implements them
in an efficient way since the loss functions will be tightly coupled with the learning process. In
the future, placeholder functions that can be overridden might be provided similarly to how the
`Scoring` submodule does.
'''

from enum import Enum, auto


class LossFunction(Enum):
    ''' Enum declaring different types of loss function '''

    L1 = auto()
    MSE = auto()
    CROSS_ENTROPY = auto()
    BINARY_CROSS_ENTROPY = auto()

    @staticmethod
    def create(kind: 'LossFunction'):
        '''
        Get an instance of the requested loss function, framework dependent implementation.

        Parameters
        ----------
        kind : LossFunction
            The specific type of loss function that an instance is being created for
        '''
        raise NotImplementedError('This method should be overridden')
