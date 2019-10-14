from enum import Enum, auto

class EncodingStrategy(Enum):
    ''' Encoding strategy enum '''

    # Automatically select based on number of classes
    AUTO = auto()

    # Drop all features that require encoding
    DROP = auto()

    # Encode classes into ordinal integers
    ORDINAL = auto()

    # Enclode classes into one-hot vectors
    ONEHOT = auto()
