from enum import Enum, auto


class NormalizationStrategy(Enum):
    """ Normalization strategy enum """

    # Automatically select based on data
    AUTO = auto()

    # Apply simple min-max normalization
    MINMAX = auto()

    # Apply normalization assuming normal distribution
    STANDARD = auto()
