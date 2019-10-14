'''
Cross validation samplers are a special kind of samplers that divide the input data into separate
subsets.

The possible subsets are:

- Train: A subset to train a model.
- Test: A subset to test the trained model.
- Validation: A subset to evaluate the results of the train-test set with the trained model.

For more information about the data splits, check out [this resource](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data)
for the train-test data split and [this documentation](https://developers.google.com/machine-learning/crash-course/validation/another-partition)
for the validation set.

## Data splits
By default, the train-test split is 80/20. As in, 80% of the input data belongs to the train subset
and 20% to the test subset. If the input data contains more than 100,000 samples, the test subset is
further split into 80/20 between test and validation subsets.

The percentage in data splits can be overriden using the `test_split` argument. For example, if we
wanted to dedicate 10% of the input data to the test subset, we could do the following:

```python
sampler = RandomCVSampler(dataset, test_split=.10)
```

To draw samples from a specific data split, we use the `subset` argument in the `__call__` function
and pass one of the `DataSplit` enum values:

```python
sampler = RandomCVSampler(dataset, random_seed=0)
sampler(batch_size=10, subset=DataSplit.TRAIN)  # [51, 77, 18, 3, 20, 36, 18, 78, 60, 26]
sampler(batch_size=10, subset=DataSplit.TEST)  # [67, 88, 83, 67, 44, 46, 58, 65, 12, 67]
```

## Types
Cross validation samplers come in two flavors: ordered and random. They correspond to the similarly
named samplers, with the addition of data splits. A very commonly used sampling technique in ML
is random cross-validation.
'''

import sys
from enum import Enum, auto
from typing import Dict

from .base import _BaseSampler
from .ordered import OrderedSampler
from .random import RandomSampler
from ..statistics.random import RandomState
from ..utils.arrays import take_elems

# Constant definitions
VALIDATION_MIN_SAMPLE_SIZE = 1E5
''' Minimum amount of samples required to create a validation subset '''

class SubsetNotFoundError(Exception):
    ''' Error used when users attempt to use a subset that does not exist in a sampler '''

class DataSplit(Enum):
    ''' Enum describing the subset of data to be sampled '''
    TRAIN = auto()
    TEST = auto()
    VALIDATION = auto()


class _CVSampler(_BaseSampler):
    def __init__(self, data, batch_size: int = 128, epochs: int = None, input_size: int = None,
                 test_split: float = 0.2, validation_split: float = None):
        '''
        Parameters
        ----------
        data
            TODO
        test_split : float
            TODO
        beatch_size : int
            TODO
        epochs : int
            TODO
        input_size : int
            TODO
        '''
        super().__init__(self)

        # Sanitize input size
        self.input_size = input_size or (len(data) if hasattr(data, '__len__') else sys.maxsize)
        assert self.input_size > 0, 'Input size must be greater than zero, found data: %r' % data

        # Compute test size based on input size, fallback to 10 * `test_split` * `batch_size`
        test_size = int((self.input_size * test_split) if self.input_size != sys.maxsize \
            else (10 * test_split * batch_size))

        # Create a subset of the test set used for validation if we have enough samples
        validation_set = validation_split is not None
        validation_split = validation_split or test_split
        self.validation_size = int(test_size * validation_split) \
            if self.input_size > VALIDATION_MIN_SAMPLE_SIZE or validation_set else 0

        # Adjust all subset sizes now that all have been computed
        self.train_size = self.input_size - test_size
        self.test_size = test_size - self.validation_size

        # Compute shape of input by looking at first item
        # TODO: use ndim instead of X[0].shape?
        self.input_shape = (1,)
        if hasattr(data[0], 'shape'):
            self.input_shape = data[0].shape
        elif hasattr(data[0], '__len__'):
            self.input_shape = (len(data[0]),)

        # Copy all arguments to internal objects
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size

        # To be populated by subclasses
        self.subsamplers = {}

    def indices(self, batch_size: int = None, subset: DataSplit = DataSplit.TRAIN):

        # By default assume the train subset
        if subset is None: subset = DataSplit.TRAIN

        # Verify that the requested subset is among the available ones
        if subset not in self.subsamplers:
            raise SubsetNotFoundError('Subset "%s" not found, expected one of %r' %
                                      (subset, list(self.subsamplers.keys())))

        # If this is the default subset, we return the indices like we normally do
        if subset == DataSplit.TRAIN:
            return self.subsamplers[subset].indices(batch_size=batch_size)

        # Otherwise, each subsampler actually returns the indices of indices, so we go down a level
        return self.subsamplers[subset](batch_size=batch_size)

    def __call__(self, batch_size: int = None, subset: DataSplit = DataSplit.TRAIN):
        return take_elems(getattr(self, 'data'), self.indices(batch_size=batch_size, subset=subset))


class OrderedCVSampler(_CVSampler):
    '''
    Sampler equivalent to `OrderedSampler` that divides input into train, test and validation
    subsets to perform independent sampling.
    '''

    def __init__(self, data, batch_size: int = 128, epochs: int = None, input_size: int = None,
                 blacklist: set = None, test_split: float = 0.2, validation_split: float = None):
        '''
        Parameters
        ----------
        data
            TODO
        test_split : float
            TODO
        beatch_size : int
            TODO
        blacklist : set
            TODO
        epochs : int
            TODO
        input_size : int
            TODO
        '''
        super().__init__(
            data, batch_size=batch_size, epochs=epochs, input_size=input_size,
            test_split=test_split, validation_split=validation_split)

        # Instantiate a subsampler for each subset
        self.subsamplers: Dict[DataSplit, _BaseSampler] = {}
        self.subsamplers[DataSplit.TRAIN] = OrderedSampler(
            data, batch_size=self.batch_size, input_size=self.train_size, blacklist=blacklist,
            epochs=self.epochs)

        if self.test_size > 0:
            ix0 = self.train_size
            test_ix = list(range(ix0, ix0 + self.test_size))
            self.subsamplers[DataSplit.TEST] = OrderedSampler(
                test_ix, batch_size=self.batch_size, input_size=self.test_size)

        if self.validation_size > 0:
            ix0 = self.train_size + self.test_size
            validate_ix = list(range(ix0, ix0 + self.validation_size))
            self.subsamplers[DataSplit.VALIDATION] = OrderedSampler(
                validate_ix, batch_size=self.batch_size, input_size=self.validation_size)


class RandomCVSampler(_CVSampler):
    '''
    Sampler equivalent to `RandomSampler` that divides input into train, test and validation
    subsets to perform independent sampling.
    '''
    def __init__(self, data, batch_size: int = 128, epochs: int = None, input_size: int = None,
                 test_split: float = 0.2, validation_split: float = None, random_seed: int = None):
        '''
        Parameters
        ----------
        data
            TODO
        test_split : float
            TODO
        beatch_size : int
            TODO
        epochs : int
            TODO
        input_size : int
            TODO
        random_seed : int
            TODO
        '''
        super().__init__(data, batch_size=batch_size, epochs=epochs, input_size=input_size,
                         test_split=test_split, validation_split=validation_split)

        # Initialize internal objects
        self.rnd = RandomState(seed=random_seed)

        # Compute validation subset indices first
        validate_idx = set(self.rnd.randint(0, self.input_size - 1, self.validation_size))

        # Create a test subset using indices not found in validation subset
        test_idx = set()
        iter_max = self.test_size * 1000
        for i in range(iter_max):
            idx = self.rnd.randint(0, self.input_size - 1)
            if idx not in validate_idx: test_idx.add(idx)
            if len(test_idx) == self.test_size: break
            if i >= iter_max - 1:
                raise RuntimeError('Unable to create test subset after %d iterations' % iter_max)

        # Create a blacklist for training which includes all items in both test and validation sets
        non_training = test_idx.union(validate_idx)

        # Instantiate a subsampler for each subset
        self.subsamplers = {}
        self.subsamplers[DataSplit.TRAIN] = RandomSampler(
            data,
            epochs=epochs,
            batch_size=self.batch_size,
            input_size=self.train_size,
            blacklist=non_training,
            random_seed=random_seed)

        if self.test_size > 0:
            self.subsamplers[DataSplit.TEST] = RandomSampler(
                list(test_idx),
                batch_size=self.batch_size,
                input_size=self.test_size,
                random_seed=random_seed)

        if self.validation_size > 0:
            self.subsamplers[DataSplit.VALIDATION] = RandomSampler(
                list(validate_idx),
                batch_size=self.batch_size,
                input_size=self.validation_size,
                random_seed=random_seed)
