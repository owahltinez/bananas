"""
Utilities that perform sampling of data using arbitrary criteria.

Sampling is a fundamental part of learning large amounts of data which may not fit in
memory at once.

## Interface

All samplers implement the same base interface, which means that the constructor **always** accepts
the following parameters:

- `data`: This is the original data being sampled. Anything implementing the `__get__` interface
  will suffice. See below for an example implementation of this parameter that does not load all
  values into memory.
- `batch_size`: How many values get returned each time the samples are drawn.
- `epochs`: How many times the original data can be iterated over until an `IndexError` is thrown.
  Defaults to `None` which is *infinite*.
- `input_size`: If the provided `data` does not implement a `__len__` method or we prefer to
  override it, we can use this parameter to set the maximum index that will be drawn from input.

## Custom data class
Here is how we can implement a dataset that is compatible with sampling and does not load all data
into memory at once:

```python
class MyDataset(object):
    \'\'\' Dataset that returns [self.columns] of repeating values equal to the index. \'\'\'

    columns = 8
    def __getitem__(self, idx):
        return [idx for _ in self.columns]

# Example usage:
dataset = MyDataset()
dataset[1]  # [1, 1, 1, 1, 1, 1, 1, 1]
dataset[8]  # [8, 8, 8, 8, 8, 8, 8, 8]
```

## `__call__`
The fundamental way to use the sampler is by using the provided `__call__` interface. For example:

```python
sampler = OrderedSampler(dataset)
batch = sampler()  # [[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], ...]
```

## `indices`
Samplers also provide an `indices()` function. Instead of the actual values, that function returns
the indices of the values to be sampled from the input data. Internally, `__call__` uses this
function to retrieve the indices and then the values are indexed from the input data.

```python
sampler = OrderedSampler(dataset)
batch = sampler.indices()  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```
"""

from enum import Enum, auto
from .ordered import OrderedSampler
from .random import RandomSampler
from .cross_validation import DataSplit, OrderedCVSampler, RandomCVSampler
from .stratified import StratifiedSampler
