import sys
from .base import _BaseSampler


class OrderedSampler(_BaseSampler):
    """
    Ordered sampling is the most basic form of sampling. It consists of simply iterating over the input
    data and returning it in batches. For example, given a dataset of [0, 1, 2, 3, ... , 99], performing
    ordered sampling using a batch size of 10 would result in the following samples:

    * Batch #1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    * Batch #2: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    * Batch #3: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    * ...
    * Batch #10: [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

    To perform ordered sampling, we can use the following code:

    ```python
    sampler = OrderedSampler(dataset)
    batch = sampler(batch_size=10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ```

    ## Blacklist
    `OrderedSampler` also accepts a `blacklist` argument, containing a set of indices from which the
    input is not to be drawn -- those indices will be skipped instead. Here's a sample usage of the
    `blacklist` argument:

    ```python
    sampler = OrderedSampler(dataset, blacklist=set([2, 5, 10]))
    batch = sampler(batch_size=10)  # [0, 1, 3, 4, 6, 7, 8, 9, 11, 12]
    ```
    """

    def __init__(
        self,
        data,
        batch_size: int = 128,
        epochs: int = None,
        input_size: int = None,
        blacklist: set = None,
    ):
        """
        Parameters
        ----------
        data
            TODO
        batch_size : int
            TODO
        epochs : int
            TODO
        input_size : int
            TODO
        blacklist : set
            TODO
        """
        super().__init__(self)

        # Sanitize input size
        self.input_size = input_size or (len(data) if hasattr(data, "__len__") else sys.maxsize)
        assert self.input_size > 0, "Input size must be greater than zero, found data: %r" % data

        # Validate that batch size is correct
        if epochs is not None and batch_size > self.input_size:
            raise ValueError(
                "Batch size [%d] cannot be larger than input size [%d]" % (batch_size, input_size)
            )

        # Compute shape of input by looking at first item
        first = data[0]
        if hasattr(first, "shape"):
            self.input_shape = first.shape
        elif hasattr(first, "__len__") and not isinstance(data, str):
            self.input_shape = (len(first),)
        else:
            self.input_shape = (1,)

        # Copy all arguments to internal objects
        self.data = data
        self.idx_curr = 0
        self.epoch_curr = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.blacklist = None if blacklist is None else set(blacklist)

    def _next_index(self):
        """ By default the next index is the current index which is auto-incremented """
        return self.idx_curr

    def indices(self, batch_size: int = None):
        """ Generates a batch of (features, target) samples """
        batch_idx = []

        # Batch size can be overriden if passed as an argument
        if batch_size is None:
            batch_size = self.batch_size

        while len(batch_idx) < batch_size:

            # Check that we didn't go over the max epochs
            if self.epochs is not None and self.epoch_curr >= self.epochs:
                raise IndexError("Maximum number of epochs reached: %d" % self.epochs)

            # Call internal function to select index
            # Subclasses are expected to override this function
            idx = self._next_index()

            # Add index to batch if not in blacklist
            if self.blacklist is None or idx not in self.blacklist:
                batch_idx.append(idx)

            # Increment index and, if finished, increment epoch and loop around
            self.idx_curr += 1
            if self.idx_curr >= self.input_size:
                self.idx_curr = 0
                self.epoch_curr += 1

        # Return batch as list of indices
        return batch_idx
