from .ordered import OrderedSampler
from..utils.arrays import unique
from ..statistics.random import RandomState

class RandomSampler(OrderedSampler):
    '''
    Random sampling produces samples uniformly distributed from the input data. Individual samples are
    not guaranteed to be unique in the output, they can be repeated by the process of random sampling.
    The simplest usage of `RandomSampler` is no different than `OrderedSampler`:

    ```python
    sampler = RandomSampler(dataset)
    batch = sampler(batch_size=10)  # [31, 90, 60, 5, 82, 79, 13, 20, 78, 33]

    ### Random seed
    A random seed can be provided to the `RandomSampler` to enable reproducible results. Internally,
    the random seed is passed to the random number generator upon initialization:

    ```python
    sampler = RandomSampler(dataset, random_seed=0)
    batch = sampler(batch_size=10)  # [7, 59, 72, 16, 92, 81, 58, 34, 95, 28]
    ```
    '''

    def __init__(self, data, batch_size: int = 128, epochs: int = None,
                 input_size: int = None, blacklist: set = None, random_seed: int = None):
        '''
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
        random_seed : int
            TODO
        '''
        super().__init__(
            data, batch_size=batch_size, epochs=epochs, input_size=input_size, blacklist=blacklist)

        # Get internal entropy provider
        self.rnd_pool_ = []
        self.rnd = RandomState(seed=random_seed)

    def _next_index(self):
        ''' Returns a random index between 0 and input_size '''
        # Cache a string of random numbers to speed things up
        if not self.rnd_pool_:
            self.rnd_pool_ = self.rnd.randint(0, self.input_size - 1, self.batch_size * 10).tolist()

        return self.rnd_pool_.pop()
