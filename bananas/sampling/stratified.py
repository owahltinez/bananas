from .cross_validation import RandomCVSampler
from ..statistics.random import RandomState
from ..utils.arrays import take_elems, unique
from ..utils.constants import SAMPLE_SIZE_LARGE


class StratifiedSampler(RandomCVSampler):
    """
    Sampler that outputs a set distribution of values. For example, using a dataset that has a
    distribution of 80% of class A and 20% of class B, each batch is expected to contain roughly
    the same distribition minus rounding error sbased on batch size. For a sufficiently large
    dataset, using this sampler as-is should be roughly equivalent.

    Passing the `distribution` argument allows users to override the output distribution. So, if
    we have an imbalanced dataset like the example above, we could request for each batch to output
    50% for each class by setting `distribution = {'A': .5, 'B': .5}`.
    """

    def __init__(
        self,
        data,
        batch_size: int = 128,
        input_size: int = None,
        test_split: float = 0.2,
        validation_split: float = None,
        distribution: dict = None,
        random_seed: int = None,
    ):
        """
        Parameters
        ----------
        data
            TODO
        distribution : dict
            TODO
        test_split : float
            TODO
        batch_size : int
            TODO
        input_size : int
            TODO
        random_seed : int
            TODO
        """
        super().__init__(
            data,
            batch_size=batch_size,
            input_size=input_size,
            test_split=test_split,
            validation_split=validation_split,
            random_seed=random_seed,
        )

        # Initialize internal objects
        self.batch_size = batch_size
        self.rnd = RandomState(seed=random_seed)

        # If no distribution is given, we assume that we want to preserve original distribution
        # So we need to compute original distribution from some random sampling ourselves
        if distribution is None:
            # We can't use __call__ directly here because it will call our indices() before setup
            sample = (
                data
                if self.input_size < SAMPLE_SIZE_LARGE
                else self.rnd.choice(data, SAMPLE_SIZE_LARGE, replace=False)
            )
            distribution = {
                label: len([x for x in sample if x == label]) / len(sample)
                for label in unique(sample)
            }
        self.distribution = distribution

    def indices(self, subset: str = None, batch_size: int = None):

        # Batch size can be overriden if passed as an argument
        if batch_size is None:
            batch_size = self.batch_size

        # Oversample until we have batch_size of each class
        # TODO: We can optimize this to just the minimum number of samples required
        oversample = {label: [] for label in self.distribution.keys()}
        while any([len(oversample[label]) < batch_size for label in self.distribution.keys()]):
            idx = super().indices(subset=subset, batch_size=batch_size)
            for i, label in zip(idx, take_elems(self.data, idx)):
                oversample[label].append(i)

        # Now that we have enough samples, randomly select the right amount of each label
        sample = []
        for label, ratio in self.distribution.items():
            idx = self.rnd.choice(oversample[label], round(batch_size * ratio), replace=False)
            sample += idx.tolist()

        self.rnd.shuffle(sample)
        return sample
