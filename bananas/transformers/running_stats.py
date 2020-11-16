""" Threshold-based transformers """

from typing import Dict, Iterable, Union
from ..changemap.changemap import ChangeMap
from ..utils.arrays import flatten, shape_of_array
from .base import ColumnHandlingTransformer


class RunningStats(ColumnHandlingTransformer):
    """
    A helpful transformer that does not perform any transformations is `RunningStats`. It implements a
    number of running statistics on the requested features that can be used by other transformers
    expending that one. Keep reading below for several examples of transformers that extend
    `RunningStats`. Here's an illustration of what `RunningStats` can do:

    ```python
    arr = [random.random() for _ in range(100)]
    transformer = RunningStats()
    transformer.fit(arr)
    transformer.print_stats()
    # Output:
    # col	min_	max_	mean_	count_	stdev_	variance_
    # 0 	0.001	0.996	0.586	100.000	0.264	0.070
    ```
    """

    def __init__(self, columns: Union[Dict, Iterable[int]] = None, verbose: bool = False, **kwargs):
        """
        Parameters
        ----------
        columns : Union[Dict, Iterable[int]]
            TODO
        verbose : bool
            TODO
        """
        super().__init__(columns=columns, verbose=verbose, **kwargs)

        # Initialize working variables
        self.max_ = {}
        self.min_ = {}
        self.mean_ = {}
        self.count_ = {}
        self.stdev_ = {}
        self.variance_ = {}
        self._delta_squared_ = {}

    def fit(self, X):
        X = self.check_X(X)

        for i, col in enumerate(X):
            if i not in self.columns_:
                continue

            # High dimensional data, like images, is treated as a 1D list
            shape = shape_of_array(col)
            if len(shape) > 1:
                col = flatten(col)

            # Computing max / min is trivial
            sample_max = max(col)
            sample_min = min(col)
            self.max_[i] = max(self.max_.get(i, sample_max), sample_max)
            self.min_[i] = min(self.min_.get(i, sample_min), sample_min)

            # Use on-line algorithm to compute variance, which unfortunately requires iterating
            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
            for val in col:
                prev_mean = self.mean_.get(i, 0.0)
                self.count_[i] = self.count_.get(i, 0) + 1
                self.mean_[i] = prev_mean + (val - prev_mean) / self.count_[i]
                self._delta_squared_[i] = self._delta_squared_.get(i, 0.0) + (
                    val - self.mean_[i]
                ) * (val - prev_mean)

            self.variance_[i] = self._delta_squared_[i] / self.count_[i]
            self.stdev_[i] = self.variance_[i] ** 0.5

        return self

    def on_input_shape_changed(self, change_map: ChangeMap):
        # Parent's callback will take care of adapting feature changes
        super().on_input_shape_changed(change_map)
        # We still need to adapt feature changes to internal data
        self._input_change_column_adapter(
            change_map, ["min_", "max_", "mean_", "count_", "stdev_", "variance_"]
        )

    def print_stats(self):
        stats = ["min_", "max_", "mean_", "count_", "stdev_", "variance_"]
        print()
        print("\t".join(["col"] + stats))
        for col in self.columns_.keys():
            print(
                "%d\t%s" % (col, "\t".join(["%.03f" % getattr(self, stat)[col] for stat in stats]))
            )
        print()
