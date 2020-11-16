""" Threshold-based transformers """

from ..changemap.changemap import ChangeMap
from ..statistics.basic import mean, variance
from ..utils.arrays import ARRAY_LIKE, flatten, shape_of_array
from .base import ColumnHandlingTransformer


class RunningStats(ColumnHandlingTransformer):
    """
    Transformer that leaves input unchanged but computes statistics of features using approximation
    techniques, since N is unknown.
    """

    def __init__(self, columns: (dict, ARRAY_LIKE) = None, verbose: bool = False, **kwargs):
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
            for v in col:
                prev_mean = self.mean_.get(i, 0.0)
                self.count_[i] = self.count_.get(i, 0) + 1
                self.mean_[i] = prev_mean + (v - prev_mean) / self.count_[i]
                self._delta_squared_[i] = self._delta_squared_.get(i, 0.0) + (v - self.mean_[i]) * (
                    v - prev_mean
                )

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
