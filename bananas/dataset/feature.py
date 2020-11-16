from typing import Any
from ..sampling.strategy import SamplingStrategy, ReplaceStrategy
from ..utils.arrays import (
    argwhere,
    check_array,
    flatten,
    is_null,
    shape_of_array,
    take_any,
    unique,
    value_counts,
)
from ..utils.constants import ARRAY_LIKE, SAMPLE_SIZE_SMALL
from ..utils.images import open_image
from .datatype import DataType


class Feature(object):
    """
    This class represents a distinct feature. A set of features form a dataset. Each feature is
    normally a single column but it can be composed of two or more columns; for example images and
    one-hot encodings are multi-column single features.
    """

    def __init__(
        self,
        values: ARRAY_LIKE,
        kind: DataType = None,
        name: str = None,
        replace_strategy: ReplaceStrategy = ReplaceStrategy.MEAN,
        sample_size: int = SAMPLE_SIZE_SMALL,
        sampling_method: SamplingStrategy = None,
        **sampling_kwargs
    ):
        """
        Parameters
        ----------
        values : ARRAY_LIKE
            Values that comprise this feature
        kind : DataType
            Type of data, if known -- otherwise it will be inferred at runtime
        name : str
            Name of this feature, useful for indexing if part of a `DataSet`
        replace_strategy : ReplaceStrategy
            Strategy used to replace missing values (i.e. NaN or None) during sampling
        sampling_method : SamplingStrategy
            Method used to draw samples from the input values
        sampling_kwargs : dict
            Arguments passed to the sampling constructor
        """
        # assert isinstance(values, ARRAY_LIKE), \
        assert hasattr(
            values, "__getitem__"
        ), "Parameter `values` must be array-like, found %r" % type(values)

        # Save internal variables
        self.name = name
        self.values = values

        # Initialize sampling strategy
        self.sampler = SamplingStrategy.create(values, sampling_method, **sampling_kwargs)

        # Take a sample from the data source for analysis
        self.input_sample = check_array(self.sampler(batch_size=sample_size), allow_nan=True)

        # Extract the input shape. Note that we will only use the sample for this.
        self.shape = shape_of_array(self.input_sample)
        self.count = len(values)

        # Get the feature kind from it
        self.kind: DataType = DataType.parse(self.input_sample) if kind is None else kind
        assert isinstance(self.kind, DataType) and self.kind != DataType.UNKNOWN, (
            "Unknown data kind found: %r" % self.kind
        )

        # Store replacement for NaN values as we draw samples
        self.replace_strategy = replace_strategy

        # Compute basic stats of sample data
        self.sample_min: float = None
        self.sample_max: float = None
        self.sample_var: float = None
        self.sample_mean: float = None
        self.sample_classes = []
        if self.kind == DataType.HIGH_DIMENSIOAL:
            pass
            # Compute the variance based on the diff of each channel from each image with others
            # TODO
            # mean = sum(x) / len(x)
            # variance = sum((x - mean ** 2) / (len(x) - 1);

        if self.kind == DataType.IMAGE_PATH:
            pass
            # TODO: load image and compute basic stats

        if self.kind == DataType.CONTINUOUS or self.kind == DataType.HIGH_DIMENSIOAL:
            # Ignore all NaN values
            nulls = is_null(flatten(self.input_sample))
            input_sample_ = [val for val, nan in zip(self.input_sample, nulls) if not nan]
            # Use transformer to compute sample stats
            # NOTE: import needs to be done here to avoid circular dependency
            from ..transformers.running_stats import RunningStats

            sample_stats = RunningStats(columns=[0]).fit(input_sample_)
            self.sample_min = sample_stats.min_[0]
            self.sample_max = sample_stats.max_[0]
            self.sample_var = sample_stats.variance_[0]
            self.sample_mean = sample_stats.mean_[0]

        if self.kind == DataType.CATEGORICAL or self.kind == DataType.BINARY:
            # TODO: compute Shanon's entropy for var?
            self.sample_classes = unique(self.input_sample)
            labels, _ = value_counts(self.input_sample)
            self.sample_min = min(labels)
            self.sample_max = max(labels)
            self.sample_mean = labels[0]

        if self.kind == DataType.BINARY:
            pass

        if self.kind == DataType.ONEHOT:
            # TODO: compute Shanon's entropy for var?
            pass

    def ix(self, key: Any, replace_strategy: ReplaceStrategy = None):
        """
        Main indexing method. Indexing a `Feature` using square brackets maps directly to this
        method, except there is no option to set replace stratrgy for the output.

        Parameters
        ----------
        key : Any
            Indexing key (which can be an *everything* slice ":")
        replace_strategy : ReplaceStrategy
            Strategy used to replace missing values (i.e. NaN or None) during sampling
        """
        values = take_any(self.values, key)

        # As long as it's not a one-hot encoded or image column, we can replace null values
        not_replaceable_kind = (DataType.ONEHOT, DataType.HIGH_DIMENSIOAL, DataType.IMAGE_PATH)
        if replace_strategy is not None and self.kind not in not_replaceable_kind:
            is_single = not isinstance(values, ARRAY_LIKE)
            nulls = is_null([values] if is_single else values)
            replace_value = {ReplaceStrategy.MEAN: self.sample_mean}.get(replace_strategy)

            # Early exit: single value needs replacing
            if is_single and nulls[0] is True:
                return replace_value

            null_idx = argwhere(nulls)
            if replace_strategy == ReplaceStrategy.DROP:
                values = [val for idx, val in enumerate(values) if idx not in null_idx]
            else:
                for idx in null_idx:
                    values[idx] = replace_value

        # Special case: image paths need to be loaded from disk
        if self.kind == DataType.IMAGE_PATH:
            values = [open_image(path) for path in values]

        # Wrap image into a list to ensure it's treated as a single feature
        # if self.kind == DataType.IMAGE:
        #     values = [values]
        return values

    def __getitem__(self, key):
        return self.ix(key, replace_strategy=self.replace_strategy)

    def __len__(self):
        return self.values.__len__()
