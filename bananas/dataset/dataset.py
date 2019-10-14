# Standard imports
import csv
from collections import OrderedDict
from typing import Any, List, Tuple

# Third party imports
import numpy

# Relative imports
from .datatype import DataType
from .feature import Feature
from ..sampling.cross_validation import DataSplit
from ..sampling.strategy import SamplingStrategy
from ..utils.arrays import shape_of_array, transpose_array


class DataSet(object):
    '''
    A `DataSet` is composed of a set of `Feature` objects, and potentially one target which is also
    represented as a `Feature`. A `DataSet` supports indexing similar to numpy arrays and provides
    sampling via the `input_fn()` method.
    '''

    def __init__(self, features: List[Feature], target: Feature = None, name: str = None,
                 sampling_method: SamplingStrategy = None, **sampling_kwargs):
        '''
        Parameters
        ----------
        features : List[Feature]
            Flat list of Features that form this dataset
        target : Feature
            Target feature of this dataset
        sampling_method : SamplingStrategy
            Strategy to use when sampling this dataset via `input_fn()`
        sampling_kwargs
            Options passed to the underlying `BaseSampler` sampling instance
        '''
        # Save internal options
        self.name = name

        # If values is not a list of features, prevent further processing
        if not isinstance(features, list) or \
            not all([isinstance(column, Feature) for column in features]):
            raise TypeError('Provided values should be a Feature list. Found unexpected type: '
                            '%r' % type(features))

        # Convert each column into our own Feature type
        self.features = OrderedDict([(col.name or str(i), col) for i, col in enumerate(features)])
        self.input_sample = {name: feat.input_sample for name, feat in self.features.items()}

        # Get target as a Feature type from provided data
        if target is not None:
            assert isinstance(target, Feature), \
                'Parameter `target` must be of type Feature, found: %r' % type(target)
            self.target = target

        # Make sure that all features and target have the same number of samples
        first_column: Feature = next(iter(self.features.values()))
        self.count = first_column.count
        for feat in self.features.values():
            assert feat.count == self.count, ('Features do not have equal number of samples. ' \
                'Expected %d, found %d' % (self.count, feat.count))
        if hasattr(self, 'target'):
            assert self.target.count == self.count, ('Target does not have equal number of ' \
                'samples as features. Expected %d, found %d' % (self.count, self.target.count))

        # Initialize sampling strategy: use target as data if available, otherwise first column
        sampling_values = getattr(self, 'target', first_column).values
        self.sampler = SamplingStrategy.create(sampling_values, sampling_method, **sampling_kwargs)

        # Identify categorical features from sample
        self.categorical = {
            i: col.sample_classes for i, col in enumerate(self.features.values())
            if (col.kind == DataType.CATEGORICAL or col.kind == DataType.BINARY)}

        # Identify continuous features from sample
        def _continuous_stats(col: Feature):
            return {
                'min': col.sample_min,
                'max': col.sample_max,
                'mean': col.sample_mean,
                'var': col.sample_var}
        self.continuous = {
            i: _continuous_stats(col) for i, col in enumerate(self.features.values())
            if col.kind == DataType.CONTINUOUS}

    def ix(self, key: (Tuple[List[str], Any], Any), transpose: bool = False):
        '''
        Main indexing method. If `key` is a `Tuple`, the first item must be a list containing the
        names of `Feature`s (or an *everything* slice ":") and the second item the indexing key;
        then a single `Feature` with the indexing applied will be returned. Otherwise, it is assumed
        to just be the indexing key and a list of all `Feature`s with the indexing applied will be
        returned.

        Indexing a `DataSet` using square brackets maps directly to this method, except there is no
        option to transpose the output.

        Parameters
        ----------
        key : (Tuple[str, Any], Any)
            Indexing key; if it's a `Tuple` then the first item must be a list containing the names
            of `Feature`s (or an *everything* slice ":")
        transpose : bool
            Whether to transpose the output to return a list of samples instead of a list of
            `Feature` columns (and potentially a target)
        '''
        # Assume index key applies to all features
        names = list(self.features.keys())

        # If a tuple is given, first element is the name(s) of the feature
        if isinstance(key, tuple): names, key = key[0], key[1:]

        # Special case: if names is an "everything" slice (i.e. ":"), return all columns
        if isinstance(names, slice) and \
            names.start is None and names.stop is None and names.step is None:
            names = list(self.features.keys())

        # Names must be a list
        if not isinstance(names, list): names = [names]

        # Index each of the selected features
        features = [feat[key] for name, feat in self.features.items() if name in names]

        # Case 1: single feature selected
        if len(features) == 1:
            features = features[0]

        # Case 2: features must be zipped together
        elif transpose:
            features = transpose_array(features)

        # Return features alone when there's no target
        if not hasattr(self, 'target'): return features

        # Otherwise, return a tuple of features, target
        return features, self.target[key]

    def __getitem__(self, key):
        return self.ix(key)

    def __len__(self):
        return self.count

    def input_fn(self, subset: DataSplit = DataSplit.TRAIN, batch_size: int = None):
        '''
        Utility sampling function that draws samples from all the underlying features. The output
        samples are *aligned* -- i.e. the same indices are used to capture items from all features.

        Parameters
        ----------
        subset : DataSplit
            subset to sample from
        batch_size : int
            number of samples to draw, falls back to internal sampler defaults (see constructor)
        '''
        # Use sampler to get indices of samples
        idx = self.sampler.indices(subset=subset, batch_size=batch_size)
        # Return the same as indexing would
        return self.ix(idx)

    @staticmethod
    def from_dataframe(values, target: str = None, **dataset_kwargs):
        ''' Load data from a pandas DataFrame '''
        raise NotImplementedError()  # TODO

    @staticmethod
    def from_ndarray(values: numpy.ndarray, target: numpy.ndarray = None, name: str = None,
                     sampling_method: str = None, **sampling_kwargs):
        ''' Load data from a numpy array '''
        assert isinstance(values, numpy.ndarray), \
            'Parameter `values` must be an ndarray, found %r' % type(values)
        ndim = len(shape_of_array(values))
        if ndim == 1:
            values = [values.reshape(-1) if hasattr(values, 'reshape') else values]
        elif ndim > 1:
            values = [values[i] for i in range(len(values))]

        values = [Feature(col, name=str(i), sampling_method=sampling_method, **sampling_kwargs)
                  for i, col in enumerate(values)]
        if target is not None:
            target = Feature(target, sampling_method=sampling_method, **sampling_kwargs)

        return DataSet(
            values, target=target, name=name, sampling_method=sampling_method, **sampling_kwargs)

    @staticmethod
    def from_csv(filename: str, target_column: int = -1, delimiter: str = ',', header: bool = True,
                 encoding: str = None, name: str = None,
                 sampling_method: SamplingStrategy = None, **sampling_kwargs):
        '''
        Load data from a symbol-delimited file, defaults to reading a CSV file given its path.

        Parameters
        ----------
        filename : str
            Path of the file to read values from
        target_column : int
            Column corresponding to the target feature, set as `None` if no target exists
        delimiter : str
            Delimiter to use to separate records, defaults to ","
        header : bool
            Whether the file has a header as the first record
        encoding : str
            Encoding to use to read the file, defaults to auto
        sampling_method : SamplingStrategy
            Strategy used to sample data from this DataSet
        sampling_kwargs : dict
            Parameters to pass to the sampler during initialization
        '''

        with open(filename) as fh:
            rows = numpy.genfromtxt(
                ('\t'.join(line) for line in csv.reader(fh, delimiter=delimiter)),
                dtype=None, names=(header or None), delimiter='\t', encoding=encoding)

        names = [name if header else None for name in rows.dtype.names]
        columns = [Feature(col, name=name, sampling_method=sampling_method, **sampling_kwargs)
                   for name, col in zip(names, transpose_array(rows.tolist()))]

        target = None
        if target_column is not None:
            target = columns[target_column]

        features = [col for col in columns if col != target]
        return DataSet(
            features, target=target, name=name, sampling_method=sampling_method, **sampling_kwargs)
