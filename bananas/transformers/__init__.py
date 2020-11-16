"""
Extensions of `Learner` objects whose object is to transform data.

Transformers are `Learner` objects that *may* modify the data in some way and, instead of the usual
`predict`, implement the method `transform`. Transformers are typically used as part of a
[Pipeline](../core/index.md#pipeline), serving as a data processing step.

While users are encouraged to write their own, custom transformers, this library aims to provide the
majority of the most commonly used transformers.

## Shape change handling
The main benefit of using one of the built-in transformers is that they all properly handle changes
of input shape, as well as firing of the corresponding output shape events. When implementing our
own custom transformers, we should consider extending `ColumnHandlingTransformer` rather than the
base `BaseTransformer` class.

## Initialization
All of the built-in transformers inherit from `ColumnHandlingTransformer`. They all accept an
argument `columns` in the constructor which describes the features that we wish to transform. This
is so we can leave the other features unmodified, without having to split our data. The `columns`
argument can be a list of indices of the features that we wish to transform, or a dictionary where
the keys are the indices that we wish to transform and the values are the potential values that the
feature may take. This is done so we can accommodate for potential unseen samples ahead of time,
and avoid additional shape changes. If no arguments `columns` is given, it defaults to applying the
transformation to all columns. We can also pass an empty list `[]` to explicitly tell the
transformer that no features should be transformed.
"""

from .base import BaseTransformer
from .encoders import LabelEncoder, OneHotEncoder
from .scalers import MinMaxScaler, StandardScaler
from .running_stats import RunningStats
from .threshold import VarianceThreshold
