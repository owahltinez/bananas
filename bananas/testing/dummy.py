"""
This module includes dummy implementations for a classifier, a regressor and a transformer that can
be used as a sample implementation but is also used for testing purposes. Take a peek at the [source
code](dummy.py) for more details.
"""

from typing import Any, List
from ..changemap.changemap import ChangeMap
from ..core.mixins import BaseClassifier, BaseRegressor
from ..statistics.random import RandomState
from ..statistics.scoring import ScoringFunction
from ..transformers.base import ColumnHandlingTransformer
from ..utils.constants import ARRAY_LIKE, DTYPE_FLOAT


class DummyClassifier(BaseClassifier):
    """
    Dummy implementation of a classifier used for testing purposes. It can output a random value
    or the expected value based on prior data depending on the strategy selected.
    """

    def __init__(
        self,
        strategy: str = "random",
        classes: List[Any] = None,
        scoring_function: ScoringFunction = ScoringFunction.ACCURACY,
        random_seed: int = 0,
    ):
        """
        Parameters
        ----------
        strategy : str
            TODO
        classes : List[Any]
            TODO
        random_seed : int
            TODO
        """
        super().__init__(
            strategy=strategy,
            classes=classes,
            scoring_function=scoring_function,
            random_seed=random_seed,
        )

        self.strategy = strategy
        self._rng = RandomState(random_seed)

        if classes is not None:
            assert isinstance(classes, ARRAY_LIKE)
            self.classes_ = list(classes)

        # Initialize value counts for all starting classes (if any)
        self.value_counts_ = {label: 0 for label in self.classes_}

    def fit(self, X, y):
        super().check_X_y(X, y)
        for label in self.classes_:
            self.value_counts_[label] = self.value_counts_.get(label, 0) + 1
        return self

    def predict_proba(self, X):
        X = self.check_X(X)
        class_count = len(self.classes_)
        sample_count = len(X[0])

        if self.strategy == "mean":
            total = sum(self.value_counts_.values())
            return [[count / total for count in self.value_counts_.values()]] * sample_count

        if self.strategy == "random":
            return [self._rng.rand(class_count) for _ in range(sample_count)]

        raise RuntimeError("Unknown strategy: %s" % self.strategy)


class DummyRegressor(BaseRegressor):
    """
    Dummy implementation of a regressor used for testing purposes. It can output a random value
    or the expected value based on prior data depending on the strategy selected.
    """

    def __init__(
        self,
        strategy="random",
        scoring_function: ScoringFunction = ScoringFunction.R2,
        random_seed: int = 0,
    ):
        """
        Parameters
        ----------
        strategy : str
            TODO
        random_seed : int
            TODO
        """
        super().__init__(
            strategy=strategy, scoring_function=scoring_function, random_seed=random_seed
        )
        self.strategy = strategy
        self.rng = RandomState(random_seed)
        self.y_min_ = None
        self.y_max_ = None
        self.y_sum_ = 0.0
        self.y_mean_ = 0.0
        self.y_count_ = 0

    def fit(self, X, y):
        X, y = self.check_X_y(X, y)
        y = y.astype(DTYPE_FLOAT[0])

        # Update sum, mean and count
        self.y_sum_ += y.sum()
        self.y_count_ += len(y)
        self.y_mean_ = self.y_sum_ / self.y_count_

        # Update min and max seen
        y_min, y_max = y.min(), y.max()
        self.y_min_ = y_min if self.y_min_ is None or y_min < self.y_min_ else self.y_min_
        self.y_max_ = y_max if self.y_max_ is None or y_max > self.y_max_ else self.y_max_

        return self

    def predict(self, X):
        X = self.check_X(X)
        sample_count = len(X[0])

        if self.strategy == "random":
            output_range = self.y_max_ - self.y_min_
            return self.y_min_ + self.rng.randn(sample_count) * output_range

        if self.strategy == "mean":
            return [self.y_mean_] * sample_count

        raise RuntimeError("Unknown strategy: %s" % self.strategy)


class DummyTransformer(ColumnHandlingTransformer):
    """ Dummy implementation of a transformer that only checks for consistent shape of input """

    def fit(self, X):
        """ Fits nothing """
        X = self.check_X(X)
        return self

    def transform(self, X):
        """ Pass-through """
        X = self.check_X(X)
        # Reshape as vector if input was vector
        if self.input_is_vector_:
            X = X[0]
        return X

    def inverse_transform(self, X):
        """ Pass-through """
        self.check_attributes("input_is_vector_")
        X = self.check_X(X, ensure_shape=False, ensure_dtype=False)
        # Reshape as vector if input was vector
        if self.input_is_vector_:
            X = X[0]
        return X

    def on_input_shape_changed(self, change_map: ChangeMap):
        super().on_input_shape_changed(change_map)
        # Since we don't change shape ourselves, simply propagate the change downstream
        self.on_output_shape_changed(change_map)
