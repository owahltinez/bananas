"""
This module also provides a series of mixins that help us establish the purpose of a `Learner`. The
mixins not only add (or sometimes customize) a number of methods relevant to the type of learner,
but they also serve as a signal to other components in the ML framework to determine expected
behavior.
"""

from typing import Any, Iterable, List
import warnings

from ..changemap.changemap import ChangeMap
from ..dataset.datatype import DataType
from ..statistics.scoring import ScoringFunction, ScoringFunctionImpl
from ..transformers.encoders import LabelEncoder
from ..utils.arrays import argmax, check_array, shape_of_array
from .learner import SupervisedLearner


class BaseClassifier(SupervisedLearner):
    """
    Base class for a classifier. A classifier consists in a supervised learning model whose
    predicted outputs are among a set of labels. Users are expected to inherit from this class
    rather than instantiating it directly.

    Learners that inherit from the `BaseClassifier` mixin belong to the family of [classifiers](
    https://en.wikipedia.org/wiki/Statistical_classification). Classifiers are a subset of
    [supervisedlearners](learner.html#bananas.core.learner.SupervisedLearner) for which the feature
    labels are one of N classes. Learners in this category output the predicted class when `predict`
    is called, or a vector of probabilities for each of the classes when `predict_proba` is called
    -- which is in the same order as the classes in the `BaseClassifier.classes_` property.
    """

    def __init__(
        self,
        classes: List[Any] = None,
        scoring_function: ScoringFunction = ScoringFunction.F1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        classes : List[Any], optional
            List of classes that this learner can be "pre-loaded" with.
        scoring_function : ScoringFunction, optional
            Scoring function to use when `Learner.score()` is called.
        """

        super().__init__(classes=classes, scoring_function=scoring_function, **kwargs)

        # We need to initialize instance variables here for when we reset our learner
        self.classes_: List[Any] = []
        self.label_encoder_: LabelEncoder = None

        # Declare internal variables initialized during fitting to aid type checking
        if classes is not None:
            assert isinstance(classes, Iterable)
            self.classes_ = list(classes)

        # Instantiate the scoring function implementation
        self.scoring_function_: ScoringFunctionImpl = ScoringFunction.create(scoring_function)

    def check_X_y(self, X: Iterable[Iterable], y: Iterable):
        X, y = super().check_X_y(X, y)

        # Check label type
        data_type = DataType.parse(y)
        if not DataType.is_categorical(data_type):
            raise ValueError(f"Unknown data type for y: {data_type}. Expected categorical type.")

        # Initialize label encoder with classes seen so far
        if self.label_encoder_ is None:
            self.label_encoder_ = LabelEncoder(columns={0: self.classes_})

        # Check if `classes` has changed
        classes_set = set(self.classes_)
        classes_new = {label for label in y if label not in classes_set}
        if self.classes_ and classes_new:
            classes_curr = [*classes_new, *self.classes_]
            output_shape = (len(classes_curr),)
            idx_classes_new = [len(self.classes_) + i + 1 for i in range(len(classes_new))]
            self.on_output_shape_changed(ChangeMap(len(classes_curr), idx_add=idx_classes_new))
            warnings.warn(
                "Fitted data now has different classes [%r x %r] => [%r x %r], "
                "overwriting model..."
                % (self.input_shape_, self.output_shape_, self.input_shape_, output_shape)
            )

        # Record last seen classes and output dimension
        for label in classes_new:
            self.classes_.append(label)
        self.output_shape_ = (len(self.classes_),)

        # Encode target labels
        y = self.label_encoder_.fit(y).transform(y)

        # Finally, return the encoded labels alongside the input samples
        return X, y

    def predict_proba(self, X: Iterable[Iterable]) -> Iterable[Iterable]:
        """
        Probability estimates for all classes are ordered by the label of classes in `self.classes_`
        """
        raise NotImplementedError("Subclasses must override this method")

    def predict(self, X: Iterable[Iterable]) -> Iterable:
        X = self.check_X(X)
        probs = self.predict_proba(X)
        return self.label_encoder_.inverse_transform(argmax(probs))

    def score(self, X: Iterable[Iterable], y: Iterable) -> float:
        self.check_attributes("input_shape_")
        y = check_array(y, max_dimension=1)
        y = self.label_encoder_.transform(y)
        probs = self.predict_proba(X)
        return self.scoring_function_(y, probs)


class BaseRegressor(SupervisedLearner):
    """
    Base class for a regressor. A regressor consists in a supervised learning model whose predicted
    outputs are a single, continuous variable.

    The `BaseRegressor` mixin describes learners that address the [regression](
    https://en.wikipedia.org/wiki/Regression_analysis) problem. Regressors are a subset of [supervised
    learners](#supervised) for which the feature label of each sample is a single, continuous value.
    """

    def __init__(self, scoring_function: ScoringFunction = ScoringFunction.R2, **kwargs):
        """
        Parameters
        ----------
        scoring_function : ScoringFunction, optional
            Scoring function to use when `Learner.score()` is called.
        """
        super().__init__(scoring_function=scoring_function, **kwargs)

        # Instantiate the scoring function implementation
        self.scoring_function_: ScoringFunctionImpl = ScoringFunction.create(scoring_function)

    def score(self, X: Iterable[Iterable], y: Iterable) -> float:
        self.check_attributes("input_shape_")
        y = check_array(y, max_dimension=2)
        return self.scoring_function_(y, self.predict(X))

    def check_X_y(self, X: Iterable[Iterable], y: Iterable):
        X, y = super().check_X_y(X, y)

        # Record the shape of the target, which cannot change in regressor learner types
        target_shape = shape_of_array(y)[1:] or (1,)
        if not self.output_shape_:
            self.output_shape_ = target_shape
        elif self.output_shape_ != target_shape:
            raise RuntimeError(
                f"Target shape changed. Expected {self.output_shape_}, found {target_shape}."
            )

        return X, y


class HighDimensionalMixin(object):
    """
    Dummy mixin used to indicate that a learner supports high-dimensional inputs. For learners that
    do not apply this mixin, high-dimensional data such as images will be automatically flattened to
    a 2D array during sampling
    """

    pass
