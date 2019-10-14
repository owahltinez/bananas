from typing import Any, Dict, Iterable, Tuple

from ..changemap.mixins import InputChangeMixin
from ..changemap.changemap import ChangeMap
from ..hyperparameters.mixins import HyperParametersMixin
from ..training.mixins import TrainableMixin
from ..utils.arrays import check_array, shape_of_features
from ..utils.constants import ARRAY_LIKE

_ATTR_NAMED_ARGUMENTS_PREFIX = '_named_argument'
''' Prefix used to store named arguments in an internal dictionary '''


class Learner(InputChangeMixin, TrainableMixin, HyperParametersMixin):
    '''
    Base class for all learners. It provides many convenience functions that can be used by
    children classes such as `check_X` and `check_X_y`, and it provides a fundamental contract
    for how users are expected to train their machine learning models. It is inspired by SciKit and
    provides the methods `fit`, `predict` and `score`.

    The learner class is expected to be the base class for all objects that implement a `fit` /
    `predict` interface, so that they can be compatible with one another. Its design has been heavily
    influenced by Scikit-Learn estimators, but has several key differences.

    ## Input data format
    Data passed to a learner is expected to be in **column-first format**. That is, learners receive a
    list of input features (i.e. *columns*). This is in contrast to the more usual *sample-first* format
    where learners accept a list of samples. The rationale for this is explained in the [design
    principes](../#input-data-format) section of the intro docs.
    '''

    def __init__(self, **kwargs):

        # Declare variables initialized during fitting to aid type-checking
        self.input_dtype_: tuple = None
        self.input_shape_: tuple = None
        self.output_shape_: tuple = (1,)
        self.input_is_vector_: bool = None

    def print(self, *msg, **kwargs):
        ''' Print message to console only if in verbose mode '''
        if getattr(self, 'verbose', False):
            print('[%s]' % (self.__class__.__name__), *msg, **kwargs)

    def check_attributes(self, *attributes: str):
        '''
        Makes sure that the provided attributes have been initialized. This method is expected
        to be used only in custom learner implementations.

        Parameters
        ----------
        attributes : str
            Attributes to verify for initialization
        '''
        assert all([getattr(self, attr, None) is not None for attr in attributes]), ( \
            'Required attributes not initialized: %r' % \
            [attr for attr in attributes if getattr(self, attr, None) is None])

    def check_X(self, X: Iterable[Iterable], ensure_2d: bool = True, ensure_shape: bool = True,
                ensure_dtype: bool = True) -> Iterable:
        '''
        Ensures that the input provided has a consistent shape and type with respect to previous
        calls to this method. It will also convert the input to the expected shape and type if
        necessary (and possible).

        Parameters
        ----------
        X : Iterable[Iterable]
            Input to be converted, expects a list of training samples in the form of a list of
            columns
        ensure_2d : bool
            Reshapes the output into a 2D matrix if necessary (for example, if a vector was given)
        ensure_shape : bool
            Checks for consistent input shape across calls to this method
        ensure_dtype : bool
            Checks for consistent input type across calls to this method

        Returns
        -------
        Iterable
            Converted input sample values after checking shape and types
        '''

        # Convert vectors to list of vectors
        input_vector = not isinstance(X[0], ARRAY_LIKE)
        if input_vector: X = [X]

        # If we are enforcing dtype, then we should be checking array against previously seen
        # dtype for each column so we can cast columns into appropriate dtype. This ensures that the
        # dtype of each column will be consistent instead of trying to optimize it later on; for
        # example converting int-like arrays like [0. 1. 1. 0. ...] to int when dtype was float.
        check_array_opts = [{} for i in range(len(X))]
        if ensure_dtype and hasattr(self, 'input_dtype_') and self.input_dtype_:
            check_array_opts = [{'dtype': self.input_dtype_.get(i)} for i in range(len(X))]

        # Convert data into a list of arrays and make sure they are all of the same length
        X = [check_array(col, **check_array_opts[i]) for i, col in enumerate(X)]
        if not all([col.shape[0] == X[0].shape[0] for col in X[1:]]):
            raise ValueError('Input features have inconsistent number of samples. Expected all to '
                             'be %d, found %r' % (X[0].shape[0], [col.shape[0] for col in X]))

        # Extract the shape of each of the features
        input_shape = shape_of_features(X)

        # If input shape changed, trigger appropriate event callback
        if ensure_shape:
            # if self.input_shape_ and input_shape != input_shape:
            #     self.on_input_shape_changed(self.input_shape_, input_shape)
            self.input_shape_ = input_shape
            self.input_is_vector_ = input_vector

        # FIXME: If this learner does not accept high-dimensional data, reshape input features
        # if len(input_shape) > 2 and not isinstance(self, ImageLearnerMixin):
        #     num_samples = input_shape[1]
        #     X = [flatten(col) for col in X]
        #    X = [col.reshape(([dim for dim in col.shape if dim != 1] or [1])[0], -1)
        #         if col.ndim > 2 and ensure_2d else col for col in X]
        #    if not input_vector and len(X) == 1 and X[0].ndim > 2: X = X[0]

        # Extract the dtype of each of the features
        # TODO: replace col.dtype with type_of_input(col)
        input_dtype = {i: col.dtype for i, col in enumerate(X)}

        # If input dtype changed, raise error
        if ensure_dtype:
            # Check again to see if conversion was successful
            get_dtype = lambda i, fallback: self.input_dtype_.get(i, fallback) or fallback
            if hasattr(self, 'input_dtype_') and self.input_dtype_ and \
                any([dtype != get_dtype(i, dtype) for i, dtype in input_dtype.items()]):
                raise TypeError('Input dtypes changed. Expected %r, found %r' %
                                (self.input_dtype_, input_dtype))
            self.input_dtype_ = input_dtype

        # Reshape input back to vector if necessary
        if not ensure_2d and input_vector: X = X[0]

        return X

    def check_X_y(self, X: Iterable[Iterable], y: Iterable) -> Tuple[Iterable, Iterable]:
        '''
        Performs all the checks described in `check_X` for `X` and ensures that the argument `y` is
        an appropriately shaped and sized array.

        Parameters
        ----------
        X : Iterable[Iterable]
            Input samples to be converted, expects a list of training samples in the form of a list
            of columns
        y : Iterable
            Input targets to be converted, expects a list of sample outputs to be predicted in the
            form of a 1D list of values

        Returns
        -------
        X : Iterable[Iterable]
            Converted input sample values after checking shape and types
        y : Iterable
            Converted target sample values after checking shape and types
        '''
        X, y = self.check_X(X), check_array(y)
        return X, y

    def apply_mixin(self, *mixins: type):
        '''
        Applies the given mixins to this instance. This can be used, for example, to establish
        a learner as a classifier without having to create a dummy subclass. So the following are
        equivalent:
        ```
        class MyLearnerBaseClass(Learner):
            ...

        # Option 1: dummy subclass
        class MyLearnerClassifier(MyLearnerBaseClass, BaseClassifier):
            pass

        # Option 2: apply mixins
        learner_classifier = MyLearnerBaseClass().apply_mixin(BaseClassifier)
        ```
        '''
        base_class = self.__class__
        base_class_name = self.__class__.__name__
        self.__class__ = type(base_class_name, (base_class, *mixins), {})
        for mixin in mixins: mixin.__init__(self)
        return self

    def predict(self, X: Iterable[Iterable]):  # pylint: disable=unused-argument
        '''
        Outputs the learner's predicted target values for the provided input samples. **Input
        samples are  expected to be a list of columns**, so that each column represents a distinct
        feature. The output target values are in the form of a 1D list of values.

        Parameters
        ----------
        X : Iterable[Iterable]
            Input samples to use for prediction, expects a list of training samples in the form of a
            list of columns

        Returns
        -------
        Iterable
            Predicted output values in the form of a 1D list of values
        '''
        raise NotImplementedError()

    def score(self, X: Iterable[Iterable], y: Iterable) -> float:
        '''
        Scores input against validation data. **Input samples are  expected to be a list of
        columns**, so that each column represents a distinct feature. The target values are expected
        to be in the form of a 1D list of values.

        Parameters
        ----------
        X : Iterable[Iterable]
            Input samples to be scored, expects a list of training samples in the form of a list
            of columns
        y : Iterable
            True input targets to score predictions against, expects a list of values in the form of
            a 1D list

        Returns
        -------
        float
            Score of the predicted output with respect to the true output
        '''
        raise NotImplementedError()

    def on_input_shape_changed(self, change_map: ChangeMap):
        self._input_change_column_adapter(change_map, ['input_dtype_', 'input_shape_'])


class SupervisedLearner(Learner):
    '''
    Base class for a model implementing supervised learning. Fitting data to the model requires
    both input samples and sample targets -- also known as "true" labels in classification problems.

    Supervised learners are those that expect the input data to come in sample-label pairs. As such,
    the `fit` function in a supervised learner takes two parameters:
    1. Feature set: column-first list of input features.
    2. Feature labels: 1D list of input labels.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X: Iterable[Iterable], y: Iterable) -> 'SupervisedLearner':
        '''
        Fit input to model incrementally. Users are expected to call this method with mini-batches
        of data that will be fed to the underlying learner. **Input samples are  expected to be a
        list of columns**, so that each column represents a distinct feature. The target values are
        expected to be in the form of a 1D list of values.

        Parameters
        ----------
        X : Iterable[Iterable]
            Input samples to be learned, expects a list of training samples in the form of a list
            of columns
        y : Iterable
            Input targets to be learned, expects a list of values to be predicted in the form of a
            1D list

        Returns
        -------
        Learner
            Instance of self
        '''
        raise NotImplementedError()


class UnsupervisedLearner(Learner):
    '''
    Base class for a model implementing unsupervised learning. Unsupervised learners only take a
    single parameter in their `fit` function: a column-first list of input features. The canonical
    example of an unsupervised learning technique is clustering.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X: Iterable[Iterable]) -> 'UnsupervisedLearner':
        '''
        Fit input to model incrementally. Users are expected to call this method with mini-batches
        of data that will be fed to the underlying learner. **Input samples are  expected to be a
        list of columns**, so that each column represents a distinct feature. The target values are
        expected to be in the form of a 1D list of values.

        Parameters
        ----------
        X : Iterable[Iterable]
            Input samples to be learned, expects a list of training samples in the form of a list
            of columns

        Returns
        -------
        Learner
            Instance of self
        '''
        raise NotImplementedError()
