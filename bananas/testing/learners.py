"""
This library comes with a comprehensive testing suite that checks for API compliance, input type
handling, change map handling, and more. Not all tests are run for every learner; tests specific
to certain estimators like [supervised learners](../core/index.md#supervised) or [transformers](
../transformers/index.md) are only run when the learner instance is of the corresponding type.
"""

import warnings
from inspect import signature, Parameter
from unittest import TestCase, TestResult
from ..changemap.changemap import ChangeMap
from ..core.learner import Learner, SupervisedLearner, UnsupervisedLearner
from ..core.mixins import BaseClassifier, BaseRegressor
from ..dataset.dataset import DataSet
from ..training.train_history import TrainHistory
from ..transformers.base import BaseTransformer
from ..utils.arrays import check_array, unique
from ..utils.misc import warn_with_traceback
from .generators import (
    generate_array_booleans,
    generate_array_chars,
    generate_array_floats,
    generate_array_ints,
    generate_array_int_floats,
    generate_array_uints,
    generate_array_nones,
    generate_array_strings,
    generate_images,
    generate_onehot_matrix,
    generate_array_infinities,
)

# Number of samples in the test data
TEST_SAMPLE_SIZE = 1024


def test_learner(learner_type: (type, Learner), **learner_kwargs):
    """
    Performs a battery of tests against the provided learner. If the learner must be initialized
    with certain parameters, those can be passed to this function too.

    Parameters
    ----------
    learner_type : type, Learner
        TODO
    learner_kwargs
        TODO
    """

    # Change warnings behavior to display stack trace
    showwarning = warnings.showwarning
    warnings.showwarning = warn_with_traceback

    # If we were given an instance instead of a type, convert back to type and guess args
    if isinstance(learner_type, Learner):
        learner_type = type(learner_type)

    # Set random state via keyword argument for all learners if they support it
    params = signature(learner_type.__init__).parameters
    if "random_seed" in params.keys() or any(
        [param.kind == Parameter.VAR_KEYWORD for param in params.values()]
    ):
        learner_kwargs["random_seed"] = 0

    # Test options apply to all tests
    test_opts = {"learner_class": learner_type, "learner_kwargs": learner_kwargs}

    # Pick test suites based on subclasses
    test_suites = []
    if issubclass(learner_type, Learner):
        test_suites.append(LearnerTests(**test_opts))
    if issubclass(learner_type, SupervisedLearner):
        test_suites.append(SupervisedLearnerTests(**test_opts))
    if issubclass(learner_type, UnsupervisedLearner):
        test_suites.append(UnsupervisedLearnerTests(**test_opts))
    if issubclass(learner_type, BaseTransformer):
        test_suites.append(TransformerTests(**test_opts))
    if issubclass(learner_type, BaseRegressor):
        test_suites.append(RegressorTests(**test_opts))
    if issubclass(learner_type, BaseClassifier):
        test_suites.append(ClassifierTests(**test_opts))

    # Accumulate results over all test suites
    result = TestResult()
    for suite in test_suites:
        suite.run(result=result)

    # Display errors
    for err in result.errors:
        print("\n%s\n" % err[1])
    for fail in result.failures:
        print("\n%s\n" % fail[1])

    # Restore warning behavior
    warnings.showwarning = showwarning

    # If any errors or failures, raise the exception
    assert not result.errors and not result.failures, (
        "One or more tests failed. Please see "
        "output for all Tracebacks.\n\n%s\n"
        % (result.errors[0][1] if result.errors else result.failures[0][1])
    )

    return True


def assert_predictions_match_cloned_learner(test: TestCase, y1, y2):
    y1, y2 = list(y1[:100]), list(y2[:100])  # limit amount of comparisons for perf and put in list
    test.assertListEqual(
        y1,
        y2,
        (
            "Calling `predict` after fitting the same set of data to a cloned learner should "
            "yield the same result"
        ),
    )


class _LearnerTests(TestCase):
    def __init__(self, learner_class: type, learner_kwargs: dict):
        super().__init__()
        self.learner_class = learner_class
        self.learner_kwargs = learner_kwargs

    def runTest(self):
        for test_name in [key for key in dir(self) if key.startswith("test_")]:
            test_case = getattr(self, test_name)
            try:
                test_case()
            except Exception as err:
                print("%s_%s ... fail" % (self.learner_class.__name__, test_name))
                raise err
            print("%s_%s ... ok" % (self.learner_class.__name__, test_name))

    def _get_learner(self, **learner_kwargs):
        learner: Learner = self.learner_class(**{**self.learner_kwargs, **learner_kwargs})
        return learner


class LearnerTests(_LearnerTests):
    """ Basic tests for the base Learner class """

    def test_learner_calls_init(self):

        # Test whether parent's init is called by monkey patching it
        flag = [False]
        __init_old__ = Learner.__init__

        def __init_new__(self, **kwargs):
            flag[0] = True

        Learner.__init__ = __init_new__

        # Initialize learner, which will trigger new __init__
        learner = self._get_learner()

        # Put back the original init
        Learner.__init__ = __init_old__

        # Perform assertion
        self.assertEqual(True, flag[0], "Learner must call super().__init__() method")


class UnsupervisedLearnerTests(_LearnerTests):
    """ Tests designed for unsupervised learners. """

    def test_unsupervised_signature(self):
        learner: UnsupervisedLearner = self._get_learner()
        X = generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0)
        y = generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0)

        # Fit returns self
        self.assertEqual(learner, learner.fit(X))

        # Fit and predict only take X, or return NotImplementedError
        for func in ("fit", "predict"):
            if not hasattr(learner, func):
                continue
            try:
                self.assertTrue(getattr(learner, func)(X) is not None)
            except NotImplementedError:
                pass
            self.assertRaises(TypeError, lambda: getattr(learner, func)(X, y))

        # score takes X, y, or returns NotImplementedError
        for func in ("score",):
            if not hasattr(learner, func):
                continue
            try:
                self.assertTrue(getattr(learner, func)(X, y) is not None)
            except NotImplementedError:
                pass
            self.assertRaises(TypeError, lambda: getattr(learner, func)(X))

    def test_unsupervised_input_changed(self):
        learner: UnsupervisedLearner = self._get_learner()
        X1 = generate_array_floats(n=TEST_SAMPLE_SIZE * 16, random_seed=0).reshape(16, -1)
        X2 = generate_array_floats(n=TEST_SAMPLE_SIZE * 8, random_seed=0).reshape(8, -1)

        # Fit first batch
        learner.fit(X1)

        # Send input changed event
        n = 16
        change_map = ChangeMap(n, idx_del=[i * 2 for i in range(n // 2)])
        learner.on_input_shape_changed(change_map)

        # Fit second batch
        learner.fit(X2)

    def test_unsupervised_batch_size_changed(self):
        learner: UnsupervisedLearner = self._get_learner()
        features = [
            (
                generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
                generate_array_floats(n=TEST_SAMPLE_SIZE // 2, random_seed=0),
            ),
            (
                generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0).reshape(2, -1),
                generate_array_floats(n=TEST_SAMPLE_SIZE // 2, random_seed=0).reshape(2, -1),
            ),
        ]
        # (generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0).reshape(2, -1, 2),
        #    generate_array_floats(n=TEST_SAMPLE_SIZE // 2, random_seed=0)).reshape(2, -1, 2)]

        for X1, X2 in features:
            learner = self._get_learner()
            # Fit first batch
            learner.fit(X1)
            # Fit second batch
            learner.fit(X2)


class SupervisedLearnerTests(_LearnerTests):
    """ Tests designed for supervised learners. """

    def test_supervised_signature(self):
        learner: SupervisedLearner = self._get_learner()
        X = generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0)
        y = generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0)

        # Fit returns self
        self.assertEqual(learner, learner.fit(X, y))

        # Fit and score take X, y
        for func in ("fit", "score"):
            self.assertTrue(getattr(learner, func)(X, y) is not None)
            self.assertRaises(TypeError, lambda: getattr(learner, func)(X))

        # Predict only takes X
        for func in ("predict",):
            self.assertTrue(getattr(learner, func)(X) is not None)
            self.assertRaises(TypeError, lambda: getattr(learner, func)(X, y))

    def test_supervised_input_changed(self):
        learner: SupervisedLearner = self._get_learner()
        X1 = generate_array_floats(n=TEST_SAMPLE_SIZE * 16, random_seed=0).reshape(16, -1)
        X2 = generate_array_floats(n=TEST_SAMPLE_SIZE * 8, random_seed=0).reshape(8, -1)
        y = generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0)

        # Fit first batch
        learner.fit(X1, y)

        # Send input changed event
        n = 16
        change_map = ChangeMap(n, idx_del=[i * 2 for i in range(n // 2)])
        learner.on_input_shape_changed(change_map)

        # Fit second batch
        learner.fit(X2, y)

    def test_supervised_batch_size_changed(self):
        features = [
            (
                generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
                generate_array_floats(n=TEST_SAMPLE_SIZE // 2, random_seed=0),
            ),
            (
                generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0).reshape(2, -1),
                generate_array_floats(n=TEST_SAMPLE_SIZE // 2, random_seed=0).reshape(2, -1),
            ),
        ]
        # (generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0).reshape(2, -1, 2),
        #    generate_array_floats(n=TEST_SAMPLE_SIZE // 2, random_seed=0)).reshape(2, -1, 2)]
        targets = [
            (
                generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
                generate_array_int_floats(n=TEST_SAMPLE_SIZE // 2, random_seed=0),
            ),
            (
                generate_array_int_floats(n=TEST_SAMPLE_SIZE // 2, random_seed=0),
                generate_array_int_floats(n=TEST_SAMPLE_SIZE // 4, random_seed=0),
            ),
        ]

        for (X1, X2), (y1, y2) in zip(features, targets):
            learner: SupervisedLearner = self._get_learner()
            # Fit first batch
            learner.fit(X1, y1)
            # Fit second batch
            learner.fit(X2, y2)


class TransformerTests(_LearnerTests):
    """ Tests designed for transformers. """

    def test_transformer_transform(self):
        learner: BaseTransformer = self._get_learner()
        X = generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0)
        Xt1 = learner.fit(X).transform(X)

        learner = self._get_learner()
        Xt2 = learner.fit(X).transform(X)

        Xt1 = check_array(Xt1).tolist()
        Xt2 = check_array(Xt2).tolist()

        self.assertListEqual(Xt1, Xt2)

    def test_transformer_inverse_transform(self):

        # Inverse transform should work at least with one of continuous or multiclass
        continous_data = generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0)
        categorical_data = generate_array_chars(n=TEST_SAMPLE_SIZE, random_seed=0)

        success = False
        for X in (continous_data, categorical_data):
            learner = self._get_learner()
            try:
                Xt = learner.fit(X).transform(X)
                success = True
            except TypeError:
                # Some transformers only accept continuous or multiclass
                continue
            try:
                X_ = learner.inverse_transform(Xt)
                # Inverse transformation may not be exact, so use fuzzy comparison
                for v1, v2 in zip(X[:100], X_[:100]):
                    self.assertAlmostEqual(v1, v2)

            except NotImplementedError:
                # Having a transformer that does not implement inverse_transform is OK
                pass
        self.assertTrue(success, "fit-transform did not work for continuous or categorical data")

    def test_transformer_input_changed(self):
        learner: BaseTransformer = self._get_learner()
        X1 = generate_array_floats(n=TEST_SAMPLE_SIZE * 16, random_seed=0).reshape(16, -1)
        X2 = generate_array_floats(n=TEST_SAMPLE_SIZE * 8, random_seed=0).reshape(8, -1)

        # Fit first batch
        learner.fit(X1).transform(X1)

        # Send input changed event
        n = 16
        change_map = ChangeMap(n, idx_del=[i * 2 for i in range(n // 2)])
        learner.on_input_shape_changed(change_map)

        # Fit second batch
        learner.fit(X2).transform(X2)


class RegressorTests(_LearnerTests):
    """ Tests designed for regressors. """

    def test_regressor_fit_1D(self):
        features = [
            generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        targets = [
            generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        for X in features:
            for y in targets:
                learner = self._get_learner()
                learner.fit(X, y)

    def test_regressor_fit_1D_single_sample(self):
        features = [
            generate_array_floats(n=1, random_seed=0),
            generate_array_int_floats(n=1, random_seed=0),
            generate_array_ints(n=1, random_seed=0),
            generate_array_uints(n=1, random_seed=0),
            generate_array_booleans(n=1, random_seed=0),
        ]

        targets = [
            generate_array_floats(n=1, random_seed=0),
            generate_array_int_floats(n=1, random_seed=0),
            generate_array_ints(n=1, random_seed=0),
            generate_array_uints(n=1, random_seed=0),
            generate_array_booleans(n=1, random_seed=0),
        ]

        for X in features:
            for y in targets:
                learner = self._get_learner()
                learner.fit(X, y)

    def test_regressor_fit_2D(self):
        learner: BaseRegressor = self._get_learner()
        features = [
            generate_array_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_ints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_uints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_booleans(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
        ]

        targets = [
            generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        for X in features:
            for y in targets:
                learner = self._get_learner()
                learner.fit(X, y)

    def test_regressor_fit_3D(self):
        features = [
            generate_array_floats(n=TEST_SAMPLE_SIZE * 3, random_seed=0).reshape(3, -1),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE * 3, random_seed=0).reshape(3, -1),
            generate_array_ints(n=TEST_SAMPLE_SIZE * 3, random_seed=0).reshape(3, -1),
            generate_array_uints(n=TEST_SAMPLE_SIZE * 3, random_seed=0).reshape(3, -1),
            generate_array_booleans(n=TEST_SAMPLE_SIZE * 3, random_seed=0).reshape(3, -1),
        ]

        targets = [
            generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        for X in features:
            for y in targets:
                learner = self._get_learner()
                learner.fit(X, y)

    def test_regressor_predict(self):
        features = [
            generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_ints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_uints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_booleans(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
        ]

        targets = [
            generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        for X in features:
            for y in targets:
                learner_ = self._get_learner()
                y1 = learner_.fit(X, y).predict(X)

                learner_ = self._get_learner()
                y2 = learner_.fit(X, y).predict(X)
                assert_predictions_match_cloned_learner(self, y1, y2)

                learner_ = self._get_learner()
                y3 = learner_.fit(X, y).predict(X)
                assert_predictions_match_cloned_learner(self, y1, y3)

    def test_regressor_train(self):
        features = [
            generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_ints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_uints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_booleans(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
        ]

        targets = [
            generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        for X in features:
            for y in targets:
                # Make sure that `train` returns history object
                learner1 = self._get_learner()
                input_fn = DataSet.from_ndarray(X, y, random_seed=0).input_fn
                history = learner1.train(input_fn, max_steps=10)
                self.assertEqual(type(history), TrainHistory)

                # Make sure that learners predict same data
                learner2 = self._get_learner()
                input_fn = DataSet.from_ndarray(X, y, random_seed=0).input_fn
                learner2_ = learner2.train(input_fn, max_steps=10)
                y1 = learner1.predict(X)[:100]
                y2 = learner2.predict(X)[:100]
                assert_predictions_match_cloned_learner(self, y1, y2)


class ClassifierTests(_LearnerTests):
    """ Tests designed for classifiers. """

    def test_classifier_fit_1D(self):
        features = [
            generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        targets = [
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_chars(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_strings(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        for X in features:
            for y in targets:
                learner = self._get_learner()
                learner.fit(X, y)

    def test_classifier_fit_1D_single_sample(self):
        features = [
            generate_array_floats(n=1, random_seed=0),
            generate_array_int_floats(n=1, random_seed=0),
            generate_array_ints(n=1, random_seed=0),
            generate_array_uints(n=1, random_seed=0),
            generate_array_booleans(n=1, random_seed=0),
        ]

        targets = [
            generate_array_int_floats(n=1, random_seed=0),
            generate_array_ints(n=1, random_seed=0),
            generate_array_uints(n=1, random_seed=0),
            generate_array_booleans(n=1, random_seed=0),
            generate_array_chars(n=1, random_seed=0),
            generate_array_strings(n=1, random_seed=0),
        ]

        for X in features:
            for y in targets:
                learner = self._get_learner()
                learner.fit(X, y)

    def test_classifier_fit_2D(self):
        features = [
            generate_array_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_ints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_uints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_booleans(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
        ]

        targets = [
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_chars(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_strings(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        for X in features:
            for y in targets:
                learner = self._get_learner()
                learner.fit(X, y)

    def test_classifier_fit_3D(self):
        learner: BaseClassifier = self._get_learner()
        features = [
            generate_array_floats(n=TEST_SAMPLE_SIZE * 3, random_seed=0).reshape(3, -1),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE * 3, random_seed=0).reshape(3, -1),
            generate_array_ints(n=TEST_SAMPLE_SIZE * 3, random_seed=0).reshape(3, -1),
            generate_array_uints(n=TEST_SAMPLE_SIZE * 3, random_seed=0).reshape(3, -1),
            generate_array_booleans(n=TEST_SAMPLE_SIZE * 3, random_seed=0).reshape(3, -1),
        ]

        targets = [
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_chars(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_strings(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        for X in features:
            for y in targets:
                learner = self._get_learner()
                learner.fit(X, y)

    def test_classifier_predict(self):
        features = [
            generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_ints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_uints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_booleans(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
        ]

        targets = [
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_chars(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_strings(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        for X in features:
            for y in targets:
                learner_ = self._get_learner()
                y1 = learner_.fit(X, y).predict(X)

                learner_ = self._get_learner()
                y2 = learner_.fit(X, y).predict(X)
                assert_predictions_match_cloned_learner(self, y1, y2)

                learner_ = self._get_learner()
                y3 = learner_.fit(X, y).predict(X)
                assert_predictions_match_cloned_learner(self, y1, y3)

    def test_classifier_predict_proba(self):
        pass  # TODO: implement this test

    def test_classifier_train(self):
        features = [
            generate_array_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_int_floats(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_ints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_uints(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
            generate_array_booleans(n=TEST_SAMPLE_SIZE * 2, random_seed=0).reshape(2, -1),
        ]

        targets = [
            generate_array_int_floats(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_ints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_uints(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_booleans(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_chars(n=TEST_SAMPLE_SIZE, random_seed=0),
            generate_array_strings(n=TEST_SAMPLE_SIZE, random_seed=0),
        ]

        for X in features:
            for y in targets:

                # Make sure that `train` returns history object
                learner1 = self._get_learner()
                input_fn = DataSet.from_ndarray(X, y, random_seed=0).input_fn
                history = learner1.train(input_fn, max_steps=10)
                self.assertEqual(type(history), TrainHistory)

                # Make sure that learners predict same data
                learner2 = self._get_learner()
                input_fn = DataSet.from_ndarray(X, y, random_seed=0).input_fn
                learner2.train(input_fn, max_steps=10)
                y1 = learner1.predict(X)
                y2 = learner2.predict(X)
                assert_predictions_match_cloned_learner(self, y1, y2)
