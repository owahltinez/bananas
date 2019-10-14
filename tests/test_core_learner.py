''' Test Utils Module '''

from bananas.core.learner import Learner
from bananas.testing.learners import test_learner

from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):

    def test_learner_builtin(self):
        learner_kwargs = {'a': 1, 'b': 2, 'c': 3}
        self.assertTrue(test_learner(Learner, **learner_kwargs))

    '''
    def test_estimator_impl_regressor(self):
        estimator = DummyRegressor('mean')
        input_fn = lambda: (generate_array_floats(n=1024), generate_array_floats(n=1024))
        for _ in range(1000):
            estimator.fit(*input_fn())
        # Expected r^2 score for mean is zero
        self.assertAlmostEqual(estimator.score(*input_fn()), 0, places=2)

    def test_estimator_impl_classifier(self):
        num_classes = 8
        estimator = DummyClassifier('mean')
        input_fn = lambda: \
            (generate_array_floats(n=1024), generate_array_ints(n=1024, max_int=num_classes))
        for _ in range(1000):
            estimator.fit(*input_fn())
        # Expected accuracy for mean class is 1/n
        self.assertAlmostEqual(estimator.score(*input_fn()), 1 / num_classes, places=1)
    '''


main()
