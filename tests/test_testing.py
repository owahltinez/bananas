''' Test Utils Module '''

from bananas.testing.dummy import DummyClassifier, DummyRegressor, DummyTransformer
from bananas.testing.learners import test_learner
from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):

    def test_dummy_transformer_builtin(self):
        self.assertTrue(test_learner(DummyTransformer))

    def test_dummy_regressor_builtin(self):
        learner_kwargs = ({'strategy': 'mean'}, {'strategy': 'random'})
        for kwargs in learner_kwargs:
            self.assertTrue(test_learner(DummyRegressor, **kwargs))

    def test_dummy_classifier_builtin(self):
        learner_kwargs = ({'strategy': 'mean'}, {'strategy': 'random'})
        for kwargs in learner_kwargs:
            self.assertTrue(test_learner(DummyClassifier, **kwargs))


main()
