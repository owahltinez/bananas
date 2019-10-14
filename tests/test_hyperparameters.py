''' Test Hyper Parameters Module '''

from bananas.core.learner import Learner
from bananas.core.pipeline import Pipeline, PipelineStep
from bananas.dataset.dataset import DataSet
from bananas.sampledata.synthetic import new_labels
from bananas.testing.dummy import DummyClassifier
from bananas.hyperparameters.bruteforce import BruteForce
from bananas.hyperparameters.gridsearch import _iter_parameters, GridSearch
from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class CustomEstimator(Learner):

    def __init__(self, a=0, b='b', c=None, verbose=False):
        self.a = a
        self.b = b
        # `c` omitted on purpose
        super().__init__()

    @staticmethod
    def hyperparameters(dataset: DataSet):
        return {'a': [1, 2, 3], 'b': ['x', 'y', 'z']}


class TestUtils(ProfilingTestCase):

    def test_iter_parameters_learner(self):
        params = CustomEstimator.hyperparameters(None)
        expected = [{'a': 1, 'b': 'x'},
                    {'a': 1, 'b': 'y'},
                    {'a': 1, 'b': 'z'},
                    {'a': 2, 'b': 'x'},
                    {'a': 2, 'b': 'y'},
                    {'a': 2, 'b': 'z'},
                    {'a': 3, 'b': 'x'},
                    {'a': 3, 'b': 'y'},
                    {'a': 3, 'b': 'z'}]
        result = list(_iter_parameters(CustomEstimator, params))
        self.assertListEqual(expected, result)

    def test_iter_parameters_pipeline(self):
        step = PipelineStep(
            name='1', learner=CustomEstimator, kwargs=CustomEstimator.hyperparameters(None))
        params = Pipeline.hyperparameters(None, [step])
        expected = [
            {'steps': [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 1, 'b': 'x'})]},
            {'steps': [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 1, 'b': 'y'})]},
            {'steps': [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 1, 'b': 'z'})]},
            {'steps': [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 2, 'b': 'x'})]},
            {'steps': [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 2, 'b': 'y'})]},
            {'steps': [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 2, 'b': 'z'})]},
            {'steps': [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 3, 'b': 'x'})]},
            {'steps': [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 3, 'b': 'y'})]},
            {'steps': [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 3, 'b': 'z'})]}]
        result = list(_iter_parameters(Pipeline, params))
        self.assertListEqual(expected, result)

    def test_iter_parameters_nested_pipeline(self):
        step = PipelineStep(
            name='1', learner=CustomEstimator, kwargs=CustomEstimator.hyperparameters(None))
        nested = PipelineStep(name='nested', learner=Pipeline, kwargs={'steps': [step]})
        params = Pipeline.hyperparameters(None, [nested])
        expected = [
            {'steps': [PipelineStep(name='nested', learner=Pipeline, kwargs={'steps':
                [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 1, 'b': 'x'})]})]},
            {'steps': [PipelineStep(name='nested', learner=Pipeline, kwargs={'steps':
                [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 1, 'b': 'y'})]})]},
            {'steps': [PipelineStep(name='nested', learner=Pipeline, kwargs={'steps':
                [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 1, 'b': 'z'})]})]},
            {'steps': [PipelineStep(name='nested', learner=Pipeline, kwargs={'steps':
                [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 2, 'b': 'x'})]})]},
            {'steps': [PipelineStep(name='nested', learner=Pipeline, kwargs={'steps':
                [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 2, 'b': 'y'})]})]},
            {'steps': [PipelineStep(name='nested', learner=Pipeline, kwargs={'steps':
                [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 2, 'b': 'z'})]})]},
            {'steps': [PipelineStep(name='nested', learner=Pipeline, kwargs={'steps':
                [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 3, 'b': 'x'})]})]},
            {'steps': [PipelineStep(name='nested', learner=Pipeline, kwargs={'steps':
                [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 3, 'b': 'y'})]})]},
            {'steps': [PipelineStep(name='nested', learner=Pipeline, kwargs={'steps':
                [PipelineStep(name='1', learner=CustomEstimator, kwargs={'a': 3, 'b': 'z'})]})]}]
        result = list(_iter_parameters(Pipeline, params))
        self.assertListEqual(expected, result)

    def test_grid_search_learner(self):
        params = CustomEstimator.hyperparameters(None)
        expected = [{'a': 1, 'b': 'x'},
                    {'a': 1, 'b': 'y'},
                    {'a': 1, 'b': 'z'},
                    {'a': 2, 'b': 'x'},
                    {'a': 2, 'b': 'y'},
                    {'a': 2, 'b': 'z'},
                    {'a': 3, 'b': 'x'},
                    {'a': 3, 'b': 'y'},
                    {'a': 3, 'b': 'z'}]
        search = GridSearch(CustomEstimator, param_grid=params)
        result = [{'a': getattr(learner, 'a'), 'b': getattr(learner, 'b')}
                  for learner in search._iter_learners()]
        self.assertListEqual(expected, result)

    def test_brute_force_learner(self):
        learners = [DummyClassifier]
        dataset = new_labels(random_seed=0)
        brute = BruteForce(dataset, learners)
        brute.train(dataset.input_fn, max_steps=100)

main()
