''' Grid Search Module '''

from itertools import product
from typing import Any, Dict, Generator, Iterable, Type

from ..core.learner import Learner
from ..core.pipeline import Pipeline, PipelineStep
from ..dataset.dataset import DataSet
from ..utils.misc import valid_parameters

from .metalearner import _MetaLearner


def _iter_parameters(learner: Type[Learner], params: Dict[str, Any]):
    # When learner has PipelineStep's...
    has_steps = issubclass(learner, Pipeline) and \
        valid_parameters(learner.__init__, {'steps': None})
    if has_steps:
        steps_iters = [
            _iter_parameters(step.learner, step.kwargs) for step in params.get('steps', [])]
        # This is gross, but it appears to work
        yield from [
            {'steps': [
                PipelineStep(name=step.name, learner=step.learner, kwargs=step_kwargs)
                for step, step_kwargs in zip(params.get('steps', []), steps_set)]
            } for steps_set in product(*steps_iters)]

    # Otherwise, each parameter is expected to be a list of possible values
    else:
        value_iters = product(*params.values())
        for value_set in value_iters:
            yield {k: v for k, v in zip(params.keys(), value_set)}

class GridSearch(_MetaLearner):
    '''
    Meta-learner that performs a parameter grid search using the provided learner and a dictionary
    of parameters. The learner is re-initialized only once for each possible set of parameters.
    '''

    def __init__(self, learner: Type[Learner], learner_parameters: Dict[str, Any], n_jobs: int = 1,
                 verbose: bool = False):
        '''
        Parameters
        ----------
        learner : Type[Learner]
            TODO
        learner_parameters : Dict[str, Any]
            TODO
        n_jobs : int
            TODO
        verbose : bool
            TODO
        '''
        super().__init__(n_jobs=n_jobs, verbose=verbose)
        self.learner: Type[Learner] = learner
        self.learner_parameters = learner_parameters
        self.parameters_: Dict[Learner, Dict[str, Any]] = {}

    def _get_learner(self, idx, parameters: Dict[str, Any]) -> Learner:
        if idx not in self._learners_cache:
            has_verbose = issubclass(self.learner, Pipeline) and \
                valid_parameters(self.learner.__init__, {'verbose': None})
            if has_verbose: parameters['verbose'] = self.verbose
            learner_instance = self.learner(**parameters)
            self._learners_cache[idx] = learner_instance
            # Store the parameters in a dict that can be retrieved later
            self.parameters_[learner_instance] = parameters
        return self._learners_cache[idx]

    def _iter_learners(self) -> Generator[Learner, None, None]:

        # Count how many learners we yield
        learner_count = 0

        # For each learner, iterate over all possible parameters
        parameter_iterator = _iter_parameters(self.learner, self.learner_parameters)
        for idx, parameter_set in enumerate(parameter_iterator):
            learner_count += 1
            yield self._get_learner(idx, parameter_set)

        # If no learners were returned, crash here
        if learner_count == 0:
            raise RuntimeError('Failed to iterate over any learners')

    @staticmethod
    def hyperparameters(dataset: DataSet, steps: Iterable[PipelineStep] = None):
        return {}
