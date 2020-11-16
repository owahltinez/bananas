from typing import Dict, Iterable, List, Tuple, Type

from ..core.learner import Learner
from ..core.pipeline import Pipeline, PipelineStep
from ..dataset.dataset import DataSet
from ..preprocessing.standard import StandardPreprocessor

from .gridsearch import GridSearch
from .metalearner import _MetaLearner


class BruteForce(_MetaLearner):
    """
    Meta-learner that iterates over all the provided learners. The learners are used as-is -- i.e.
    unlike `GridSearch` learners are not cloned.
    """

    def __init__(
        self,
        dataset: DataSet,
        learners: Iterable[Type[Learner]],
        n_jobs: int = 1,
        verbose: bool = False,
        **learner_kwargs
    ):
        """
        Parameters
        ----------
        dataset : DataSet
            TODO
        learners : Iterable[Type[Learner]]
            TODO
        n_jobs : int
            TODO
        verbose : bool
            TODO
        learner_kwargs
            TODO
        """
        super().__init__(n_jobs=n_jobs, verbose=verbose)

        self.estimators: List[Learner] = []
        for learner in learners:
            pipeline_steps = [
                PipelineStep(
                    name="preprocessor",
                    learner=StandardPreprocessor,
                    kwargs={"categorical": dataset.categorical, "continuous": dataset.continuous},
                ),
                PipelineStep(
                    name="estimator",
                    learner=GridSearch,
                    kwargs={
                        "learner": learner,
                        "verbose": verbose,
                        "learner_parameters": {
                            **learner.hyperparameters(dataset),
                            **learner_kwargs,
                        },
                    },
                ),
            ]
            self.estimators.append(Pipeline(pipeline_steps, verbose=verbose))

    def _iter_learners(self):
        yield from self.estimators
