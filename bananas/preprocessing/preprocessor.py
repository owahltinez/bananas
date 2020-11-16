from typing import Any, Dict, Iterable, Tuple, Union
from ..core.learner import Learner
from ..core.pipeline import Pipeline, PipelineStep
from ..testing.dummy import DummyTransformer
from ..utils.arrays import ARRAY_LIKE


class _Preprocessor(Pipeline):
    """
    Internal class used as a base for all preprocessors targeting a mix of categorical and
    continuous features.
    """

    def __init__(
        self,
        categorical: Union[Dict, ARRAY_LIKE] = None,
        continuous: Union[Dict, ARRAY_LIKE] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        categorical : Union[Dict, ARRAY_LIKE]
            Columns that contain categorical data. If a dict is given, values must contain
            categories for that column.
        continuous : Union[Dict, ARRAY_LIKE]
            Columns that contain continuous data. If a dict is passed, its values are ignored.
        verbose : Boolean
            Prints debug info
        """
        self.verbose = verbose

        # Default to no categorical columns
        if categorical is None:
            self.categorical_ = {}
        if isinstance(categorical, dict):
            self.categorical_ = {**categorical}
        if isinstance(categorical, ARRAY_LIKE):
            self.categorical_ = {c: [] for c in categorical}

        # Default to no continuous columns
        if continuous is None:
            self.continuous_ = {}
        if isinstance(continuous, dict):
            self.continuous_ = {**continuous}
        if isinstance(continuous, ARRAY_LIKE):
            self.continuous_ = {c: None for c in continuous}

        # Initialize steps
        steps = self._init_steps(**kwargs)

        # If no steps were added, add dummy transformer
        if not steps:
            steps.append(PipelineStep(name="dummy", learner=DummyTransformer))
            self.print("Initialize dummy step")

        # Initialize parent's pipeline
        super().__init__(steps, verbose=verbose)

    def _init_steps(self, **kwargs) -> Iterable[PipelineStep]:
        raise NotImplementedError(
            "Subclasses of `%s` must override method `%s`" % self.__class__.__name__, "_init_steps"
        )
