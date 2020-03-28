from collections import OrderedDict
from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Dict, Iterable, List, Type

from ..changemap.changemap import ChangeMap
from ..core.learner import Learner
from ..dataset.dataset import DataSet
from ..transformers.base import BaseTransformer
from ..utils.arrays import check_array
from ..utils.misc import valid_parameters


@dataclass
class PipelineStep:
    ''' Data class representing a single step in a `Pipeline` '''
    learner: Type[Learner]
    name: str = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


class Pipeline(Learner):
    '''
    Meta-learner that consists of a series of steps that are themselves learners. The Pipeline
    implements all the functions expected from `SupervisedLearner`, `UnsupervisedLearner` and
    `BaseTransfomer`. A Pipeline chains calls depending on the type of the underlying steps.

    Pipelines are a very important component of the ML framework. It allows us to *chain* a series of
    transformers and estimators together, so the output of one is used as the input to the next.
    Further, objects within a `Pipeline` will signal each other when the input/output has changed in
    shape. For examples and more information about input shape change handling, see the [change map
    documentation](../changemap/index.md).

    Generally, a pipeline consists of a series of transformers followed by an estimator as the last
    step. However, a pipeline may contain only transformers, a single step, or even other pipelines!

    ## Traversing
    Pipelines are traversed by calling `fit` followed by `predict` on each element of the pipeline. The
    output of each step's `predict` is used as the input for the next step. For steps in which the
    learner is a transformer, `transform` is called instead of `predict`.

    ## Examples
    For example, a typical Pipeline would consist of N transformers (T1, T2, ..., Tn) and one
    Learner (L). When users call `Pipeline.fit(X, y)`, the following happens:

    ```python
    X = T1.fit(X).transform(X)
    X = T2.fit(X).transform(X)
    # ...
    X = Tn.fit(X).transform(X)
    L.fit(X, y)
    ```

    Similarly, caling `Pipeline.predict(X)` will call `fit` followed by `transform` for all
    transformers, and then `predict` using the final learner.
    '''

    def __init__(self, steps: List[PipelineStep], verbose: bool = False):
        '''
        Parameters
        ----------
        steps : List[PipelineStep]
            Flat iterable of components of this pipeline, which are `PipelineStep` objects
        verbose : bool
            Logging verbosity enabled
        '''
        super().__init__(verbose=verbose)

        # Process steps from given arguments
        self.steps: Dict[str, Learner] = OrderedDict()
        for i, step in enumerate(steps):

            # Make sure that all steps are the correct type
            assert isinstance(step, PipelineStep) and issubclass(step.learner, Learner), \
                'Step %d has unknown type. Expected PipelineStep, found %r' % (i, step)

            # Name defaults to index
            name = str(i) if step.name is None else step.name

            # Instantiate learner and add it to internal field
            learner_instance = step.learner(**step.kwargs)
            self.steps[name] = learner_instance

        # Setup event callbacks linking one step to the next
        learner_iter = [learner for (name, learner) in self.steps.items()]
        for i, curr_estimator in enumerate(learner_iter[:-1]):
            next_estimator = learner_iter[i + 1]
            curr_estimator.add_output_shape_changed_callback(next_estimator.on_input_shape_changed)

        # Link last step's output callback to pipeline's
        learner_iter[-1].add_output_shape_changed_callback(self.on_output_shape_changed)

        self.verbose = verbose

    def __repr__(self):
        return 'Pipeline(steps=%r)' % self.steps

    @staticmethod
    def _call_method(instance, method_name, *args, **kwargs):
        method = getattr(instance, method_name)
        kwargs = {name: value for name, value in kwargs.items()
                  if name in signature(method).parameters}
        return method(*args, **kwargs)

    def is_transformer_only(self):
        ''' Recursively query to check if this pipeline is made of only transformers '''
        for _, step in self.steps.items():
            if not (isinstance(step, BaseTransformer) or
                (isinstance(step, Pipeline) and step.is_transformer_only())):
                return False
        return True

    def _traverse(self, X: Iterable[Iterable], y: Iterable = None, inter_method: str = None,
                  final_method: str = None, inverse: bool = False):
        '''
        Recursively iterate over the steps and call `inter_method` followed by `transform` on
        all the steps except of the last one. Returns the result of `final_method` applied to
        the last step of the pipeline. Passing `None` as the method name results in no-op.
        '''
        # Special case: if the pipeline is all transformers, there is no final estimator
        all_transformers = self.is_transformer_only()
        iter_steps = list(self.steps.items())
        steps = iter_steps if all_transformers else iter_steps[:-1]

        # Reverse steps if we are doing the inverse operation
        if inverse: steps = steps[::-1]

        # Iterate over each step
        for _, estimator in steps:
            if inter_method is not None:
                # `inter_method` is expected to return `self`
                estimator = Pipeline._call_method(estimator, inter_method, X, y=y)
            X = Pipeline._call_method(
                estimator, 'inverse_transform' if inverse else 'transform', X, y=y)

        # Final estimator depends on pipeline characteristics
        _, final_estimator = (None, None) if all_transformers else iter_steps[-1]
        return check_array(X) if inverse or not hasattr(final_estimator, final_method) else \
            Pipeline._call_method(final_estimator, final_method, X, y=y)

    def fit(self, X: Iterable[Iterable], y: Iterable = None):
        '''
        Recursively iterate over the steps and call `fit` on all, including last.

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
        Pipeline
            Instance of self

        '''
        self._traverse(X, y=y, inter_method='fit', final_method='fit')
        return self

    def predict(self, X: Iterable[Iterable]):
        '''
        Recursively iterate over the steps and call `transfrom` on all steps, then `predict` on the
        final one.

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
        return self._traverse(X, final_method='predict')

    def transform(self, X: Iterable[Iterable]):
        '''
        Recursively iterate over the steps and call `transform` on all steps, then again `transform`
        on the final one.

        Parameters
        ----------
        X : Iterable[Iterable]
            Input samples to be transformed, expects a list of input samples in the form of a list
            of columns

        Returns
        -------
        Iterable
            Transformed output values in the form of a 1D list of values
        '''
        return self._traverse(X, final_method='transform')

    def score(self, X: Iterable[Iterable], y: Iterable):
        '''
        Recursively iterate over the steps and call `transfrom` on all steps, then `score` on the
        final one.

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
        return self._traverse(X, y=y, final_method='score')

    def inverse_transform(self, X: Iterable[Iterable]):
        '''
        Recursively iterate over the steps in reverse order and call `inverse_transform` on all but
        the final step.

        Parameters
        ----------
        X : Iterable[Iterable]
            Input samples to inverse a transformation for, expects a list of training samples in the
            form of a list of columns

        Returns
        -------
        Iterable
            Inverse-transformed output values in the form of a 1D list of values

        '''
        return self._traverse(X, inverse=True)

    @staticmethod
    def hyperparameters(dataset: DataSet, steps: Iterable[PipelineStep]):
        steps_out = []
        for i, step in enumerate(steps):
            name = step.name or str(i + 1)
            has_steps = issubclass(step.learner, Pipeline) and \
                valid_parameters(step.learner.__init__, {'steps': None})
            opts = {'steps': step.kwargs.get('steps')} if has_steps else {}
            step_kwargs_filtered = {k: v for k, v in step.kwargs.items() if k != 'steps'}
            step_kwargs = {**step.learner.hyperparameters(dataset, **opts), **step_kwargs_filtered}
            steps_out.append(PipelineStep(name=name, learner=step.learner, kwargs=step_kwargs))
        return {'steps': steps_out}

    def on_input_shape_changed(self, change_map: ChangeMap):
        list(self.steps.items())[0][1].on_input_shape_changed(change_map)
