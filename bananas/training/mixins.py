from math import log10
from typing import List
from ..sampling.cross_validation import DataSplit
from ..training.criteria import TrainStep, HaltCriteria
from ..statistics.window import WindowStats
from ..utils import tqdm_


class TrainableMixin(object):
    '''
    Mixin that adds a `train()` function to estimators.

    # Input function
    The input function provided to `train` can be any function that abides by the following two
    requirements:

    1. If called with no arguments, it produces a set of samples.
    2. It accepts an optional argument, `subset`, which would be `DataSplit.TEST`.

    When the `subset` argument is passed, the input function is expected to draw from the test
    subset of input data.

    # Sampling
    Instead of implementing a custom input function, users are encouraged to use one of the classes
    provided in the [sampling module](../sampling/index.md). To avoid loading all data into memory,
    take a look at implementing a [custom data class](../sampling/index.md#custom-data-class) that
    is compatible with the sampling classes.
    '''

    best_score_: float = None
    running_score_: float = None
    best_score_validation_: float = None
    running_score_validation: float = None

    # pylint: disable=too-many-arguments
    def train(self, input_fn: callable, halt_criteria: List[HaltCriteria] = None,
              max_steps: int = 10000, max_score: float = None, progress: bool = False,
              callback: callable = None):
        '''
        Keeps drawing samples from `input_fn` and fitting them to the underlying estimator
        while scoring against test dataset.
        '''

        # Process arguments
        if halt_criteria is None:
            halt_criteria = [HaltCriteria.IMPROVE_SCORE]
            if max_score is not None:
                halt_criteria += [HaltCriteria.TARGET_SCORE]
        input_fn_test = lambda: input_fn(subset=DataSplit.TEST)

        # Initialize train step object to keep track of iterations
        step_data = TrainStep(max_steps=max_steps, max_score=max_score)
        if getattr(self, 'verbose', False):
            self.print('Starting training with %d max steps' % max_steps)

        # Keep the last handful of scores to compute a running average
        window_scores = WindowStats(window_size=8)

        # Iterate over max_steps, show progress bar if requested
        iterations = range(max_steps)
        iterations_desc = self.__class__.__name__
        if progress:
            # TODO: overwrite self.print to use tqdm.write temporarily
            # TODO: if pipeline, use name of estimator step for description
            iterations = tqdm_(range(max_steps), desc=iterations_desc, leave=False)

        # Compute the maximum log10 that we care about for the purpose of printing progress
        max_n = int(log10(max_steps))

        # Used as a workaround for https://github.com/tqdm/tqdm/issues/352
        break_flag = False

        for idx in iterations:

            # Avoid breaking out of the loop, just keep calling continue until this is over
            if break_flag: continue

            # Print progress at 1, 2, 3, ..., 10, 20, 30, ..., 100, 200, 300, ..., 1000, 2000, ...
            if any([idx < (10 ** (n + 1)) and (idx + 1) % (10 ** n) == 0 for n in range(max_n)]):
                loop_log = 'Training iteration %d/%d.' % (idx + 1, max_steps)
                test_log = 'Current test score: %.03f.' % (self.running_score_ or 0)
                self.print('%s %s' % (loop_log, test_log))

            # Keep progress bar updated with latest score
            if progress and self.running_score_:
                iterations.set_description(iterations_desc + ' (%.03f)' % self.running_score_)

            # Pull a sample for training and fit the estimator with it
            X_train, y_train = input_fn()
            self.fit(X_train, y_train)

            # Keep track of the test scores after every fit
            X_test, y_test = input_fn_test()
            score = self.score(X_test, y_test)

            # Update running score average
            window_scores.push(score)
            self.running_score_ = window_scores.mean_
            step_data.running_score = self.running_score_

            # Update best score if appropriate
            if self.best_score_ is None or self.running_score_ > self.best_score_:
                step_data.best_idx = idx
                self.best_score_ = score
                step_data.best_score = self.best_score_

            # Update train step object
            step_data.idx = idx
            step_data.score = score
            step_data.X_test = X_test
            step_data.y_test = y_test
            step_data.X_train = X_train
            step_data.y_train = y_train
            step_data.iterator = iterations

            # Exit criteria: custom user criteria
            if callback is not None and callback(step_data):
                break_flag = True

            # Exit criteria: canned exit criteria
            if any([crit(step_data) for crit in halt_criteria]):
                break_flag = True

        # https://github.com/tqdm/tqdm/issues/352 ?
        if progress:
            iterations.clear()
            iterations.close()

        return self
    # pylint: enable=too-many-arguments
