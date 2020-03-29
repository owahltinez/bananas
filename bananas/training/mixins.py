import time
from math import log10
from typing import List
from ..sampling.cross_validation import DataSplit
from ..training.criteria import HaltCriteria
from ..training.train_history import TrainHistory
from ..training.train_step import TrainStep
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

    # pylint: disable=too-many-arguments
    def train(self, input_fn: callable, halt_criteria: List[HaltCriteria] = None,
              max_steps: int = 10000, max_score: float = None, progress: bool = False,
              callback: callable = None, **input_fn_opts):
        '''
        Keeps drawing samples from `input_fn` and fitting them to the underlying estimator
        while scoring against test dataset.
        '''

        # Process halt criteria
        if halt_criteria is None:
            halt_criteria = [HaltCriteria.IMPROVE_SCORE]
            if max_score is not None:
                halt_criteria += [HaltCriteria.TARGET_SCORE]

        # Create a test subset sampler
        input_fn_test = lambda **kwargs: input_fn(subset=DataSplit.TEST, **kwargs)

        # Initialize train step and history objects to keep track of iterations
        history = TrainHistory()
        step_data = TrainStep(max_steps=max_steps, max_score=max_score)
        if getattr(self, 'verbose', False):
            self.print('Starting training with %d max steps' % max_steps)

        # Keep the last handful of scores to compute a running average
        window_scores = WindowStats(window_size=8)

        # Iterate over max_steps, show progress bar if requested
        if progress:
            progress_desc = self.__class__.__name__
            # TODO: overwrite self.print to use tqdm.write temporarily
            # TODO: if pipeline, use name of estimator step for description

            # Creating progress bar sometimes fails due to threading issues in tqdm, there's nothing
            # we can do to work around that except for retrying until it works (up to 100 times)
            tqdm_exception = None
            for _ in range(100):
                try:
                    progress_bar = tqdm_(range(max_steps), desc=progress_desc, leave=False)
                    break
                except AssertionError as exc:
                    self.print(exc)
                    tqdm_exception = exc
                    time.sleep(0.01)
            if not progress_bar:
                raise tqdm_exception

        # Compute the maximum log10 that we care about for logging progress
        max_n = int(log10(max_steps))

        # Keep track of the time at which training begun and begin training loop
        train_start = time.monotonic_ns()
        for idx in range(max_steps):

            # Avoid breaking out of the loop, just keep calling continue until this is over
            # Used as a workaround for https://github.com/tqdm/tqdm/issues/352
            if history.early_exit: continue

            # Print progress at 1, 2, 3, ..., 10, 20, 30, ..., 100, 200, 300, ..., 1000, 2000, ...
            if any([idx < (10 ** (n + 1)) and (idx + 1) % (10 ** n) == 0 for n in range(max_n)]):
                loop_log = 'Training iteration %d/%d.' % (idx + 1, max_steps)
                test_log = 'Current test score: %.03f.' % (step_data.running_score)
                self.print('%s %s' % (loop_log, test_log))

            # Pull a sample for training and fit the estimator with it
            X_train, y_train = input_fn(**input_fn_opts)
            self.fit(X_train, y_train)

            # Keep track of the test scores after every fit
            X_test, y_test = input_fn_test(**input_fn_opts)
            score = self.score(X_test, y_test)

            # Update running score average
            window_scores.push(score)
            step_data.running_score = window_scores.mean_

            # Keep progress bar updated with latest running score
            if progress:
                progress_bar.set_description(progress_desc + ' (%.03f)' % step_data.running_score)
                progress_bar.update()

            # Update best score if appropriate
            if step_data.running_score > step_data.best_score:
                step_data.best_iteration = idx
                step_data.best_score = step_data.running_score

            # Update train step data
            step_data.iteration = idx
            step_data.score = score
            step_data.X_test = X_test
            step_data.y_test = y_test
            step_data.X_train = X_train
            step_data.y_train = y_train

            # Update history data
            history.scores.append(score)
            history.iterations = idx + 1

            # Exit criteria: custom user criteria
            if callback is not None and callback(step_data):
                history.early_exit = True

            # Exit criteria: canned exit criteria
            if any([crit(step_data) for crit in halt_criteria]):
                history.early_exit = True

        # Dispose of the progress bar
        if progress:
            progress_bar.clear()
            progress_bar.close()

        # Update the training time and return history
        history.time_millis = (time.monotonic_ns() - train_start) // 1000
        return history
    # pylint: enable=too-many-arguments
