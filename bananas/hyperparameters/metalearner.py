''' Meta-Learner Module '''

from functools import partial
from multiprocessing.pool import ThreadPool as Pool
from typing import Dict, Generator

from ..core.learner import Learner
from ..training.mixins import TrainableMixin
from ..utils import tqdm_


class _MetaLearner(Learner, TrainableMixin):
    '''
    Learner composed of other learners. This class can't be used as-is. Subclasses must override
    the `_iter_learners` method to yield instance of learners in the desired order of iteration.
    '''

    def __init__(self, n_jobs: int = 1, verbose: bool = False, **kwargs):
        '''
        Parameters
        ----------
        n_jobs : int
            TODO
        verbose : bool
            TODO
        '''
        super().__init__(n_jobs=n_jobs, verbose=verbose, **kwargs)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._scores_cache = {}
        self._learners_cache: Dict[str, Learner] = {}
        self.best_score_: float = 0.
        self.best_learner_: Learner = None

    def _iter_learners(self) -> Generator[Learner, None, None]:
        raise NotImplementedError('Subclasses of `%s` must override method `%s`' %
                                  self.__class__.__name__, '_iter_learners')

    def fit(self, X, y=None):
        ''' Fits data for all learners by calling `fit()` '''
        for learner in self._iter_learners():
            learner.fit(X, y)
        if getattr(self, 'best_learner_', None) is None:
            self.best_learner_ = next(self._iter_learners())
        return self

    def score_all(self, X, y=None):
        ''' Iterates over all learners and returns pairs of (learner, score) '''
        for learner in self._iter_learners():
            yield learner, learner.score(X, y=y)

    def _score_all(self, X, y=None):
        ''' Private version of `score_all` that updates best_score_ and best_learner_ each time
            it is called. Returns the best score. '''
        delattr(self, 'best_score_')
        for learner, score in self.score_all(X, y=y):
            self._scores_cache[learner] = score

            if not hasattr(self, 'best_score_') or score > self.best_score_:
                self.best_score_ = score
                self.best_learner_ = learner

        return self.best_score_

    def predict(self, X, y=None):
        ''' Predicts batch against the best learner '''
        self.check_attributes('best_learner_')
        return self.best_learner_.predict(X)

    def score(self, X, y=None):
        ''' Scores batch against the best learner '''
        self.check_attributes('best_learner_')
        return self.best_learner_.score(X, y=y)

    def _train_worker(self, *train_args, **train_kwargs):
        # Early exit: if goal score has been reached
        if hasattr(self, 'best_score_') and \
            self.best_score_ >= train_kwargs.get('max_score', 1): return

        # Call training on underlying learner
        learner = train_args[-1]
        learner.train(*train_args[:-1], **train_kwargs)

        # Update best score
        if not hasattr(self, 'best_score_') or learner.best_score_ > self.best_score_:
            self.best_learner_ = learner
            self.best_score_ = learner.best_score_

    def train(self, *train_args, progress: bool = False, **train_kwargs):
        ''' Trains all learners with data by calling `learner.train(...)` on each one '''

        learners = list(self._iter_learners())
        train_kwargs = {'progress': progress, **train_kwargs}

        # Build partial function combining parameters
        worker = partial(self._train_worker, *train_args, **train_kwargs)

        if self.n_jobs > 1:
            futures = Pool(self.n_jobs).imap_unordered(worker, learners)
        else:
            futures = (worker(learner) for learner in learners)

        if progress:
            # TODO: overwrite self.print to use tqdm.write temporarily
            futures = tqdm_(
                futures, desc='%s parameter sets' % type(self).__name__, total=len(learners), leave=False)

        # Consume all the future results
        for _ in futures: pass

        return self
