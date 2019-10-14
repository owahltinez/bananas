from enum import Enum, auto
from typing import Iterable
from .base import _BaseSampler
from .cross_validation import OrderedCVSampler, RandomCVSampler
from .ordered import OrderedSampler
from .random import RandomSampler
from .stratified import StratifiedSampler

class SamplingStrategy(Enum):
    ''' Strategy used to draw samples from dataset '''
    ORDERED = auto()
    RANDOM = auto()
    STRATIFIED = auto()
    CROSS_VALIDATION_ORDERED = auto()
    CROSS_VALIDATION_RANDOM = auto()
    CROSS_VALIDATION_STRATIFIED = auto()

    @staticmethod
    def create(values: Iterable, strategy: 'SamplingStrategy' = None,
               **sampling_kwargs) -> _BaseSampler:
        '''
        Create an instance of a sampler. Defaults to `SamplingStrategy.CROSS_VALIDATION_RANDOM`.

        Parameters
        ----------
        values : Iterable
            List to draw samples from, can be of any iterable type but for some samplers a numpy
            array will be more efficient
        strategy : SamplingStrategy, optional
            Strategy to use when sampling
        sampling_kwargs : dict, optional
            Additional parameters to pass to the sampler constructor
        '''
        sampler: _BaseSampler = None
        if strategy is None: strategy = DEFAULT_SAMPLING_STRATEGY
        if strategy == SamplingStrategy.ORDERED:
            sampler = OrderedSampler(values, **sampling_kwargs)
        if strategy == SamplingStrategy.RANDOM:
            sampler = RandomSampler(values, **sampling_kwargs)
        # if strategy == SamplingStrategy.STRATIFIED:
        #     sampler = StratifiedSampler(values, **sampling_kwargs)
        if strategy == SamplingStrategy.CROSS_VALIDATION_ORDERED:
            sampler = OrderedCVSampler(values, **sampling_kwargs)
        if strategy == SamplingStrategy.CROSS_VALIDATION_RANDOM:
            sampler = RandomCVSampler(values, **sampling_kwargs)
        if strategy == SamplingStrategy.CROSS_VALIDATION_STRATIFIED:
            sampler = StratifiedSampler(values, **sampling_kwargs)
        if sampler is None:
            raise TypeError('Unexpected sampler requested. It must be one of: %r' %
                            [strategy_.name for strategy_ in SamplingStrategy])
        return sampler


class ReplaceStrategy(Enum):
    ''' Strategy used to replace values while sampling '''
    MEAN = auto()
    DROP = auto()


DEFAULT_SAMPLING_STRATEGY = SamplingStrategy.CROSS_VALIDATION_RANDOM
