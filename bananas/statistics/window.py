"""
Statistical windowing functions, used for simple running stats use cases that don't require a
full-blown transformer.
"""

from typing import List
from .basic import mean, median, variance


class WindowStats:
    """
    Keeps track of basic statistics using only the last `window_size` samples.
    """

    def __init__(self, window_size: int = 32):
        self._window_size = window_size
        self.min_: float = None
        self.max_: float = None
        self.mean_: float = None
        self.median_: float = None
        self.variance_: float = None
        self.history_: List[float] = []

    def push(self, sample: float):
        """
        Add a sample to this object's history. If we exceed the window size, the oldest sample is
        removed.
        """
        while len(self.history_) >= self._window_size:
            self.history_.pop(0)

        self.history_.append(sample)
        self.min_ = min(self.history_)
        self.max_ = max(self.history_)
        self.mean_ = mean(self.history_)
        self.median_ = median(self.history_)
        if len(self.history_) > 3:
            self.variance_ = variance(self.history_)
