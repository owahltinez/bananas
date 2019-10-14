''' Test Hyper Parameters Module '''

import sys
import cProfile
import warnings
from pstats import Stats
from unittest import TestCase, main as main_

# Show traceback for all warninngs
from bananas.utils.misc import warn_with_traceback


# pylint: disable=missing-docstring
class ProfilingTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.showwarning = warnings.showwarning
        warnings.showwarning = warn_with_traceback
        cls.profiler = cProfile.Profile()
        cls.profiler.enable()

    @classmethod
    def tearDownClass(cls):
        warnings.showwarning = cls.showwarning
        stats = Stats(cls.profiler)
        stats.strip_dirs()
        stats.sort_stats('cumtime')
        stats.print_stats(20)

def main():
    if __name__ == '__main__':
        sys.exit(main_())
