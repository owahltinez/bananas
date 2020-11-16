""" Test Datasets Module """

import random
from bananas.statistics import RandomState
from bananas.sampledata.local import load_bike, load_boston, load_california, load_titanic
from bananas.sampledata.synthetic import (
    new_labels,
    new_line,
    new_poly,
    new_trig,
    new_wave,
    new_mat9,
)
from bananas.utils.constants import SAMPLE_SIZE_SMALL

from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):
    def test_synthetic(self):
        rng = RandomState(0)
        for ds_fn in (new_labels, new_line, new_mat9, new_poly, new_trig, new_wave):
            n = rng.randint(1, SAMPLE_SIZE_SMALL)
            dataset = ds_fn(samples=n, random_seed=0)
            self.assertTrue(dataset is not None)
            self.assertEqual(n, dataset.count)

    def test_load_csv(self):
        for ds_fn in (load_bike, load_boston, load_california, load_titanic):
            ds_train, ds_test = ds_fn()
            self.assertTrue(ds_train is not None)
            self.assertTrue(ds_test is not None)
            self.assertGreater(len(ds_train), 100)
            self.assertGreater(len(ds_test), 100)

            sample_train_X, sample_train_target = ds_train.input_fn()
            sample_test_X, sample_test_target = ds_test.input_fn()
            self.assertGreater(len(sample_train_X[0]), 100)
            self.assertGreater(len(sample_train_target), 100)
            self.assertGreater(len(sample_test_X[0]), 100)
            self.assertGreater(len(sample_test_target), 100)


main()
