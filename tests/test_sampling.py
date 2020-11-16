""" Test Utils Module """

import sys
import numpy
from bananas.sampling.ordered import OrderedSampler
from bananas.sampling.random import RandomSampler
from bananas.sampling.cross_validation import OrderedCVSampler, RandomCVSampler, DataSplit
from bananas.sampling.stratified import StratifiedSampler
from bananas.utils.arrays import concat_arrays, take_elems

from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):
    def test_from_1d_list(self):
        batch_size = 10
        num_batches = 5
        dataset = sum([[i] * batch_size for i in range(num_batches)], [])
        input_fn = OrderedSampler(dataset, batch_size=batch_size, epochs=1)
        for i in range(num_batches):
            sample = input_fn()
            self.assertEqual(len(sample), batch_size)
            self.assertListEqual(sample, [i] * batch_size)
        self.assertRaises(IndexError, input_fn)

    def test_from_numpy(self):
        batch_size = 10
        num_batches = 5
        num_features = 5
        ones = numpy.ones((batch_size, num_features))
        dataset = concat_arrays(*[ones * i for i in range(num_batches)])
        input_fn = OrderedSampler(dataset, batch_size=batch_size, epochs=1)
        for i in range(num_batches):
            sample = input_fn()
            self.assertEqual(batch_size, len(sample))
            self.assertListEqual(sample, (ones * i).tolist())
        self.assertRaises(IndexError, input_fn)

    def test_indices_from_numpy(self):
        batch_size = 10
        num_batches = 5
        num_features = 5
        ones = numpy.ones((batch_size, num_features))
        dataset = concat_arrays(*[ones * i for i in range(num_batches)])
        sampler = OrderedSampler(dataset, batch_size=batch_size, epochs=1)
        input_fn = lambda: take_elems(dataset, sampler.indices())
        for i in range(num_batches):
            sample = input_fn()
            self.assertEqual(batch_size, len(sample))
            self.assertListEqual(sample, (ones * i).tolist())
        self.assertRaises(IndexError, input_fn)

    def test_from_getitem(self):
        batch_size = 10
        num_batches = 5
        num_features = 5
        ones = numpy.ones((batch_size, num_features))

        class MyDataset:
            def __getitem__(self, idx):
                return (ones[0] * (idx // batch_size)).tolist()

        dataset = MyDataset()
        input_fn = OrderedSampler(
            dataset, batch_size=batch_size, epochs=1, input_size=num_batches * batch_size
        )
        for i in range(num_batches):
            sample = input_fn()
            self.assertEqual(len(sample), batch_size)
            self.assertListEqual(sample, (ones * i).tolist())
        self.assertRaises(IndexError, input_fn)

    def test_indices_from_getitem(self):
        batch_size = 10
        num_batches = 5
        num_features = 5
        ones = numpy.ones((batch_size, num_features))

        class MyDataset:
            def __getitem__(self, idx):
                return (ones[0] * (idx // batch_size)).tolist()

        dataset = MyDataset()
        sampler = OrderedSampler(
            dataset, batch_size=batch_size, epochs=1, input_size=num_batches * batch_size
        )
        input_fn = lambda **kwargs: take_elems(dataset, sampler.indices(**kwargs))
        for i in range(num_batches):
            sample = input_fn()
            self.assertEqual(len(sample), batch_size)
            self.assertListEqual(sample, (ones * i).tolist())
        self.assertRaises(IndexError, input_fn)

    def test_override_batch_size(self):
        batch_size = 10
        num_batches = 5
        num_features = 5
        ones = numpy.ones((batch_size, num_features))
        dataset = concat_arrays(*[(ones * i).tolist() for i in range(num_batches)])
        sampler = OrderedSampler(dataset, batch_size=batch_size, epochs=1)
        sample = sampler(batch_size=len(dataset))
        self.assertEqual(len(dataset), len(sample))
        self.assertListEqual(sample, dataset)
        self.assertRaises(IndexError, sampler)

    def test_indices_override_batch_size(self):
        batch_size = 10
        num_batches = 5
        num_features = 5
        ones = numpy.ones((batch_size, num_features))
        dataset = concat_arrays(*[(ones * i).tolist() for i in range(num_batches)])
        sampler = OrderedSampler(dataset, batch_size=batch_size, epochs=1)
        sample = take_elems(dataset, sampler.indices(batch_size=len(dataset)))
        self.assertEqual(len(dataset), len(sample))
        self.assertListEqual(sample, dataset)
        self.assertRaises(IndexError, sampler)

    def test_ordered_sampler_blacklist(self):
        batch_size = 10
        num_batches = 5
        dataset = sum([[i] * batch_size for i in range(num_batches)], [])
        sampler = OrderedSampler(dataset, batch_size=batch_size, blacklist=set(range(10)))
        sample = sampler(batch_size=len(dataset) - batch_size)
        self.assertEqual(len(dataset) - batch_size, len(sample))
        # Verify that the returned sample has skipped the indices
        self.assertListEqual(sample, dataset[batch_size:])

    def test_random_sampler_blacklist(self):
        batch_size = 10
        num_batches = 5
        dataset = sum([[i] * batch_size for i in range(num_batches)], [])
        sampler = RandomSampler(dataset, batch_size=batch_size, blacklist=set(range(10)))

        num_iters = 10
        for _ in range(num_iters):
            sample = sampler(batch_size=len(dataset) - batch_size)
            self.assertEqual(len(dataset) - batch_size, len(sample))
            # Verify that the returned sample has skipped the indices
            self.assertTrue(all([idx not in dataset[:batch_size] for idx in sample]))

    def test_cross_validation_1d(self):
        batch_size = 10
        num_batches = 5
        dataset = sum([[i] * batch_size for i in range(num_batches)], [])
        input_fn = OrderedCVSampler(dataset, batch_size=batch_size, epochs=1, test_split=0.2)

        for i in range(num_batches - 1):
            # Get samples from the default subset
            sample = input_fn()
            # Verify that each sample produces exactly `batch_size` items
            self.assertEqual(len(sample), batch_size)
            # Verify that each sample comes out in the expected order
            self.assertListEqual(sample, [i] * batch_size)

        # Get one last sample from the test subset
        sample = input_fn(subset=DataSplit.TEST)
        # Verify that it corresponds to the last batch
        self.assertListEqual(sample, [num_batches - 1] * batch_size)
        # Drawing any more samples should put us over the next epoch
        self.assertRaises(IndexError, input_fn)

    def test_cross_validation_2d(self):
        batch_size = 10
        num_batches = 5
        num_features = 5
        ones = numpy.ones((batch_size, num_features))
        dataset = concat_arrays(*[ones * i for i in range(num_batches)])
        input_fn = OrderedCVSampler(dataset, batch_size=batch_size, epochs=1, test_split=0.2)

        for i in range(num_batches - 1):
            sample = input_fn()
            self.assertEqual(batch_size, len(sample))
            self.assertListEqual(sample, (ones * i).tolist())
        sample = input_fn(subset=DataSplit.TEST)
        self.assertListEqual(sample, (ones * (num_batches - 1)).tolist())
        self.assertRaises(IndexError, input_fn)

    def test_indices_cross_validation(self):
        batch_size = 10
        num_batches = 5
        num_features = 5
        ones = numpy.ones((batch_size, num_features))
        dataset = concat_arrays(*[ones * i for i in range(num_batches)])
        sampler = OrderedCVSampler(dataset, batch_size=batch_size, epochs=1, test_split=0.2)
        input_fn = lambda **kwargs: take_elems(dataset, sampler.indices(**kwargs))

        for i in range(num_batches - 1):
            sample = input_fn()
            self.assertEqual(batch_size, len(sample))
            self.assertListEqual(sample, (ones * i).tolist())
        sample = input_fn(subset=DataSplit.TEST)
        self.assertListEqual(sample, (ones * (num_batches - 1)).tolist())
        self.assertRaises(IndexError, input_fn)

    def test_ordered_cross_validation_force_validation_subset(self):
        batch_size = 10
        num_batches = 5
        validation_split = 0.4
        dataset = sum([[i] * batch_size for i in range(num_batches)], [])
        sampler = OrderedCVSampler(
            dataset,
            batch_size=batch_size,
            epochs=1,
            test_split=0.2,
            validation_split=validation_split,
        )

        # Verify that the size of the test and validation subsets is correct
        self.assertEqual(
            sampler.test_size, int(batch_size * num_batches * 0.2 * (1 - validation_split))
        )
        self.assertEqual(
            sampler.validation_size, int(batch_size * num_batches * 0.2 * validation_split)
        )

        for i in range(num_batches - 1):
            # Get samples from the default subset
            sample = sampler()
            # Verify that each sample produces exactly `batch_size` items
            self.assertEqual(len(sample), batch_size)
            # Verify that each sample comes out in the expected order
            self.assertListEqual(sample, [i] * batch_size)

        # Get one sample from the test subset
        test_sample = sampler.indices(subset=DataSplit.TEST)
        # Verify that it corresponds to the last batch
        test_sample_vals = [dataset[idx] for idx in test_sample]
        self.assertListEqual(test_sample_vals, [num_batches - 1] * batch_size)

        # Get one sample from the validation subset
        validation_sample = sampler.indices(subset=DataSplit.VALIDATION)
        # Verify that it corresponds to the last batch
        validation_sample_vals = [dataset[idx] for idx in validation_sample]
        self.assertListEqual(validation_sample_vals, [num_batches - 1] * batch_size)

        # Verify that none of the items in the test sample are part of the validation subset
        self.assertTrue(all([i not in test_sample for i in validation_sample]))

        # Drawing any more samples should put us over the next epoch
        self.assertRaises(IndexError, sampler)

    def test_random_cross_validation_force_validation_subset(self):
        batch_size = 20
        num_batches = 5
        validation_split = 0.4
        dataset = sum([[i] * batch_size for i in range(num_batches)], [])
        sampler = RandomCVSampler(
            dataset, batch_size=batch_size, test_split=0.2, validation_split=validation_split
        )

        # Verify that the size of the test and validation subsets is correct
        self.assertEqual(
            sampler.test_size, int(batch_size * num_batches * 0.2 * (1 - validation_split))
        )
        self.assertEqual(
            sampler.validation_size, int(batch_size * num_batches * 0.2 * validation_split)
        )

        for i in range(num_batches - 1):
            # Get samples from the default subset
            sample = sampler()
            # Verify that each sample produces exactly `batch_size` items
            self.assertEqual(len(sample), batch_size)
            # Verify that each sample does not out in exact order
            self.assertTrue(any(j != k for j, k in zip(sample, [i] * batch_size)))

        num_iters = 10
        for _ in range(num_iters):

            # Get one sample from the test subset
            test_sample = sampler.indices(subset=DataSplit.TEST)
            # Verify that each sample does not out in exact order
            self.assertTrue(
                any(j != k for j, k in zip(test_sample, [num_batches - 1] * batch_size))
            )

            # Get one sample from the validation subset
            validation_sample = sampler.indices(subset=DataSplit.VALIDATION)
            # Verify that each sample does not out in exact order
            self.assertTrue(
                any(j != k for j, k in zip(validation_sample, [num_batches - 1] * batch_size))
            )

            # Verify that none of the items in the test sample are part of the validation subset
            self.assertTrue(all([i not in test_sample for i in validation_sample]))

    def test_stratified_1d(self):
        batch_size = 10
        num_batches = 5
        label_dist = 0.2

        # Pick a class such that we pick 0 and 1 satisfying the desired `label_dist` percent
        pick_class = lambda i: 0 if i < int(num_batches * label_dist) else 1
        dataset = sum([[pick_class(i)] * batch_size for i in range(num_batches)], [])
        input_fn = StratifiedSampler(dataset, batch_size=batch_size, test_split=0.0)

        for _ in range(num_batches):
            sample = input_fn()
            one_count = sum(sample)
            zero_count = len(sample) - one_count
            self.assertEqual(zero_count / len(sample), label_dist)

    # def test_stratified_2d(self):
    #     batch_size = 10
    #     num_batches = 5
    #     num_features = 5
    #     ones = numpy.ones((batch_size, num_features)).astype(int)
    #     dataset = numpy.concatenate(*[[ones * i for i in range(num_batches)] * 1000])
    #     sampler = StratifiedSampler(dataset[:, -1], batch_size=batch_size, test_split=0.)
    #     input_fn = lambda **kwargs: take_elems(dataset, sampler(**kwargs))
    #     for _ in range(num_batches):
    #         sample = input_fn()
    #         X, y = sample[:, :-1], sample[:, -1].astype(int)
    #         self.assertEqual(X.shape[0], batch_size)
    #         self.assertEqual(y.shape[0], batch_size)
    #         for i in range(num_batches):
    #             print(numpy.where(y == i))
    #             self.assertEqual(2, len(numpy.where(y == i)[0]))

    # TODO: test build_input_fn when:
    # * input size == batch size
    # * test size > input size
    # * etc.


if __name__ == "__main__":
    sys.exit(main())
