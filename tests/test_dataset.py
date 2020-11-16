""" Test Utils Module """

import os
import numpy
from pathlib import Path

from bananas.dataset import DataType, Feature, DataSet
from bananas.testing.generators import (
    generate_array_booleans,
    generate_array_chars,
    generate_array_floats,
    generate_array_ints,
    generate_array_int_floats,
    generate_array_uints,
    generate_array_nones,
    generate_array_strings,
    generate_images,
    generate_onehot_matrix,
    generate_array_infinities,
)
from bananas.utils.constants import SAMPLE_SIZE_SMALL

from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):
    def test_check_feature_types(self):
        onehot = generate_onehot_matrix()
        self.assertEqual(DataType.ONEHOT, DataType.parse(onehot))

        image_bw = generate_images(n=128, c=1, w=32, h=32).reshape(128, 32, 32)
        self.assertEqual(DataType.HIGH_DIMENSIOAL, DataType.parse(image_bw))

        image_rgb = generate_images(n=128, c=3, w=32, h=32)
        self.assertEqual(DataType.HIGH_DIMENSIOAL, DataType.parse(image_rgb))

        binary = generate_array_booleans()
        self.assertEqual(DataType.BINARY, DataType.parse(binary))
        self.assertEqual(DataType.BINARY, DataType.parse(binary.astype(int)))
        self.assertEqual(DataType.BINARY, DataType.parse(binary.astype(numpy.uint8)))
        self.assertEqual(DataType.BINARY, DataType.parse(binary.reshape(-1, 1)))

        categorical = generate_array_ints()
        self.assertEqual(DataType.CATEGORICAL, DataType.parse(categorical))
        self.assertEqual(DataType.CATEGORICAL, DataType.parse(categorical.astype(numpy.uint8)))
        self.assertEqual(DataType.CATEGORICAL, DataType.parse(categorical.astype(numpy.float64)))
        self.assertEqual(DataType.CATEGORICAL, DataType.parse(categorical.reshape(-1, 1)))

        categorical = generate_array_chars()
        self.assertEqual(DataType.CATEGORICAL, DataType.parse(categorical))
        self.assertEqual(DataType.CATEGORICAL, DataType.parse(categorical.reshape(-1, 1)))

        categorical = generate_array_strings()
        self.assertEqual(DataType.CATEGORICAL, DataType.parse(categorical))
        self.assertEqual(DataType.CATEGORICAL, DataType.parse(categorical.reshape(-1, 1)))

        continuous = generate_array_floats()
        self.assertEqual(DataType.CONTINUOUS, DataType.parse(continuous))
        self.assertEqual(DataType.CONTINUOUS, DataType.parse(continuous.reshape(-1, 1)))

        vector = generate_array_ints().reshape(-1, 2)
        self.assertEqual(DataType.VECTOR, DataType.parse(vector))
        self.assertEqual(DataType.VECTOR, DataType.parse(vector.astype(numpy.uint8)))
        self.assertEqual(DataType.VECTOR, DataType.parse(vector.astype(numpy.float64)))

        unknown = generate_array_nones()
        self.assertEqual(DataType.UNKNOWN, DataType.parse(unknown))
        self.assertEqual(DataType.UNKNOWN, DataType.parse(unknown.reshape(-1, 1)))
        self.assertEqual(DataType.UNKNOWN, DataType.parse(unknown.reshape(-1, 4, 4)))

        self.assertTrue(DataType.is_categorical(binary))
        self.assertTrue(DataType.is_categorical(onehot))
        self.assertTrue(DataType.is_categorical(categorical))

    def test_feature_init(self):
        num = 1024
        data = generate_array_floats(n=num)

        feat = Feature(data)
        self.assertEqual(num, len(feat))
        self.assertEqual(num, len(feat[:]))
        self.assertTrue(numpy.array_equal(data, feat.values))
        self.assertTrue(numpy.array_equal(data[:num], feat[:num]))
        self.assertNotEqual(feat.sample_var, -1)

    def test_feature_indexing(self):
        num = 1024
        data = generate_array_floats(n=num)

        feat = Feature(data)
        self.assertEqual(10, len(feat[:10]))
        self.assertEqual(10, len(feat[0:10]))
        self.assertEqual(10, len(feat[0:10:1]))
        self.assertEqual(5, len(feat[0:10:2]))

    def test_feature_sampling(self):
        num = 1024
        data = generate_array_floats(n=num)

        feat1 = Feature(data, random_seed=0)
        feat2 = Feature(data, random_seed=0)
        self.assertTrue(numpy.array_equal(feat1.input_sample, feat2.input_sample))

    def test_feature_custom_loader(self):
        num = 1024
        data = generate_array_floats(n=num)

        class MyCustomDataLoader(object):
            def __len__(self):
                return len(data)

            def __getitem__(self, idx):
                return data[idx]

        feat1 = Feature(data, random_seed=0)
        feat2 = Feature(MyCustomDataLoader(), random_seed=0)
        self.assertTrue(numpy.array_equal(feat1.input_sample, feat2.input_sample))
        self.assertTrue(numpy.array_equal(feat1.sampler(), feat2.sampler()))

    def test_dataset_init(self):
        num = SAMPLE_SIZE_SMALL
        feat1 = Feature(generate_array_floats(n=num), name="feat1")
        feat2 = Feature(generate_array_floats(n=num), name="feat2")
        dataset = DataSet([feat1, feat2])

        self.assertEqual(num, dataset.count)
        self.assertEqual(num, dataset.count)
        self.assertTrue(numpy.array_equal(feat1.values, dataset.features["feat1"].values))
        self.assertTrue(numpy.array_equal(feat2.values, dataset.features["feat2"].values))
        self.assertTrue(numpy.array_equal(dataset["feat1", :num], feat1[:num]))
        self.assertTrue(numpy.array_equal(dataset["feat2", :num], feat2[:num]))

    def test_dataset_sampling(self):
        num = SAMPLE_SIZE_SMALL
        data = generate_array_floats(n=num)
        dataset1 = DataSet.from_ndarray(numpy.asarray([data]), random_seed=0)
        dataset2 = DataSet.from_ndarray(numpy.asarray([data]), random_seed=0)
        input_fn1 = dataset1.input_fn
        input_fn2 = dataset2.input_fn

        input_samples1 = dataset1.input_sample.items()
        input_samples2 = dataset2.input_sample.items()
        for (name1, sample1), (name2, sample2) in zip(input_samples1, input_samples2):
            self.assertEqual(name1, name2)
            self.assertTrue(numpy.array_equal(sample1, sample2), name1)
        for _ in range(10):
            X1, X2 = input_fn1(), input_fn2()
            self.assertTrue(numpy.array_equal(X1, X2))

    def test_dataset_feature_names(self):
        num = SAMPLE_SIZE_SMALL
        feat1 = Feature(generate_array_floats(n=num), name="a")
        feat2 = Feature(generate_array_floats(n=num), name="b")
        dataset = DataSet([feat1, feat2])

        self.assertEqual(num, dataset.count)
        self.assertEqual(num, dataset.count)
        self.assertTrue(numpy.array_equal(feat1.values, dataset.features["a"].values))
        self.assertTrue(numpy.array_equal(feat2.values, dataset.features["b"].values))
        self.assertTrue(numpy.array_equal(dataset["a", :num], feat1[:num]))
        self.assertTrue(numpy.array_equal(dataset["b", :num], feat2[:num]))

    def test_dataset_different_shapes(self):
        num = SAMPLE_SIZE_SMALL
        feat1 = Feature(generate_array_floats(n=num), name="feat1")
        feat2 = Feature(generate_onehot_matrix(n=num), name="feat2")
        dataset = DataSet([feat1, feat2])

        self.assertEqual(num, dataset.count)
        self.assertEqual(num, dataset.count)
        self.assertTrue(numpy.array_equal(feat1.values, dataset.features["feat1"].values))
        self.assertTrue(numpy.array_equal(feat2.values, dataset.features["feat2"].values))
        self.assertTrue(numpy.array_equal(dataset["feat1", :num], feat1[:num]))
        self.assertTrue(numpy.array_equal(dataset["feat2", :num], feat2[:num]))

        arr1 = dataset[:, :num]
        arr2 = [feat.values[:num] for feat in (feat1, feat2)]
        for col1, col2 in zip(arr1, arr2):
            self.assertTrue(numpy.array_equal(col1, col2))

    def test_dataset_mismatch_len(self):
        num = SAMPLE_SIZE_SMALL
        feat1 = Feature(generate_array_floats(n=num))
        feat2 = Feature(generate_array_floats(n=num * 2))
        self.assertRaises(AssertionError, lambda: DataSet([feat1, feat2]))

    def test_dataset_load_csv(self):
        cwd = os.path.dirname(os.path.realpath(__file__))
        dataset = DataSet.from_csv(
            Path(cwd) / ".." / "bananas" / "sampledata" / "dummy" / "train.csv", random_seed=0
        )
        sampleX, sampleY = dataset.input_fn(batch_size=10)
        self.assertListEqual(sampleX, [[1] * 10] * 3)
        self.assertListEqual(sampleY, ["a"] * 10)

    def test_dataset_custom_loader(self):
        num = SAMPLE_SIZE_SMALL
        arr = generate_array_floats(n=num)

        class MyCustomDataLoader(object):
            def __len__(self):
                return len(arr)

            def __getitem__(self, idx):
                return arr[idx]

        dataset1 = DataSet([Feature(arr)], random_seed=0)
        dataset2 = DataSet([Feature(MyCustomDataLoader())], random_seed=0)
        self.assertTrue(numpy.array_equal(dataset1.input_fn(), dataset2.input_fn()))

    # TODO: test with target, test input_fn()


main()
