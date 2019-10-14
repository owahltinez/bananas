''' Test Preprocessing Module '''

from bananas.changemap.changemap import ChangeMap
from bananas.sampledata.local import load_bike, load_boston, load_california, load_titanic
from bananas.sampledata.synthetic import new_line, new_labels
from bananas.utils.arrays import unique, shape_of_array
from bananas.utils.constants import DTYPE_UINT8

from bananas.preprocessing.encoding_strategy import EncodingStrategy
from bananas.preprocessing.normalization_strategy import NormalizationStrategy
from bananas.preprocessing.standard import StandardPreprocessor

from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):

    def test_preprocessor_normalization(self):
        dataset = new_line(random_seed=0)
        target_preprocessor = StandardPreprocessor(continuous=[0], threshold=-1)
        feature_preprocessor = StandardPreprocessor(continuous=[0], threshold=-1)

        num_batches = 8
        for _ in range(num_batches):
            X, y = dataset.input_fn()

            # Inputs are all 1D
            self.assertEqual(1, len(shape_of_array(X)))
            self.assertEqual(1, len(shape_of_array(y)))

            # Scale up to be able to put pressure on normalization
            X = (X - .5) * 100
            y = (y - .5) * 100
            self.assertGreaterEqual(X.max() - X.min(), 1)
            self.assertGreaterEqual(y.max() - y.min(), 1)

            X_ = feature_preprocessor.fit(X).transform(X)
            self.assertEqual(1, X_.ndim)
            self.assertEqual(round(X_.mean()), 0, X_.mean())
            self.assertNotAlmostEqual(X_.min(), 0)
            self.assertNotAlmostEqual(X_.max(), 0)

            y_ = target_preprocessor.fit(y).transform(y)
            self.assertEqual(round(y_.mean()), 0, y_.mean())
            self.assertNotAlmostEqual(y_.min(), 0)
            self.assertNotAlmostEqual(y_.max(), 0)

    def test_preprocessor_encode_ordinal(self):
        dataset = new_labels(random_seed=0)
        preprocessor = StandardPreprocessor(
            categorical=[0], normalization=None, encoding=EncodingStrategy.ORDINAL, threshold=-1)

        num_batches = 10
        for _ in range(num_batches):
            X, y = dataset.input_fn()

            # Inputs are all 1D
            self.assertEqual(1, len(shape_of_array(X)))
            self.assertEqual(1, len(shape_of_array(y)))

            # Feed labels to preprocessor
            y_ = preprocessor.fit(y).transform(y)
            self.assertEqual(1, y_.ndim)
            self.assertEqual(DTYPE_UINT8[0], y_.dtype)

            # Reverse transformation to get back original data
            y_ = preprocessor.inverse_transform(y_)
            self.assertEqual(1, y_.ndim)
            self.assertListEqual(y.tolist(), y_.tolist())

    def test_preprocessor_encode_drop(self):
        dataset = new_labels(random_seed=0)

        num_columns = 10
        categorical = list(range(num_columns // 2))
        preprocessor = StandardPreprocessor(
            categorical=categorical, normalization=None, encoding=EncodingStrategy.DROP,
            threshold=-1)

        num_batches = 10
        for _ in range(num_batches):
            X, y = dataset.input_fn()

            # Inputs are all 1D
            self.assertEqual(1, len(shape_of_array(X)))
            self.assertEqual(1, len(shape_of_array(y)))

            # Feed labels to preprocessor
            labels_2d = [y] * num_columns
            y_ = preprocessor.fit(labels_2d).transform(labels_2d)
            self.assertEqual(2, y_.ndim)
            self.assertEqual(num_columns - len(categorical), y_.shape[0])

            # Reverse transformation to get back original data should fail
            self.assertRaises(NotImplementedError, lambda: preprocessor.inverse_transform(y_))

    def test_preprocessor_encode_onehot(self):
        dataset = new_labels(random_seed=0)
        preprocessor = StandardPreprocessor(
            categorical=[0], normalization=None, encoding=EncodingStrategy.ONEHOT, threshold=-1)

        num_batches = 10
        for _ in range(num_batches):
            X, y = dataset.input_fn()

            # Inputs are all 1D
            self.assertEqual(1, len(shape_of_array(X)))
            self.assertEqual(1, len(shape_of_array(y)))

            # Feed labels to preprocessor
            y_ = preprocessor.fit(y).transform(y)
            self.assertEqual(2, len(shape_of_array(y_)))
            self.assertEqual(len(unique(y)), shape_of_array(y_)[0])
            self.assertEqual(DTYPE_UINT8[0], y_.dtype)

            # Reverse transformation to get back original data
            y_ = preprocessor.inverse_transform(y_)
            self.assertEqual(1, len(shape_of_array(y_)))
            self.assertListEqual(y.tolist(), y_.tolist())

        # TODO: test datasets with categorical columns

    def test_preprocessor_datasets(self):
        for dataset_loader in [load_bike, load_boston, load_california, load_titanic]:
            dataset, _ = dataset_loader(random_seed=0)
            preprocessor = StandardPreprocessor(
                categorical=dataset.categorical, continuous=dataset.continuous)

            output_len = [None]
            def cb_shape_change(change_map: ChangeMap):
                output_len[0] = change_map.output_len
            preprocessor.add_output_shape_changed_callback(cb_shape_change)

            for _ in range(100):
                X, _ = dataset.input_fn()
                try:
                    X_ = preprocessor.fit(X).transform(X)
                    self.assertEqual(output_len[0], len(X_), dataset.name)
                except Exception as ex:
                    print('Preprocessing dataset %s failed' % dataset.name)
                    raise ex

main()
