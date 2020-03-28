''' Test Transformers Module '''

import numpy

from bananas.testing.learners import test_learner
from bananas.testing.generators import \
    generate_array_booleans, generate_array_chars, generate_array_floats, generate_array_ints, \
    generate_array_int_floats, generate_array_uints, generate_array_nones, generate_array_strings, \
    generate_images, generate_onehot_matrix, generate_array_infinities
from bananas.transformers.scalers import MinMaxScaler, StandardScaler

from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):

    def test_transformer_builtin(self):
        for transformer in [
            MinMaxScaler, StandardScaler]:
            self.assertTrue(test_learner(transformer))

    def test_scaler_minmax(self):
        ones = numpy.ones(10)
        for i in range(1, 5):
            scaler = MinMaxScaler()
            data = numpy.concatenate([ones * i, ones * -i])

            # Verify that normalization of data works as expected
            norm = scaler.fit(data).transform(data)
            self.assertEqual(norm.min(), 0)
            self.assertEqual(norm.max(), 1)

    def test_scaler_minmax_inverse(self):
        ones = numpy.ones(10)
        for i in range(1, 5):
            scaler = MinMaxScaler()
            data = numpy.concatenate([ones * i, ones * -i])
            norm = scaler.fit(data).transform(data)
            self.assertEqual(norm.min(), 0)
            self.assertEqual(norm.max(), 1)

            norm_inv = scaler.inverse_transform(norm)
            self.assertListEqual(data.tolist(), norm_inv.tolist())

    def test_scaler_minmax_1D(self):
        ones = numpy.ones(10).reshape(1, -1)
        for i in range(1, 5):
            scaler = MinMaxScaler()
            data = numpy.hstack([ones * i, ones * -i])

            # Verify that normalization of data works as expected
            norm = scaler.fit(data).transform(data)
            self.assertEqual(norm[0].min(), 0)
            self.assertEqual(norm[0].max(), 1)

    def test_scaler_minmax_1D_batches(self):
        ones = numpy.ones(10).reshape(1, -1)
        for i in range(1, 5):
            scaler = MinMaxScaler()
            scaler = scaler.fit(ones * i)
            scaler = scaler.fit(ones * -i)

            # Verify that normalization of data works as expected
            norm = scaler.transform(numpy.hstack([ones * i, ones * -i]))
            self.assertEqual(norm[0].min(), 0)
            self.assertEqual(norm[0].max(), 1)

    def test_scaler_minmax_2D(self):
        ones = numpy.ones(20).reshape(2, -1)
        for i in range(1, 5):
            scaler = MinMaxScaler(columns=[0, 1])
            data = numpy.hstack([ones * i, ones * -i])

            # Verify that normalization of data works as expected
            norm = scaler.fit(data).transform(data)
            self.assertEqual(norm[0].min(), 0)
            self.assertEqual(norm[1].min(), 0)
            self.assertEqual(norm[0].max(), 1)
            self.assertEqual(norm[1].max(), 1)

    def test_scaler_minmax_2D_batches(self):
        ones = numpy.ones(20).reshape(2, -1)
        for i in range(1, 5):
            scaler = MinMaxScaler(columns=[0, 1])
            scaler = scaler.fit(ones * i)
            scaler = scaler.fit(ones * -i)

            # Verify that normalization of data works as expected
            norm = scaler.transform(numpy.hstack([ones * i, ones * -i]))
            self.assertEqual(norm[0].min(), 0)
            self.assertEqual(norm[1].min(), 0)
            self.assertEqual(norm[0].max(), 1)
            self.assertEqual(norm[1].max(), 1)

    def test_scaler_minmax_3D(self):
        # Single column with 10 samples of an 8x8 matrix
        ones = numpy.ones((10, 8, 8))
        for i in range(1, 5):
            scaler = MinMaxScaler()

            # Wrap the matrix into an array to delineate single column
            data = [numpy.vstack([ones * i, ones * -i])]

            # Verify that normalization of data works as expected
            norm = scaler.fit(data).transform(data)
            self.assertEqual(norm[0].min(), 0)
            self.assertEqual(norm[0].max(), 1)

    def test_scaler_minmax_3D_batches(self):
        # Single column with 10 samples of an 8x8 matrix
        ones = numpy.ones((10, 8, 8))
        for i in range(1, 5):
            scaler = MinMaxScaler()

            # Wrap the matrix into an array to delineate single column
            scaler = scaler.fit([ones * i])
            scaler = scaler.fit([ones * -i])

            # Verify that normalization of data works as expected
            norm = scaler.transform([numpy.vstack([ones * i, ones * -i])])
            self.assertAlmostEqual(norm[0].min(), 0)
            self.assertAlmostEqual(norm[0].max(), 1)

    def test_scaler_standard(self):
        ones = numpy.ones(10)
        for i in range(1, 3):
            scaler = StandardScaler()
            data = numpy.concatenate([ones * i, ones * -i])

            # Verify that normalization of data works as expected
            norm = scaler.fit(data).transform(data)
            self.assertEqual(norm.min(), -1)
            self.assertEqual(norm.max(), 1)

    def test_scaler_standard_inverse(self):
        ones = numpy.ones(10)
        for i in range(1, 5):
            scaler = StandardScaler()
            data = numpy.concatenate([ones * i, ones * -i])
            norm = scaler.fit(data).transform(data)
            self.assertEqual(norm.min(), -1)
            self.assertEqual(norm.max(), 1)

            # NOTE: comparisons are fuzzy since inverse depends on variance which is estimated
            norm_inv = scaler.inverse_transform(norm)
            for v1, v2 in zip(data.tolist(), norm_inv.tolist()):
                self.assertAlmostEqual(v1, v2)

    def test_scaler_standard_1D(self):
        ones = numpy.ones(10).reshape(1, -1)
        for i in range(1, 5):
            scaler = StandardScaler()
            data = numpy.hstack([ones * i, ones * -i])

            # Verify that normalization of data works as expected
            norm = scaler.fit(data).transform(data)
            self.assertEqual(norm[0].min(), -1)
            self.assertEqual(norm[0].max(), 1)

    def test_scaler_standard_1D_batches(self):
        ones = numpy.ones(10).reshape(1, -1)
        for i in range(1, 5):
            scaler = StandardScaler()
            scaler = scaler.fit(ones * i)
            scaler = scaler.fit(ones * -i)

            # Verify that normalization of data works as expected
            norm = scaler.transform(numpy.hstack([ones * i, ones * -i]))
            self.assertEqual(norm[0].min(), -1)
            self.assertEqual(norm[0].max(), 1)

    def test_scaler_standard_2D(self):
        ones = numpy.ones(20).reshape(2, -1)
        for i in range(1, 5):
            scaler = StandardScaler(columns=[0, 1])
            data = numpy.hstack([ones * i, ones * -i])

            # Verify that normalization of data works as expected
            norm = scaler.fit(data).transform(data)
            self.assertEqual(norm[0].min(), -1)
            self.assertEqual(norm[1].min(), -1)
            self.assertEqual(norm[0].max(), 1)
            self.assertEqual(norm[1].max(), 1)

    def test_scaler_standard_2D_batches(self):
        ones = numpy.ones(20).reshape(2, -1)
        for i in range(1, 5):
            scaler = StandardScaler(columns=[0, 1])
            scaler = scaler.fit(ones * i)
            scaler = scaler.fit(ones * -i)

            # Verify that normalization of data works as expected
            norm = scaler.transform(numpy.hstack([ones * i, ones * -i]))
            self.assertEqual(norm[0].min(), -1)
            self.assertEqual(norm[1].min(), -1)
            self.assertEqual(norm[0].max(), 1)
            self.assertEqual(norm[1].max(), 1)

    def test_scaler_standard_3D(self):
        # Single column with 10 samples of an 8x8 matrix
        ones = numpy.ones((10, 8, 8))
        for i in range(1, 5):
            scaler = StandardScaler()
            # Wrap the matrix into an array to delineate single column
            data = [numpy.vstack([ones * i, ones * -i])]

            # Verify that normalization of data works as expected
            norm = scaler.fit(data).transform(data)
            self.assertAlmostEqual(norm[0].min(), -1)
            self.assertAlmostEqual(norm[0].max(), 1)

    def test_scaler_standard_3D_batches(self):
        # Single column with 10 samples of an 8x8 matrix
        ones = numpy.ones((10, 8, 8))
        for i in range(1, 5):
            scaler = StandardScaler()

            # Wrap the matrix into an array to delineate single column
            scaler = scaler.fit([ones * i])
            scaler = scaler.fit([ones * -i])

            # Verify that normalization of data works as expected
            norm = scaler.transform([numpy.vstack([ones * i, ones * -i])])
            self.assertAlmostEqual(norm[0].min(), -1)
            self.assertAlmostEqual(norm[0].max(), 1)

main()
