''' Test Utils Module '''

import numpy
from bananas.utils.arrays import check_array, concat_arrays, shape_of_array, transpose_array
from bananas.testing.generators import *

from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):

    def test_check_array_valid(self):
        valid_generators = [
            generate_array_booleans,
            generate_array_chars,
            generate_array_floats,
            generate_array_ints,
            generate_array_strings,
            generate_array_uints]
        for gen in valid_generators:
            arr = gen()
            check_array(arr)

    def test_check_array_invalid(self):
        invalid_generators = [
            generate_array_infinities,
            generate_array_nones]
        for gen in invalid_generators:
            self.assertRaises(ValueError, lambda: check_array(gen()))

    def test_get_shape_of_list_1d(self):
        shape = (10,)
        arr = list(range(shape[0]))
        self.assertEqual(shape, shape_of_array(arr))

    def test_get_shape_of_list_2d(self):
        shape = (10, 2)
        arr = [list(range(shape[1])) for _ in range(shape[0])]
        self.assertEqual(shape, shape_of_array(arr))

    def test_get_shape_of_ndarray(self):
        for shape in [(10,), (10, 10), (10, 10, 10), (20, 10, 10)]:
            arr = numpy.empty(shape)
            self.assertEqual(shape, shape_of_array(arr))

    def test_get_shape_of_objects_ndarray(self):
        shape = (10, 2)
        arr = numpy.empty((shape[0]), dtype=object)
        for i in range(shape[0]):
            arr[i] = numpy.array(list(range(shape[1])))
        self.assertEqual(shape, shape_of_array(arr))
        self.assertNotEqual(shape, arr.shape)

    def test_concat_arrays(self):
        array_count = 4
        sample_size = 128

        # 1D arrays
        array_list = [generate_array_ints(n=sample_size) for _ in range(array_count)]
        array_concat = concat_arrays(*array_list)
        self.assertEqual(shape_of_array(array_list[0])[1:], shape_of_array(array_concat)[1:])
        self.assertEqual(len(array_concat), sample_size * array_count)

        # 2D arrays (2 columns)
        array_list = [
            generate_array_ints(n=sample_size * 2).reshape(-1, 2) for _ in range(array_count)]
        array_concat = concat_arrays(*array_list)
        self.assertEqual(shape_of_array(array_list[0])[1:], shape_of_array(array_concat)[1:])
        self.assertEqual(len(array_concat), sample_size * array_count)

        # N-D arrays (array of vectors)
        array_list = generate_onehot_matrix(n=sample_size, ndim=array_count)
        array_concat = concat_arrays(*array_list)
        self.assertEqual(shape_of_array(array_list[0])[1:], shape_of_array(array_concat)[1:])
        self.assertEqual(len(array_concat), sample_size * array_count)

        # N-D arrays (array of images)
        array_list = [generate_images(n=sample_size) for _ in range(array_count)]
        array_concat = concat_arrays(*array_list)
        self.assertEqual(shape_of_array(array_list[0])[1:], shape_of_array(array_concat)[1:])
        self.assertEqual(len(array_concat), sample_size * array_count)

    def test_transpose_array(self):
        array_1d = [1, 2, 3, 4]
        array_expected = list(array_1d)
        array_transposed = transpose_array(array_1d)
        self.assertListEqual(array_expected, array_transposed)

        array_2d = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        array_expected = [[1] * 4, [2] * 4, [3] * 4, [4] * 4]
        array_transposed = transpose_array(array_2d)
        self.assertListEqual(array_expected, array_transposed)

        array_tuples = [(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)]
        array_expected = [[1] * 4, [2] * 4, [3] * 4, [4] * 4]
        array_transposed = transpose_array(array_tuples)
        self.assertListEqual(array_expected, array_transposed)


main()
