''' Test Transformers Module '''

import itertools
import numpy

from bananas.changemap.changemap import ChangeMap
from bananas.testing.learners import test_learner
from bananas.testing.generators import \
    generate_array_booleans, generate_array_chars, generate_array_floats, generate_array_ints, \
    generate_array_int_floats, generate_array_uints, generate_array_nones, generate_array_strings, \
    generate_images, generate_onehot_matrix, generate_array_infinities
from bananas.testing.dummy import DummyTransformer
from bananas.transformers.encoders import LabelEncoder, OneHotEncoder
from bananas.transformers.threshold import VarianceThreshold

from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):

    def test_transformer_builtin(self):
        for transformer in [LabelEncoder, OneHotEncoder]:
            self.assertTrue(test_learner(transformer, columns=[]))

    def test_encoder_label(self):
        nums = list(range(10))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0)

        encoder = LabelEncoder(columns={0: nums})
        output = encoder.fit(ints).transform(ints)
        self.assertEqual(len(output.shape), 1)
        self.assertEqual(ints.shape, output.shape)
        self.assertTrue(numpy.array_equal(ints, output))

    def test_encoder_label_1D(self):
        nums = list(range(10))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0).reshape(1, -1)

        encoder = LabelEncoder(columns={0: nums})
        output = encoder.fit(ints).transform(ints)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].ndim, 1)
        self.assertTrue(numpy.array_equal(ints, output))

    def test_encoder_label_1D_offset(self):
        nums = list(range(1, 11))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0).reshape(1, -1)
        ints_test = ints.copy() + 1

        encoder = LabelEncoder(columns={0: nums})
        output = encoder.fit(ints_test).transform(ints_test)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].ndim, 1)
        self.assertTrue(numpy.array_equal(ints, output))

    def test_encoder_label_strings(self):
        classes = list('abcdefghij')
        vocab = {i: c for i, c in enumerate(classes)}
        lmap = lambda func, *iters: list(map(func, *iters))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0)
        ints_test = numpy.asarray(lmap(vocab.get, ints), dtype='object')

        encoder = LabelEncoder(columns={0: classes})
        output = encoder.fit(ints_test).transform(ints_test)
        output_inv = encoder.inverse_transform(output)
        self.assertEqual(len(output.shape), 1)
        self.assertEqual(ints_test.shape, output.shape)
        # self.assertTrue(numpy.array_equal(ints, output))
        self.assertTrue(numpy.array_equal(ints_test, output_inv))

    def test_encoder_label_strings_guess_classes(self):
        classes = list('abcdefghij')
        vocab = {i: c for i, c in enumerate(classes)}
        lmap = lambda func, *iters: list(map(func, *iters))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0)
        chars = numpy.asarray(lmap(vocab.get, ints), dtype='object')

        encoder = LabelEncoder(columns={0: []})
        output = encoder.fit(chars).transform(chars)
        output_inv = encoder.inverse_transform(output)
        self.assertEqual(len(output.shape), 1)
        self.assertEqual(chars.shape, output.shape)
        # self.assertTrue(numpy.array_equal(ints, output))
        self.assertTrue(numpy.array_equal(chars, output_inv))

    def test_encoder_label_2D(self):
        nums = list(range(10))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0).reshape(2, -1)
        encoder = LabelEncoder(columns={0: nums, 1: nums})
        output = encoder.fit(ints).transform(ints)
        self.assertEqual(len(output), 2)
        self.assertEqual(len(output[0]), len(ints[0]))
        self.assertTrue(numpy.array_equal(ints, output))

    def test_encoder_label_2D_strings(self):
        classes = list('abcdefghij')
        vocab = {i: c for i, c in enumerate(classes)}
        lmap = lambda func, *iters: list(map(func, *iters))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0)
        chars = numpy.asarray(lmap(vocab.get, ints), dtype='object')
        ints_2d = ints.reshape(2, -1)
        chars_2d = chars.reshape(2, -1)

        encoder = LabelEncoder(columns={0: classes, 1: classes})
        output = encoder.fit(chars_2d).transform(chars_2d)
        output_inv = encoder.inverse_transform(output)
        self.assertEqual(len(output), 2)
        # self.assertTrue(numpy.array_equal(ints_2d, output))
        self.assertTrue(numpy.array_equal(chars_2d, output_inv))

    def test_encoder_label_2D_strings_guess_classes(self):
        classes = list('abcdefghij')
        vocab = {i: c for i, c in enumerate(classes)}
        lmap = lambda func, *iters: list(map(func, *iters))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0)
        chars = numpy.asarray(lmap(vocab.get, ints), dtype='object')
        ints_2d = ints.reshape(2, -1)
        chars_2d = chars.reshape(2, -1)

        encoder = LabelEncoder(columns={0: [], 1: []})
        output = encoder.fit(chars_2d).transform(chars_2d)
        output_inv = encoder.inverse_transform(output)
        self.assertEqual(len(output), 2)
        # self.assertTrue(numpy.array_equal(ints_2d, output))
        self.assertTrue(numpy.array_equal(chars_2d, output_inv))

    def test_encoder_label_2D_mixed(self):
        classes = list('abcdefghij')
        vocab = {i: c for i, c in enumerate(classes)}
        lmap = lambda func, *iters: list(map(func, *iters))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0)
        chars = numpy.asarray(lmap(vocab.get, ints), dtype='object')
        ints_2d = ints.reshape(2, -1)
        chars_2d = chars.reshape(2, -1)

        encoder = LabelEncoder(columns={0: classes})
        output = encoder.fit(chars_2d).transform(chars_2d)
        output_inv = encoder.inverse_transform(output)
        self.assertEqual(len(output), 2)
        # self.assertTrue(numpy.array_equal(ints_2d[0], output[0]))
        # self.assertTrue(numpy.array_equal(chars_2d[1], output[1]))
        self.assertTrue(numpy.array_equal(chars_2d, output_inv))

    def test_encoder_label_1D_objects(self):
        nums = list(range(10))
        class CustomObject(object):
            def __init__(self, data): self.data = data
        ints = generate_array_uints(n=200, random_seed=0)
        lmap = lambda func, *iters: list(map(func, *iters))
        ints_test = numpy.asarray(lmap(CustomObject, ints), dtype='object')
        encoder = LabelEncoder(columns={0: nums})
        # NOTE: Only numerical and string types are supported by the encoder
        # Arbitrary types would require for us to pickle the object so it passes `check_array` test
        self.assertRaises(TypeError, lambda: encoder.fit(ints_test))

    def test_encoder_label_1D_set_classes(self):
        ints = generate_array_uints(n=200, max_int=10, random_seed=0)
        lmap = lambda func, *iters: list(map(func, *iters))
        chars = numpy.asarray(lmap(str, ints), dtype=str)

        # Create 100 placeholder classes, even though we will only see 10 different ones
        nums = list(map(str, range(100)))
        encoder = LabelEncoder(columns={0: nums})

        output = encoder.fit(chars[0:1]).transform(chars)
        output_inv = encoder.inverse_transform(output)
        # self.assertTrue(numpy.array_equal(ints, output))
        self.assertTrue(numpy.array_equal(chars, output_inv))

    # TODO: test mixed categorical and non-categorical columns with label encoding

    def test_threshold_variance_2D(self):
        n = 100
        data_0 = [numpy.zeros(n)]
        data_1 = [numpy.random.random(n) for _ in range(4)]
        data_concat = data_0 + data_1
        transformer_init = lambda: VarianceThreshold(columns=[i for i in range(len(data_concat))])
        thresholder = transformer_init()

        output = thresholder.fit(data_0).transform(data_0)
        self.assertEqual(0, len(output))
        thresholder = transformer_init()

        output = thresholder.fit(data_1).transform(data_1)
        self.assertEqual(4, len(output))
        self.assertEqual(n, len(output[0]))
        thresholder = transformer_init()

        output = thresholder.fit(data_concat).transform(data_concat)
        self.assertEqual(4, len(output))
        self.assertEqual(n, len(output[0]))

    def test_threshold_variance_onchange(self):
        counter = [0]
        def cb(change_map):
            counter[0] += 1

        # Callback fires after transforming zeros
        data = [numpy.zeros(10)]
        thresholder = VarianceThreshold(columns=[i for i in range(len(data))])
        thresholder.add_output_shape_changed_callback(cb)
        self.assertEqual(0, len(thresholder.fit(data).transform(data)))
        self.assertEqual(1, counter[0])

        # Callback does not fire after fitting random data
        data = [generate_array_floats(n=16, random_seed=0) for _ in range(4)]
        thresholder = VarianceThreshold(columns=[i for i in range(len(data))])
        thresholder.add_output_shape_changed_callback(cb)
        self.assertEqual(4, len(thresholder.fit(data).transform(data)))
        self.assertEqual(1, counter[0])
        thresholder = VarianceThreshold(columns=[i for i in range(len(data))])

        data_1 = [generate_array_floats(n=16, random_seed=0) for _ in range(4)]
        data_2 = [numpy.zeros(128, dtype=float)] + [
            generate_array_floats(n=128, random_seed=0) for _ in range(3)]
        thresholder = VarianceThreshold(columns=[i for i in range(len(data_1))])
        thresholder.add_output_shape_changed_callback(cb)
        self.assertEqual(4, len(thresholder.fit(data_1).transform(data_1)))
        for _ in range(10000): thresholder.fit(data_2)
        self.assertEqual(3, len(thresholder.transform(data_2)), thresholder.variance_)
        self.assertEqual(2, counter[0])

    def test_encoder_onehot(self):
        nums = list(range(10))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0)
        encoder = OneHotEncoder(columns={0: nums})

        # Setup output change callback
        callback = [False]
        def cb(change_map: ChangeMap): callback[0] = True
        encoder.add_output_shape_changed_callback(cb)

        output = encoder.fit(ints).transform(ints)
        self.assertEqual(len(output), 10)
        self.assertEqual(len(output[0]), len(ints))
        self.assertTrue(callback)

        output_inv = encoder.inverse_transform(output)
        self.assertTrue(numpy.array_equal(ints, output_inv))

        # Send input change event
        change_map = ChangeMap(len(output), idx_add=[0], idx_del=[0])
        encoder.on_input_shape_changed(change_map)

    def test_encoder_onehot_strings(self):
        classes = list('abcdefghij')
        vocab = {i: c for i, c in enumerate(classes)}
        ints = numpy.random.randint(0, 10, 200, dtype=numpy.uint8)
        chars = numpy.asarray(list(map(vocab.get, ints)), dtype='object')
        encoder = OneHotEncoder(columns={0: classes})

        output = encoder.fit(chars).transform(chars)
        self.assertEqual(len(output), 10)
        self.assertEqual(len(output[0]), len(chars))

        output_inv = encoder.inverse_transform(output)
        self.assertTrue(numpy.array_equal(chars, output_inv))

        # Send input change event
        change_map = ChangeMap(len(output), idx_add=[0], idx_del=[0])
        encoder.on_input_shape_changed(change_map)

    def test_encoder_onehot_1D(self):
        nums = list(range(10))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0).reshape(1, -1)
        encoder = OneHotEncoder(columns={0: nums})

        output = encoder.fit(ints).transform(ints)
        self.assertEqual(len(output), 10)
        self.assertEqual(len(output[0]), len(ints[0]))

        output_inv = encoder.inverse_transform(output)
        self.assertTrue(numpy.array_equal(ints, output_inv))

        # Send input change event
        change_map = ChangeMap(len(output), idx_add=[0], idx_del=[0])
        encoder.on_input_shape_changed(change_map)

    def test_encoder_onehot_1D_strings(self):
        classes = list('abcdefghij')
        vocab = {i: c for i, c in enumerate(classes)}
        ints = generate_array_uints(n=200, max_int=10, random_seed=0)
        chars = numpy.asarray(list(map(vocab.get, ints)), dtype='object').reshape(1, -1)
        encoder = OneHotEncoder(columns={0: classes})

        output = encoder.fit(chars).transform(chars)
        self.assertEqual(len(output), 10)
        self.assertEqual(len(output[0]), len(chars[0]))

        output_inv = encoder.inverse_transform(output)
        self.assertTrue(numpy.array_equal(chars, output_inv))

        # Send input change event
        change_map = ChangeMap(len(output), idx_add=[0], idx_del=[0])
        encoder.on_input_shape_changed(change_map)

    def test_encoder_onehot_2D(self):
        nums = list(range(10))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0).reshape(2, -1)
        encoder = OneHotEncoder(columns={0: nums, 1: nums})

        output = encoder.fit(ints).transform(ints)
        self.assertEqual(len(output), 20)
        self.assertEqual(len(output[0]), len(ints[0]))
        output_inv = encoder.inverse_transform(output)
        self.assertTrue(numpy.array_equal(ints, output_inv))

        # Send input change event
        change_map = ChangeMap(len(output), idx_add=[0], idx_del=[0])
        encoder.on_input_shape_changed(change_map)

    def test_encoder_onehot_2D_strings(self):
        classes = list('abcdefghij')
        vocab = {i: c for i, c in enumerate(classes)}
        ints = generate_array_uints(n=200, max_int=10, random_seed=0)
        chars = numpy.asarray(list(map(vocab.get, ints)), dtype='object').reshape(2, -1)
        encoder = OneHotEncoder(columns={0: classes, 1: classes})

        output = encoder.fit(chars).transform(chars)
        self.assertEqual(len(output), 20)
        self.assertEqual(len(output[0]), len(chars[0]))
        output_inv = encoder.inverse_transform(output)
        self.assertTrue(numpy.array_equal(chars, output_inv))

        # Send input change event
        change_map = ChangeMap(len(output), idx_add=[0], idx_del=[0])
        encoder.on_input_shape_changed(change_map)

    def test_encoder_onehot_2D_mixed(self):
        nums = list(range(10))
        ints = generate_array_uints(n=200, max_int=10, random_seed=0).reshape(2, -1)
        encoder = OneHotEncoder(columns={0: nums})

        output = encoder.fit(ints).transform(ints)
        self.assertEqual(len(output), 11)
        self.assertEqual(len(output[0]), len(ints[0]))
        self.assertTrue(numpy.array_equal(ints[1], output[-1]))
        output_inv = encoder.inverse_transform(output)
        self.assertTrue(numpy.array_equal(ints, output_inv), output_inv)

        # Send input change event
        change_map = ChangeMap(len(output), idx_add=[0], idx_del=[0])
        encoder.on_input_shape_changed(change_map)

    # TODO: test mixed categorical and non-categorical one-hot encoding

main()
