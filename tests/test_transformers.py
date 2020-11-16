""" Test Transformers Module """

import itertools
import numpy
import numpy.random

from bananas.changemap.changemap import ChangeMap
from bananas.core.pipeline import Pipeline, PipelineStep
from bananas.testing.learners import test_learner
from bananas.testing.generators import (
    generate_array_floats,
    generate_array_uints,
)
from bananas.testing.dummy import DummyTransformer
from bananas.transformers.drop import FeatureDrop
from bananas.transformers.encoders import LabelEncoder, OneHotEncoder
from bananas.transformers.scalers import MinMaxScaler
from bananas.transformers.running_stats import RunningStats
from bananas.transformers.threshold import VarianceThreshold

from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):
    def test_transformer_builtin(self):
        for transformer in [FeatureDrop, VarianceThreshold]:
            self.assertTrue(test_learner(transformer, columns=[]))

    def test_stats_values(self):
        arr = generate_array_floats(random_seed=0)
        stats = RunningStats(columns=[0]).fit([arr])
        self.assertEqual(numpy.max(arr), stats.max_[0])
        self.assertEqual(numpy.min(arr), stats.min_[0])
        self.assertAlmostEqual(numpy.mean(arr), stats.mean_[0], 3)
        self.assertAlmostEqual(numpy.var(arr), stats.variance_[0], 3)
        # TODO: test ints, uints, images...

    def test_feature_drop(self):
        n = 100
        data_0 = [numpy.ones(n) * -1]
        data_1 = [numpy.random.random(n) for _ in range(4)]
        data_concat = data_0 + data_1
        dropper = FeatureDrop(columns=[0])

        # Setup output change callback
        callback = [False]

        def cb(change_map: ChangeMap):
            callback[0] = True

        dropper.add_output_shape_changed_callback(cb)

        output = dropper.fit(data_concat).transform(data_concat)
        self.assertTrue(callback)
        self.assertEqual(4, len(output))
        self.assertEqual(n, len(output[0]))
        self.assertGreaterEqual(output[0].min(), 0)

        # Send input change event
        change_map = ChangeMap(len(output), idx_add=[0], idx_del=[0])
        dropper.on_input_shape_changed(change_map)

    def test_feature_drop_twice_different(self):
        n = 100
        data = [numpy.ones(n) * i for i in range(4)]
        pipeline = Pipeline(
            [
                PipelineStep(name=1, learner=FeatureDrop, kwargs={"columns": [0]}),
                PipelineStep(name=2, learner=FeatureDrop, kwargs={"columns": [1]}),
            ]
        )

        output = pipeline.fit(data).transform(data)
        self.assertEqual(2, len(output))
        for i in range(2):
            self.assertListEqual((numpy.ones(n) * i + 2).tolist(), output[i].tolist())

        # Send input change event
        change_map = ChangeMap(len(output), idx_add=[0], idx_del=[0])
        pipeline.on_input_shape_changed(change_map)
        output = pipeline.fit(data).transform(data)

    def test_feature_drop_twice_same(self):
        n = 100
        data = [numpy.ones(n) * i for i in range(4)]
        pipeline = Pipeline(
            [
                PipelineStep(name=1, learner=FeatureDrop, kwargs={"columns": [0]}),
                PipelineStep(name=2, learner=FeatureDrop, kwargs={"columns": [0]}),
            ]
        )

        # The expectation is that dropping a column that no longer exists is a no-op
        output = pipeline.fit(data).transform(data)
        self.assertEqual(3, len(output))
        for i in range(3):
            self.assertListEqual((numpy.ones(n) * i + 1).tolist(), output[i].tolist())

    def test_feature_drop_thrice_different(self):
        n = 100
        data = [numpy.ones(n) * i for i in range(4)]
        pipeline = Pipeline(
            [
                PipelineStep(name=1, learner=FeatureDrop, kwargs={"columns": [0]}),
                PipelineStep(name=2, learner=FeatureDrop, kwargs={"columns": [1]}),
                PipelineStep(name=3, learner=FeatureDrop, kwargs={"columns": [2]}),
            ]
        )

        output = pipeline.fit(data).transform(data)
        self.assertEqual(1, len(output))
        for i in range(1):
            self.assertListEqual((numpy.ones(n) * i + 3).tolist(), output[i].tolist())

    def test_feature_drop_thrice_same(self):
        n = 100
        data = [numpy.ones(n) * i for i in range(4)]
        pipeline = Pipeline(
            [
                PipelineStep(name=1, learner=FeatureDrop, kwargs={"columns": [0]}),
                PipelineStep(name=2, learner=FeatureDrop, kwargs={"columns": [0]}),
                PipelineStep(name=3, learner=FeatureDrop, kwargs={"columns": [0]}),
            ]
        )

        # The expectation is that dropping a column that no longer exists is a no-op
        output = pipeline.fit(data).transform(data)
        self.assertEqual(3, len(output))
        for i in range(3):
            self.assertListEqual((numpy.ones(n) * i + 1).tolist(), output[i].tolist())

    def test_feature_drop_following_scaler(self):
        n = 100
        data_0 = [numpy.ones(n) * -1]
        data_1 = [numpy.random.random(n) * 10 for _ in range(4)]
        data_concat = data_0 + data_1

        pipeline = Pipeline(
            [
                PipelineStep(name=1, learner=FeatureDrop, kwargs={"columns": [0]}),
                PipelineStep(name=2, learner=MinMaxScaler, kwargs={"columns": [1, 2, 3]}),
            ]
        )

        output = pipeline.fit(data_concat).transform(data_concat)
        self.assertEqual(4, len(output))
        self.assertGreaterEqual(output[-1].max(), 1)
        for i, col in enumerate(output[:-1]):
            self.assertLessEqual(col.max(), 1, "Column %d" % i)

    def test_feature_drop_following_encoder(self):
        n = 100
        nums = list(range(10))
        data_0 = [numpy.random.random(n) * 10 for _ in range(4)]
        data_1 = [generate_array_uints(n=n, max_int=10, random_seed=0)]
        data_concat = data_0 + data_1

        pipeline = Pipeline(
            [
                PipelineStep(name=1, learner=FeatureDrop, kwargs={"columns": [0]}),
                PipelineStep(name=2, learner=LabelEncoder, kwargs={"columns": {0: nums, 4: nums}}),
            ]
        )

        output = pipeline.fit(data_concat).transform(data_concat)
        self.assertEqual(4, len(output))
        self.assertGreaterEqual(output[-1].max(), 1)

    def test_feature_drop_following_encoder_onehot(self):
        n = 100
        nums = list(range(10))
        data_0 = [numpy.random.random(n) * 10]
        data_1 = [generate_array_uints(n=n, max_int=10, random_seed=0)]
        data_concat = data_0 + data_1 + data_0 + data_1

        pipeline = Pipeline(
            [
                PipelineStep(name=1, learner=FeatureDrop, kwargs={"columns": [0]}),
                PipelineStep(name=2, learner=OneHotEncoder, kwargs={"columns": {0: nums, 3: nums}}),
            ]
        )

        output = pipeline.fit(data_concat).transform(data_concat)
        self.assertEqual(12, len(output))
        for col in output[:2]:
            self.assertLessEqual(output[-1].max(), 1)
        for col in output[2:]:
            self.assertGreaterEqual(output[-1].max(), 1)

    def test_encoder_onehot_following_feature_drop(self):
        n = 100
        nums = list(range(10))
        data_0 = [generate_array_uints(n=n, max_int=10, random_seed=0)]
        data_1 = [numpy.random.random(n) for _ in range(4)]
        data_concat = data_0 + data_1

        pipeline = Pipeline(
            [
                PipelineStep(name=1, learner=FeatureDrop, kwargs={"columns": [1, 2, 3]}),
                PipelineStep(name=2, learner=OneHotEncoder, kwargs={"columns": {0: nums}}),
            ]
        )

        output = pipeline.fit(data_concat).transform(data_concat)
        self.assertEqual(len(nums) + 1, len(output))
        for i, col in enumerate(output[:-1]):
            self.assertGreaterEqual(col.max(), 1, i)
        self.assertLessEqual(output[-1].max(), 1)

    def test_feature_drop_following_scaler_then_encoder(self):
        n = 100
        nums = list(range(10))
        data_0 = [numpy.ones(n) * -1]
        data_1 = [numpy.random.random(n) * 10 for _ in range(3)]
        data_2 = [generate_array_uints(n=n, max_int=10, random_seed=0)]
        data_concat = data_0 + data_1 + data_2

        pipeline = Pipeline(
            [
                PipelineStep(name=1, learner=FeatureDrop, kwargs={"columns": [0]}),
                PipelineStep(name=2, learner=MinMaxScaler, kwargs={"columns": [1, 2, 3]}),
                PipelineStep(name=3, learner=LabelEncoder, kwargs={"columns": {4: nums}}),
            ]
        )

        output = pipeline.fit(data_concat).transform(data_concat)
        self.assertEqual(4, len(output))
        self.assertGreaterEqual(output[0].max(), 1)
        for col in output[1:-1]:
            self.assertLessEqual(col.max(), 1)

    def test_chaining_feature_drop_encoder_onehot_scaler_variance_threshold(self):
        n = 100
        nums = list(range(10))
        data_0 = [numpy.ones(n) * -1]
        data_1 = [numpy.random.random(n) * 10 for _ in range(3)]
        data_2 = [generate_array_uints(n=n, max_int=10, random_seed=0)]
        data_concat = data_0 + data_1 + data_2

        all_steps = (
            PipelineStep(learner=DummyTransformer, kwargs={}),
            PipelineStep(learner=FeatureDrop, kwargs={"columns": [0]}),
            PipelineStep(learner=MinMaxScaler, kwargs={"columns": [1, 2, 3]}),
            PipelineStep(learner=LabelEncoder, kwargs={"columns": {0: [], 4: nums}}),
            PipelineStep(learner=VarianceThreshold, kwargs={"columns": [1, 2, 3]}),
        )

        transformers = itertools.combinations_with_replacement(all_steps, len(all_steps))
        for steps in transformers:
            pipeline = Pipeline(steps)
            try:
                pipeline.fit(data_concat).transform(data_concat)
            except Exception as ex:
                print("Pipeline failed")
                for step in pipeline.steps:
                    print(step, getattr(step[1], "classes_", None))
                raise (ex)

    # TODO: test one-hot encoder after an input change that affects one of its columns


main()
