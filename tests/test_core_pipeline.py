''' Test Pipeline Module '''

import numpy
from bananas.core.learner import Learner
from bananas.core.pipeline import Pipeline, PipelineStep
from bananas.dataset.dataset import DataSet
from bananas.testing.dummy import DummyRegressor, DummyTransformer
from bananas.testing.learners import test_learner
from bananas.testing.generators import \
    generate_array_booleans, generate_array_chars, generate_array_floats, generate_array_ints, \
    generate_array_int_floats, generate_array_uints, generate_array_nones, generate_array_strings, \
    generate_images, generate_onehot_matrix, generate_array_infinities

from .test_profiling import ProfilingTestCase, main


# pylint: disable=missing-docstring
class TestUtils(ProfilingTestCase):

    def test_create_pipeline(self):
        pipeline = Pipeline([PipelineStep(name='transformer', learner=DummyTransformer)])
        self.assertTrue(pipeline)

    def test_pipeline_builtin(self):
        steps1 = [PipelineStep(name='transformer', learner=DummyTransformer)]
        steps2 = [PipelineStep(name='estimator', learner=DummyRegressor)]
        steps3 = [
            PipelineStep(name='transformer', learner=DummyTransformer),
            PipelineStep(name='estimator', learner=DummyRegressor)
            ]
        for steps in (steps1, steps2, steps3):
            self.assertTrue(test_learner(Pipeline, steps=steps))

    def test_fit_pipeline(self):
        pipeline1 = Pipeline([PipelineStep(name='transformer', learner=DummyTransformer)])
        pipeline2 = Pipeline([PipelineStep(name='estimator', learner=DummyRegressor)])
        pipeline3 = Pipeline([
            PipelineStep(name='transformer', learner=DummyTransformer),
            PipelineStep(name='estimator', learner=DummyRegressor)
            ])
        data, target = numpy.random.random(10), numpy.ones(10)

        for pipeline in [pipeline1, pipeline2, pipeline3]:
            pipeline_ = pipeline.fit(data, y=target)
            self.assertEqual(pipeline, pipeline_)

    def test_fit_transform_pipeline(self):
        pipeline = Pipeline([PipelineStep(name='transformer', learner=DummyTransformer)])
        data1 = numpy.random.random(10)

        data2 = pipeline.fit(data1).transform(data1)
        self.assertTrue(numpy.array_equal(data1, data2))

    def test_fit_predict_pipeline(self):
        pipeline = Pipeline([PipelineStep(name='estimator', learner=DummyRegressor)])
        data1, target1 = numpy.random.random(10), numpy.ones(10)

        target2 = pipeline.fit(data1, target1).predict(data1)
        self.assertTrue(numpy.array_equal(target1, target2))

    def test_inverse_transform_pipeline(self):
        pipeline = Pipeline([PipelineStep(name='transformer', learner=DummyTransformer)])
        data1 = numpy.random.random(10)

        # Calling inverse transform without fitting first should fail
        self.assertRaises(AssertionError, lambda: pipeline.inverse_transform(data1))

        data2 = pipeline.fit(data1).transform(data1)
        data3 = pipeline.inverse_transform(data1)
        self.assertTrue(numpy.array_equal(data1, data2))
        self.assertTrue(numpy.array_equal(data1, data3))


main()
