from typing import Iterable

from ..core.pipeline import Pipeline, PipelineStep
from ..dataset.dataset import DataSet
from ..transformers.drop import FeatureDrop
from ..transformers.scalers import MinMaxScaler, StandardScaler
from ..transformers.encoders import LabelEncoder, OneHotEncoder
from ..transformers.threshold import VarianceThreshold
from ..utils.constants import ONEHOT_MAX_CLASSES
from .preprocessor import _Preprocessor
from .encoding_strategy import EncodingStrategy
from .normalization_strategy import NormalizationStrategy


class StandardPreprocessor(_Preprocessor):
    '''
    Preprocessor consisting of a [Pipeline](../../core/pipeline.html#bananas.core.pipeline.Pipeline)
    that performs automatic encoding of categorical features, normalization of continuous features
    and thresholding based on sampled feature variance.
    '''

    def __init__(self, categorical: dict = None, continuous: dict = None,
                 encoding: EncodingStrategy = EncodingStrategy.AUTO,
                 normalization: NormalizationStrategy = NormalizationStrategy.AUTO,
                 threshold: float = 1E-2, verbose: bool = False):
        '''
        Parameters
        ----------
        categorical : dict
            TODO
        continuous : dict
            TODO
        encoding : EncodingStrategy
            TODO
        normalization : NormalizationStrategy
            TODO
        threshold : float
            TODO
        verbose : bool
            TODO
        '''

        # Initialize parent's pipeline, which will call our _init_steps()
        super().__init__(categorical=categorical, continuous=continuous,
                         normalization=normalization, encoding=encoding, threshold=threshold,
                         verbose=verbose)

    def _init_steps(self, encoding: EncodingStrategy = None,
                    normalization: NormalizationStrategy = None, threshold: float = None):

        # Initialize steps
        steps = []
        classes = self.categorical_

        # Normalize the data
        if normalization is not None and self.continuous_:
            if not isinstance(normalization, NormalizationStrategy):
                raise ValueError('Unknown normalizer requested: %s. Expected one of: %r' %
                                 (normalization, NormalizationStrategy))
            if normalization == NormalizationStrategy.AUTO:
                normalization = NormalizationStrategy.STANDARD  # Default to standard scaler

            if normalization == NormalizationStrategy.MINMAX:
                normalizer, args = MinMaxScaler, {
                    'columns': self.continuous_, 'verbose': self.verbose}
            if normalization == NormalizationStrategy.STANDARD:
                normalizer, args = StandardScaler, {
                    'columns': self.continuous_, 'verbose': self.verbose}

            steps.append(PipelineStep(name='normalize', learner=normalizer, kwargs=args))
            self.print('Initialized normalizer:', normalizer)

        # Purge the features that do not add any information by implementing a variance threshold
        if threshold is not None and threshold >= 0.:
            # TODO: Fix categorical columns
            thresholder, args = VarianceThreshold, {
                'threshold': threshold, 'columns': self.continuous_, 'verbose': self.verbose}

            steps.append(PipelineStep(name='thresholder', learner=thresholder, kwargs=args))
            self.print('Initialized thresholder:', thresholder)

        # Encode labels into numerical features
        # TODO: large one-hot encodings should be reduced into an embedding
        if encoding is not None and classes:
            if not isinstance(encoding, EncodingStrategy):
                raise ValueError('Unknown encoder requested: %s. Expected one of: %r' %
                                 (encoding, EncodingStrategy))
            if encoding == EncodingStrategy.ORDINAL:
                encoder, args = LabelEncoder, {'columns': classes, 'verbose': self.verbose}
            if encoding == EncodingStrategy.ONEHOT:
                encoder, args = OneHotEncoder, {'columns': classes, 'verbose': self.verbose}
            if encoding == EncodingStrategy.DROP:
                encoder, args = FeatureDrop, {'columns': classes, 'verbose': self.verbose}

            if encoding == EncodingStrategy.AUTO:
                # Use one-hot encoder for categorical variables with known classes, drop otherwise
                drop_columns = [
                    k for k, v in classes.items() if v and len(v) > ONEHOT_MAX_CLASSES]
                encode_columns = {
                    k: v for k, v in classes.items() if v and len(v) <= ONEHOT_MAX_CLASSES}
                encoder, args = Pipeline, {
                    'steps': [
                        PipelineStep(name='drop', learner=FeatureDrop,
                                     kwargs={'columns': drop_columns, 'verbose': self.verbose}),
                        PipelineStep(name='onehot', learner=OneHotEncoder,
                                     kwargs={'columns': encode_columns, 'verbose': self.verbose})]}

                self.print('Auto encoder dropping columns:', drop_columns)
                self.print('Auto encoder onehot-encoding columns:', list(encode_columns.keys()))

            steps.append(PipelineStep(name='encoder', learner=encoder, kwargs=args))
            self.print('Initialized %s encoder:' % encoding, encoder)

        return steps

    @staticmethod
    def hyperparameters(dataset: DataSet):
        return {
            'normalization': [NormalizationStrategy.AUTO, None],
            'threshold': [1E-2, 1E-1, 0.],
            'encoding': [EncodingStrategy.AUTO, None]}

# TODO: ImagePreprocessor, TextPreprocessor
