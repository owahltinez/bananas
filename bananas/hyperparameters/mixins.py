''' Hyper Parameters Module '''

from ..dataset.dataset import DataSet

class HyperParametersMixin(object):
    ''' Mixin that adds a `hyperparameters()` function to estimators. '''

    @staticmethod
    def hyperparameters(dataset: DataSet):
        '''
        Potential values for each one of the parameters.

        Parameters
        ----------
        dataset : DataSet
            TODO
        '''
        return {}
