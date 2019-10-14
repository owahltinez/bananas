from ..utils.arrays import take_elems

class _BaseSampler(object):
    ''' Interface to be used by all samplers '''
    def __init__(self, data, batch_size: int = 128, epochs: int = None, input_size: int = None):
        pass

    def indices(self, batch_size: int = None):
        raise NotImplementedError()

    def __call__(self, batch_size: int = None):
        return take_elems(getattr(self, 'data'), self.indices(batch_size=batch_size))
