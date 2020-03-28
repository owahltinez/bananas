''' Miscellaneous constant definitions; utility methods for arrays, images and more. '''

import warnings

# Import numpy
__NUMPY_AVAILABLE__ = False
try:
    import numpy
    __NUMPY_AVAILABLE__ = True
except ImportError:
    pass

# Import tqdm and workaround some of their issues
import tqdm
try:
    __IPYTHON__
    from tqdm.notebook import tqdm as tqdm_
except NameError:
    from tqdm import tqdm as tqdm_

# https://github.com/tqdm/tqdm/issues/481
tqdm_.monitor_interval = 0
warnings.filterwarnings('ignore', category=tqdm.TqdmSynchronisationWarning)
