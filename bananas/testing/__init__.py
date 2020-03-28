'''
Test suite and a number of utilities used to ensure compatiblity with the API.

The testing module is of great importance and any users of this package are greatly discouraged
from modifying it. It ensures compatibility across different ML frameworks, so their estimators and
transformers may be used in combination with one another. It also allows for the creation of tools
and other libraries that can be agnostic to the underlying implementation and can target a stable
API.
'''

from .generators import \
    generate_array_booleans, generate_array_chars, generate_array_floats, generate_array_ints, \
    generate_array_int_floats, generate_array_uints, generate_array_nones, generate_array_strings, \
    generate_images, generate_onehot_matrix, generate_array_infinities
from .dummy import DummyClassifier, DummyRegressor
from .learners import test_learner
