""" Miscellaneous Functions """

import sys
import copy
import warnings
import traceback
from typing import Any, Callable, Dict
from inspect import signature


def warn_ignore(*args, **kwargs):  # pylint: disable=unused-argument
    """Function used to ignore all warnings"""
    pass


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    """Function used to print callstack of warnings"""
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def valid_parameters(func: Callable, params: Dict[str, Any]):
    """ Returns the subset of `params` that can be applied to `func` """
    func_params = signature(func).parameters
    # If kwargs param is present, all potential params are valid
    if "kwargs" in func_params:
        return params
    # Otherwise, return the overlapping parameters
    return {key: val for key, val in params.items() if key in func_params}
