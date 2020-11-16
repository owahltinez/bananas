import sys
from typing import Dict, Iterable, Set, Union
from ..changemap.changemap import ChangeMap
from ..core.learner import UnsupervisedLearner
from ..utils.arrays import ARRAY_LIKE


class BaseTransformer(UnsupervisedLearner):
    """ Implements interface methods relevant to transformers. """

    def __init__(self, verbose: bool = False, **kwargs):
        """
        Parameters
        ----------
        verbose : bool
            TODO
        """
        super().__init__(verbose=verbose, **kwargs)
        self.verbose = verbose

    def transform(self, X: Iterable[Iterable]):
        raise NotImplementedError()

    def inverse_transform(self, X: Iterable[Iterable]):
        raise NotImplementedError()


class ColumnHandlingTransformer(BaseTransformer):
    """ Implements a transformer that is aware of columns and handles input changes """

    def __init__(self, columns: Union[Dict, Iterable] = None, verbose: bool = False, **kwargs):
        """
        Parameters
        ----------
        columns : dict, Iterable
            TODO
        verbose : bool
            TODO
        """
        # TODO: document None as placeholder for # of classes
        super().__init__(columns=columns, verbose=verbose, **kwargs)
        self.columns_: Dict[int, Set] = {}
        self._transform_all = False
        if columns is None:
            self._transform_all = True
        elif isinstance(columns, dict):
            self.columns_ = {**columns}
        elif isinstance(columns, Iterable):
            self.columns_ = {c: None for c in columns}

    def check_X(self, X: Iterable[Iterable], ensure_2d=True, ensure_shape=True, ensure_dtype=True):

        # If we must transform all columns, add them as we find new ones
        if self._transform_all:
            for idx, _ in enumerate(X):
                if idx not in self.columns_:
                    self.columns_[idx] = None

        # Capture exception to print more details and re-raise immediately
        try:
            return super().check_X(
                X, ensure_2d=ensure_2d, ensure_shape=ensure_shape, ensure_dtype=ensure_dtype
            )
        except Exception as exc:
            self.print("Input validation failed.")
            self.print("Columns: %r" % self.columns_, file=sys.stderr)
            try:
                self.print("Input: %r" % [col[0] for col in X], file=sys.stderr)
            except Exception:
                pass
            raise exc

    def on_input_shape_changed(self, change_map: ChangeMap):
        self.print("Received input change: %r" % change_map)

        if not change_map:
            raise ValueError(
                "Input shape changed but no change map provided, so it cannot be "
                "handled gracefully."
            )

        # Modify internal columns to follow variables wherever they went
        self._input_change_column_adapter(change_map, ["columns_", "input_dtype_", "input_shape_"])

        # If we must transform all columns, make sure that all new columns are added
        if self._transform_all:
            for idx in change_map.idx_add:
                self.columns_[idx] = None

    def __repr__(self):
        cols = (
            self.columns_
            if any([v is not None for v in self.columns_.values()])
            else list(self.columns_.keys())
        )
        return "%s(columns=%r)" % (self.__class__.__name__, cols)
