""" Threshold-based transformers """

from typing import Dict, Iterable, Union
from ..changemap.changemap import ChangeMap
from ..testing.dummy import DummyTransformer


class FeatureDrop(DummyTransformer):
    """
    One of the most basic kinds of transformers is `FeatureDrop`. It simply drops the features marked
    during initialization. While this could be done at the input or sampling stages, this transformer
    becomes very useful during exploratory phases and when used in conjunction with other transformers.
    """

    def __init__(self, columns: Union[Dict, Iterable] = None, verbose: bool = False):
        """
        Parameters
        ----------
        columns : Union[Dict, Iterable]
            TODO
        verbose : bool
            TODO
        """
        super().__init__(columns=columns, verbose=verbose)
        self.fresh_transformer_ = True

    def transform(self, X):
        X = self.check_X(X)

        # Remove if column is in list of being dropped
        output = [col for i, col in enumerate(X) if i not in self.columns_]

        # Notify downstream if first transformation
        if self.fresh_transformer_:
            self.fresh_transformer_ = False
            change_map = ChangeMap(len(X), idx_del=self.columns_)
            self.print("Triggered output change: %r" % change_map)
            self.on_output_shape_changed(change_map)

        return output

    def inverse_transform(self, X):
        raise NotImplementedError("Inverse transformation not supported by this transformer")

    def on_input_shape_changed(self, change_map: ChangeMap):
        self.print("Input changed: %r" % change_map)

        # Early exit: fresh transformer, just propagate changes downstream
        if self.fresh_transformer_:
            self.on_output_shape_changed(change_map)
            self._input_change_column_adapter(
                change_map, ["columns_", "input_dtype_", "input_shape_"]
            )
            return

        # First reverse changes previously sent downstream
        curr_input_len = change_map.input_len - len(self.columns_)
        self.on_output_shape_changed(ChangeMap(curr_input_len, idx_add=self.columns_.keys()))

        # Send new change downstream
        self.on_output_shape_changed(change_map)

        # Adapt feature changes by parent and by our own attributes
        self._input_change_column_adapter(change_map, ["columns_", "input_dtype_", "input_shape_"])

        # Send adapted output change
        self.on_output_shape_changed(ChangeMap(change_map.output_len, idx_del=self.columns_.keys()))
