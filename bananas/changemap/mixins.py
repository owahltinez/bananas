from typing import Callable
from ..utils.errors import UnexpectedShapeError
from .changemap import ChangeMap


class InputChangeMixin(object):
    """
    Mixin class that adds relevant methods to a learner to handle shape changing with a `ChangeMap`.

    `InputChangeMixin` lets inheriting classes emit and handle
    feature set change events. When an object like a transformer modifies the feature set, it is its
    responsibility to emit the appropriate `ChangeMap` by calling the `on_output_shape_changed`
    function with the computed change map.

    Conversely, classes inheriting from `InputChangeMixin` should override the
    `on_input_shape_changed` function to appropriately handle changes in input. By default, the
    function raises `UnexpectedShapeError`, which is very undesirable! Classes implementing proper
    change map handling should try to avoid raising an error at all costs, even if that means losing
    some of the previous work. For example, a neural network may reset all of its weights so it can
    begin learning the data anew, rather than throwing an error. In a situation like that, raising a
    warning would be more appropriate.
    """

    def on_input_shape_changed(self, change_map: ChangeMap):
        """ Notify this learner that the input shape has changed """
        raise UnexpectedShapeError("Input shape changed. %r" % change_map)

    def on_output_shape_changed(self, change_map: ChangeMap):
        """ Notify all callbacks attached to this learner that the output shape has changed """
        for callback in getattr(self, "output_shape_changed_callbacks_", []):
            callback(change_map)

    def add_output_shape_changed_callback(self, callback: Callable[[ChangeMap], None]):
        """ Attach a new callback to be notified when the output shape of this learner changes """
        if not hasattr(self, "output_shape_changed_callbacks_"):
            self.output_shape_changed_callbacks_ = []
        self.output_shape_changed_callbacks_.append(callback)

    def _input_change_column_adapter(self, change_map: ChangeMap, attributes: list):
        for attribute in attributes:
            attr_curr = {}
            attr_prev = getattr(self, attribute, {}) or {}
            for i, j in change_map.build().items():
                if i in attr_prev and j != -1:
                    attr_curr[j] = attr_prev[i]
            setattr(self, attribute, attr_curr)
