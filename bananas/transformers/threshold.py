''' Threshold-based transformers '''

from typing import Iterable
from ..changemap.changemap import ChangeMap
from .running_stats import RunningStats

class VarianceThreshold(RunningStats):
    '''
    A common technique to discard features that do not add a lot of value to the final output is to
    evaluate their total variance and apply a threshold so those features that fall below the specified
    limit can be dropped from the output. While this could be done at a prior phase, doing it as a
    transformer poses the additiona benefit of accommodating for data potentially changing. This is done
    by the `VarianceThreshold` transformer.

    For example, a feature may have great variance in the first few thousand samples but, as we dive
    deeper into the dataset, its variance eventually falls below the threshold. Once a feature's
    variance falls below the threshold, it will be excluded from the output even if the variance goes
    back above the threshold.

    Example of `VarianceThreshold`:

    ```python
    arr_zero = [0. for _ in range(100)]
    arr_ones = [1. for _ in range(100)]
    arr_rand = [random.random() for _ in range(100)]
    data = [arr_zero, arr_ones, arr_rand]
    transformer = VarianceThreshold()
    output = transformer.fit(data).transform(data)
    print('Input columns: %d' % len(data))
    print('Output columns: %d' % len(output))
    print('Input matches output: %r' % array_equal(data[-1], output[0]))
    # Output:
    # Input columns: 3
    # Output columns: 1
    # Input matches output: True
    ```
    '''

    def __init__(self, columns: (dict, Iterable[int]) = None, threshold: float = 1E-2,
                 verbose: bool = False):
        '''
        Parameters
        ----------
        columns : dict, Iterable[int]
            TODO
        threshold : float
            TODO
        verbose : bool
            TODO
        '''
        super().__init__(threshold=threshold, columns=columns, verbose=verbose)
        self.threshold = threshold

        # Initialize working variables
        self.idx_del_ = {}
        self.fresh_transformer_ = True

    def transform(self, X):
        X = self.check_X(X)

        # Excluding categorical columns, remove if variance is below threshold
        output, idx_del = [], {}
        for i, col in enumerate(X):
            if i in self.idx_del_:
                # Do not let columns to be added after being removed
                idx_del[i] = True
            elif i not in self.columns_:
                # Skip if not looking at this column
                output.append(col)
            elif self.variance_[i] > self.threshold:
                # Keep as long as variance is higher than threshold
                output.append(col)
            else:
                # Delete from output if variance does not meet threshold
                idx_del[i] = True

        # Keep track of output shape and trigger the shape changed event if needed
        if idx_del != self.idx_del_:
            idx_del_out = [i for i in idx_del if i not in self.idx_del_]
            idx_add_out = [i for i in self.idx_del_ if i not in idx_del]
            input_len = len(X) - len(self.idx_del_)
            change_map = ChangeMap(input_len, idx_del=idx_del_out, idx_add=idx_add_out)

            self.print('Triggered output change, dropping:')
            for i in idx_del:
                self.print('Column: %d\tVariance: %.05f' % (i, self.variance_[i]))

            self.idx_del_ = idx_del
            self.on_output_shape_changed(change_map)

        return output

    def inverse_transform(self, X):
        raise NotImplementedError('Inverse transformation not supported by this transformer')

    def on_input_shape_changed(self, change_map: ChangeMap):
        self.print('Input changed: %r' % change_map)

        # Early exit: fresh transformer, just propagate changes downstream
        if self.fresh_transformer_:
            self.on_output_shape_changed(change_map)
            self._input_change_column_adapter(
                change_map, ['columns_', 'input_dtype_', 'input_shape_', 'idx_del_'])
            return

        # First reverse changes previously sent downstream
        self.on_output_shape_changed(
            ChangeMap(change_map.input_len - len(self.idx_del_), idx_add=self.idx_del_.keys()))

        # Send new change downstream
        self.on_output_shape_changed(change_map)

        # Adapt feature changes by parent and by our own attributes
        self._input_change_column_adapter(
            change_map, ['columns_', 'input_dtype_', 'input_shape_', 'idx_del_'])

        # Send adapted output change
        self.on_output_shape_changed(
            ChangeMap(change_map.output_len, idx_del=self.idx_del_.keys()))
