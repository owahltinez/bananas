from typing import Dict, List


class ChangeMap(object):
    '''
    `ChangeMap` objects represent a modification of the feature set via addition or deletion of
    columns. Reordering of columns is not a supported modification. Change maps come in handy when
    we have setup a pipeline of transformers and estimators that may modify the feature set, for
    example:

    - [One-hot encoders](../transformers/encoders.html#bananas.transformers.encoders.OneHotEncoder)
      will convert a single feature into N categorical columns.

    - [Variance threshold](../transformers/threshold.html#bananas.transformers.threshold.VarianceThreshold)
      may supress features that fall below a certain threshold after the first batch of samples has
      already been output.

    ## Pipelines
    Objects that are part of a [Pipeline](../core/pipeline.html#bananas.core.pipeline.Pipeline) are
    automatically chained so when one emits `on_output_shape_changed`, the corresponding object
    downstream in the pipeline receives an `on_input_shape_changed` with the appropriate
    `ChangeMap`.
    '''
    def __init__(self, column_count: int, idx_del: List[int] = None, idx_add: List[int] = None):
        '''
        Parameters
        ----------
        column_count : int
            Total number of columns in this input
        idx_del : List[int]
            Indices of columns being deleted
        idx_add : List[int]
            Indices of columns being added

        TODO: Add idx_add type information to signal if new columns are continuous, categorical...
        '''
        self.input_len = column_count
        self.idx_del = sorted(idx_del or [])
        self.idx_add = sorted(idx_add or [])
        self.output_len = self.input_len - len(self.idx_del) + len(self.idx_add)
        assert column_count - len(self.idx_del) >= 0, \
            ('Change map cannot delete more columns from output than there are in input. Expected '
             '<=%d, found: %d' % (column_count, len(self.idx_del)))

    def build(self) -> Dict[int, int]:
        '''
        Construct a dictionary mapping from each index in the column set prior to transformation to
        the column set posterior to transformation. Columns being deleted map to `-1`, and columns
        being added map from `-1`.
        '''

        # Step 0: Put indices in a straight map and start with zero offset
        change_map_dict = {idx: idx for idx in range(self.input_len)}
        change_map_offsets = {idx: 0 for idx in range(self.input_len)}

        # Step 1: Compute offsets for all indices from idx_del
        for idx in self.idx_del:
            change_map_dict[idx] = -1
            for i, j in change_map_offsets.items():
                if i >= idx: change_map_offsets[i] -= 1

        # Step 2: Compute offsets for all indices from idx_add
        for idx in self.idx_add:
            change_map_dict[-1] = change_map_dict.get(-1, []) + [idx]
            for i, j in change_map_offsets.items():
                if i + change_map_offsets[i] >= idx: change_map_offsets[i] += 1

        # Apply offsets to indices and return
        for i, j in change_map_dict.items():
            if i != -1 and j != -1:
                change_map_dict[i] += change_map_offsets[i]
        return change_map_dict

    def inverse(self):
        '''
        Reverse the transformation created by this `ChangeMap`. Essentially, this re-adds all
        deleted columns and deletes all added columns.
        '''
        return ChangeMap(self.input_len, idx_del=self.idx_add, idx_add=self.idx_del)

    def __repr__(self):
        return ('ChangeMap(input_len=(%d → %d). idx_del=%r, idx_add=%r)' %
                (self.input_len, self.output_len, self.idx_del, self.idx_add))

    def __str__(self):
        return '%r' % self.build()
