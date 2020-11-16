""" Adaptation of multiple transformers to handle N-dimensional data instead of only 2D """

import warnings
from typing import Iterable, Set

import numpy
from ..changemap.changemap import ChangeMap
from ..utils.arrays import ARRAY_LIKE, check_array, difference, transpose_array, equal_nested
from ..utils.constants import DTYPE_UINT8
from .base import ColumnHandlingTransformer


class LabelEncoder(ColumnHandlingTransformer):
    """
    Encoding of features is a crucial task in any fully-fledged ML framework. this library implements
    two feature encoders: `LabelEncoder` and `OneHotEncoder`. The `LabelEncoder` transforms the
    requested features into ordinal integers. Consider the following example:

    ```python
    word_vector = ['banana', 'apple', 'apple', 'orange', 'banana', 'banana', 'banana', 'orange']
    encoder = LabelEncoder()
    encoder.fit(word_vector).transform(word_vector)
    # Output: array([1, 0, 0, 2, 1, 1, 1, 2], dtype=uint8)
    ```

    Note that the ordering of the features is not guaranteed. I.e. even though `banana` appeared first
    in the sample batch, it did not get assigned the integer `0`. Alternatively, we could re-write the
    previous example using a `dict` type for the `columns` argument to explicitly tell the encoder which
    possible values it may be encoding:

    ```python
    word_vector = ['banana', 'apple', 'apple', 'orange', 'banana', 'banana', 'banana', 'orange']
    encoder = LabelEncoder(columns={0: ['apple', 'banana', 'orange', 'watermelon']})
    encoder.fit(word_vector).transform(word_vector)
    # Output: array([1, 0, 0, 2, 1, 1, 1, 2], dtype=uint8)
    ```
    """

    def __init__(self, columns: (dict, Set[int]) = None, verbose: bool = False):
        """
        Parameters
        ----------
        columns : dict, Set[int]
            TODO
        verbose : bool
            TODO
        """
        super().__init__(columns=columns, verbose=verbose)
        for col, classes in self.columns_.items():
            # Ensure that each column contains at least an empty set
            # if classes is None:
            #     self.columns_[col] = set()
            # Ensure that the type is of Set
            self.columns_[col] = set(classes)

    def fit(self, X: Iterable[Iterable]):
        X = self.check_X(X)
        for idx, classes in self.columns_.items():
            ix_last = classes.index(None) if None in classes else len(classes)
            # Cast column to object type before comparison since classes can be int or str type
            # This gets rid of a DeprecationWarning about element-wise comparison (because __iter__)
            # TODO: X[idx] may not be a numpy array
            for label in difference(X[idx].astype(object), classes):
                if ix_last >= len(classes):
                    # classes.append(label)
                    classes.add(label)
                else:
                    classes[ix_last] = label
                ix_last += 1
        return self

    def transform(self, X):
        X = self.check_X(X)

        # Go through each of the classes and find the corresponding column
        for i, classes in self.columns_.items():

            # Create a corresponding column of enum int type
            arr = (numpy.ones(len(X[i])) * -1).astype(DTYPE_UINT8[0])

            # Swap labels for their corresponding encoded index
            for idx, label in enumerate(classes):
                if label is None:
                    break
                arr[X[i] == label] = idx

            # Replace column in original data
            X[i] = arr

        # Check each column array again to optimize possible dtype conversions
        X = [check_array(col) for col in X]

        # Reshape as vector if input was vector.
        if self.input_is_vector_:
            X = X[0]

        return X

    def inverse_transform(self, X):
        self.check_attributes("input_is_vector_")
        X = self.check_X(X, ensure_shape=False, ensure_dtype=False)

        for idx1, classes in self.columns_.items():
            # Create a corresponding column of the original dtype
            arr = numpy.zeros(shape=len(X[idx1]), dtype=self.input_dtype_[idx1])

            # Swap encoded indices for their corresponding labels
            for idx2, label in enumerate(classes):
                arr[X[idx1] == idx2] = label

            # Replace column in original data
            X[idx1] = arr

        # Check each column array again to optimize possible dtype conversions
        X = [check_array(col) for col in X]

        # Reshape as vector if input was vector
        if self.input_is_vector_:
            X = X[0]

        return X

    def on_input_shape_changed(self, change_map: ChangeMap):
        # Parent's callback will take care of adapting feature changes
        super().on_input_shape_changed(change_map)

        # This transformer does not change shape of input, so we must propagate the change upwards
        self.on_output_shape_changed(change_map)


class OneHotEncoder(LabelEncoder):
    """
    The `OneHotEncoder` extends the label encoder and outputs a one-hot vector for each sample. We
    must keep in mind that this library uses a **column-first format**, so the output will be a list
    of features representing a set of each of the one-or-zero elements in the one-hot vector for
    each sample. This is better illustrated with an example:

    ```python
    word_vector = ['banana', 'apple', 'apple', 'orange', 'banana', 'banana', 'banana', 'orange']
    encoder = OneHotEncoder()
    encoder.fit(word_vector).transform(word_vector)
    # Output:
    # [array([0, 1, 1, 0, 0, 0, 0, 0], dtype=uint8),
    #  array([1, 0, 0, 0, 1, 1, 1, 0], dtype=uint8),
    #  array([0, 0, 0, 1, 0, 0, 0, 1], dtype=uint8)]
    ```

    The output of one-hot encoding out `word_vector` should be read top-to-bottom as seen above. Then,
    the one-hot vector for the first element in the sample is `[0,1,0]`, the second one `[1,0,0]`, etc.

    Giving the encoder explicit potential values for the input and adding a not-yet-seen value will
    result in a different output:

    ```python
    word_vector = ['banana', 'apple', 'apple', 'orange', 'banana', 'banana', 'banana', 'orange']
    encoder = OneHotEncoder(columns={0: ['apple', 'banana', 'orange', 'watermelon']})
    encoder.fit(word_vector).transform(word_vector)
    # Output:
    # [array([0, 1, 1, 0, 0, 0, 0, 0], dtype=uint8),
    #  array([1, 0, 0, 0, 1, 1, 1, 0], dtype=uint8),
    #  array([0, 0, 0, 1, 0, 0, 0, 1], dtype=uint8),
    #  array([0, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)]
    ```

    Note that the last output feature is zero for all samples, this is because the expected word
    (watermelon) is never encountered.
    """

    def __init__(self, columns: (dict, Iterable[int]) = None, verbose: bool = False):
        """
        If classes are not known before hand, a dict of {column: [None, None, None...]} can be
        passed as placeholders.

        Parameters
        ----------
        columns : dict, Iterable[int]
            TODO
        verbose : bool
            TODO
        """
        super().__init__(columns=columns, verbose=verbose)

        # Initialize working variables
        self.output_len_ = None
        self.warned_once_ = False
        self.fresh_transformer_ = True
        self.last_known_classes_ = {}
        self.output_changed_pending_ = []

    def _num_to_vec(self, num, idx=0):
        vec = [0] * len(self.columns_[idx])
        try:
            vec[num] = 1
        except IndexError:
            if not self.warned_once_:
                warnings.warn("Unable to vectorize unseen class %d" % num)
                self.warned_once_ = True
        return tuple(vec)

    def _vec_to_num(self, vec):
        for idx, val in enumerate(vec):
            if val == 1:
                return idx
        return -1

    def fit(self, X: Iterable[Iterable]):
        super().fit(X)

        idx_add = []
        for idx, classes in self.columns_.items():
            idx_known = self.last_known_classes_.get(idx, [])
            if idx_known and all(equal_nested(classes, idx_known)):
                continue

            # Find any unseen classes and adapt on the go if more columns need to be added
            idx_unseen = [c for c in classes if c not in idx_known]
            if idx_unseen or self.fresh_transformer_:
                self.last_known_classes_[idx] = idx_known + idx_unseen
                idx_add += [idx for _ in range(idx, idx + len(idx_unseen))]

        # Store pending changes in internal object
        if idx_add:
            # Input size depends on whether we have already sent out transformations
            input_len = self.output_len_ or len(self.input_shape_)
            # If no transformation has been sent yet, we need to signal deletion of unencoded cols
            idx_del = self.columns_.keys() if self.fresh_transformer_ else None
            # Build change map with all info and fire event
            change_map = ChangeMap(input_len, idx_del=idx_del, idx_add=idx_add)
            self.output_changed_pending_.append(change_map)
            # Flag transformer as dirty
            self.fresh_transformer_ = False

        return self

    def transform(self, X: Iterable[Iterable]):
        # First, convert each of the classes to ordered ints using parent's transform()
        X = self.check_X(super().transform(X), ensure_shape=False, ensure_dtype=False)

        # Keep track of output column shape
        self.output_len_ = len(X) + sum(map(len, self.columns_.values())) - len(self.columns_)

        if self.output_changed_pending_:
            self.print("Triggered output change, encoding:")
            for idx, classes in self.columns_.items():
                self.print("Column: %d\tVector length: %d" % (idx, len(classes)))

        # If there are pending changes, fire event now
        while self.output_changed_pending_:
            change_map = self.output_changed_pending_.pop(0)
            self.on_output_shape_changed(change_map)

        # Then break out each of the columns containing classes into an N-dimensional vector
        output = []
        for i, col in enumerate(X):
            # When the column is not being encoded, copy as-is
            if i not in self.columns_:
                output.append(col)
                continue

            # Otherwise breakout into encoded matrix. E.g. [[0 0 1] [0 1 0] [1 0 0] [0 1 0] ...]
            self.warned_once_ = False
            arr_of_vecs = numpy.array(
                list(map(lambda x, idx=i: self._num_to_vec(x, idx), col)), dtype=DTYPE_UINT8[0]
            )

            # Copy each column of the matrix to final output
            for j in range(arr_of_vecs.shape[1]):
                output.append(arr_of_vecs[:, j])

        # Check each column array again to optimize possible dtype conversions
        X = [check_array(col) for col in output]

        # Reshape as vector if input was vector
        if self.input_is_vector_ and len(X) == 1:
            X = X[0]
        return X

    def inverse_transform(self, X: Iterable[Iterable]):
        self.check_attributes("input_is_vector_")
        X = self.check_X(X, ensure_shape=False, ensure_dtype=False)

        # Compute the total number of expected columns
        max_idx = (
            len(self.input_shape_)
            + sum([len(labels) for labels in self.columns_.values()])
            - len(self.columns_)
        )
        assert len(X) == max_idx, "Unexpected number of columns found. Expected: %d, found: %d" % (
            max_idx,
            len(X),
        )

        # Compute columns that are result of one-hot encoding breakout
        output = []
        idx_in = 0  # the index of the current input column
        idx_out = 0  # the index of the current output column
        while idx_in < max_idx:

            # If the current index does not correspond to an encoded column, add it as-is
            if idx_out not in self.columns_:
                output.append(X[idx_in])
                idx_in += 1
                idx_out += 1
                continue

            # Otherwise, inverse the encoding operation and add a single column to output
            num_labels = len(self.columns_[idx_out])
            arr_of_vecs = numpy.array(
                transpose_array([X[idx_in + j] for j in range(num_labels)]), dtype=DTYPE_UINT8[0]
            )
            # TODO: vectorize this operation for performance
            output.append(check_array([self._vec_to_num(vec) for vec in arr_of_vecs]))

            # Increase index by number of columns processed in this batch
            idx_in += num_labels
            idx_out += 1

        # Now pass the new columns through the parent's inversion
        output = super().inverse_transform(output)

        return output

    def on_input_shape_changed(self, change_map: ChangeMap):
        self.print("Input changed: %r" % change_map)

        # Early exit: fresh transformer, just propagate changes downstream
        if not self.output_len_:
            self.print("Fresh transformer, passing change map through")
            self.on_output_shape_changed(change_map)
            self._input_change_column_adapter(
                change_map, ["columns_", "input_dtype_", "input_shape_", "last_known_classes_"]
            )
            return

        # First reverse changes previously sent downstream
        idx_offset = 0
        idx_del, idx_add_1, idx_add_2 = [], [], []
        for idx, classes in self.columns_.items():
            idx_ = idx + idx_offset
            # Delete all added classes
            idx_del += list(range(idx_, idx_ + len(classes)))
            # Re-add a column for original labels
            idx_add_1 += [idx_]
            # Later, re-add all the classes
            idx_add_2 += [idx for _ in range(len(classes))]
            # Offset is number of classes deleted minus the original column re-added
            idx_offset += len(classes) - 1

        change_map_reverse = ChangeMap(self.output_len_, idx_del=idx_del, idx_add=idx_add_1)
        self.on_output_shape_changed(change_map_reverse)

        # Send new change downstream
        self.on_output_shape_changed(change_map)

        # Adapt feature changes by parent and by our own attributes
        self._input_change_column_adapter(
            change_map, ["columns_", "input_dtype_", "input_shape_", "last_known_classes_"]
        )

        # Send adapted output change
        change_map_adapted = ChangeMap(
            change_map.output_len, idx_del=self.columns_.keys(), idx_add=idx_add_2
        )
        self.on_output_shape_changed(change_map_adapted)
