""" Collection of Data Scaling Transformers """

from typing import Dict, Iterable, Tuple, Union
from ..changemap.changemap import ChangeMap
from .running_stats import RunningStats


class MinMaxScaler(RunningStats):
    """
    Scalers are also very important to ML frameworks. Many model types, notably neural networks among
    others, have a significant increase in performance when the input features are normalized into
    comparable scales. `MinMaxScaler` extends `RunningStats` to keep track of the minimum and maximum of
    each marked feature. Then, it computes a simple normalization technique to scale the output between
    `0` and `1` by adding the minimum value found and dividing by the maximum for each sample. Example:

    ```python
    arr = [random.random() * 100 for _ in range(100)]
    transformer = MinMaxScaler()
    transformer.fit(arr)
    transformer.transform(arr)[:10]
    # Output:
    # array([0.84741866, 0.76052674, 0.42148772, 0.25903933, 0.51263612,
    #        0.40577351, 0.7864978 , 0.30365325, 0.47778812, 0.58509741])
    ```
    """

    def __init__(
        self,
        columns: Union[Dict, Iterable[int]] = None,
        output_range: Tuple[int, int] = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        columns : dict, Iterable[int]
            TODO
        output_range : Tuple[int, int]
            TODO
        verbose : bool
            TODO
        """
        if columns is None:
            columns = [0]
        super().__init__(output_range=output_range, columns=columns, verbose=verbose)
        self.output_range = output_range or (0, 1)

    def transform(self, X: Iterable[Iterable]):
        X = self.check_X(X)

        for i, col in enumerate(X):
            if i not in self.columns_:
                continue
            X[i] = (col - self.min_[i]) / (self.max_[i] - self.min_[i])

        # Reshape as vector if input was vector
        if self.input_is_vector_:
            X = X[0]

        return X

    def inverse_transform(self, X):
        X = self.check_X(X, ensure_shape=False, ensure_dtype=False)

        for i, col in enumerate(X):
            if i not in self.columns_:
                continue
            X[i] = col * (self.max_[i] - self.min_[i]) + self.min_[i]

        # Reshape as vector if input was vector
        if self.input_is_vector_:
            X = X[0]

        return X

    def on_input_shape_changed(self, change_map: ChangeMap):
        # Parent's callback will take care of adapting feature changes
        super().on_input_shape_changed(change_map)

        # This transformer does not change shape of input, so we must propagate the change upwards
        self.on_output_shape_changed(change_map)


class StandardScaler(RunningStats):
    """
    Similar to `MinMaxScaler`, `StandardScaler` uses `RunningStats` to compute [standard scaling](
    https://en.wikipedia.org/wiki/Feature_scaling#Standardization) that takes into account the mean and
    standard deviation, instead of the minimum and maximum, when performing the normalization. Example:

    ```python
    arr = [random.random() * 100 for _ in range(100)]
    transformer = StandardScaler()
    transformer.fit(arr)
    transformer.transform(arr)[:10]
    # Output:
    # array([ 0.97829734,  0.6513745 , -0.62422864, -1.23542577, -0.28129117,
    #        -0.6833519 ,  0.74908821, -1.06757002, -0.41240357, -0.00866223])
    ```
    """

    def __init__(self, columns: Union[Dict, Iterable[int]] = None, verbose: bool = False):
        """
        Parameters
        ----------
        columns : dict, Iterable[int]
            TODO
        verbose : bool
            TODO
        """
        if columns is None:
            columns = [0]
        super().__init__(columns=columns, verbose=verbose)

    def transform(self, X: Iterable[Iterable]):
        X = self.check_X(X)

        for i, col in enumerate(X):
            if i not in self.columns_:
                continue
            X[i] = (col - self.mean_[i]) / self.stdev_[i]

        # Reshape as vector if input was vector
        if self.input_is_vector_:
            X = X[0]

        return X

    def inverse_transform(self, X):
        X = self.check_X(X, ensure_shape=False, ensure_dtype=False)

        for i, col in enumerate(X):
            if i not in self.columns_:
                continue
            X[i] = col * self.stdev_[i] + self.mean_[i]

        # Reshape as vector if input was vector
        if self.input_is_vector_:
            X = X[0]

        return X

    def on_input_shape_changed(self, change_map: ChangeMap):
        # Parent's callback will take care of adapting feature changes
        super().on_input_shape_changed(change_map)

        # This transformer does not change shape of input, so we must propagate the change upwards
        self.on_output_shape_changed(change_map)
