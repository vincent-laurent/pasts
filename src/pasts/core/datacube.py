# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import functools
import operator

import pandas as pd


def _check_temporal_index(index: pd.Index) -> None:
    """Raise TypeError if *index* is neither datetime, timedelta, nor numeric."""
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return
    if pd.api.types.is_numeric_dtype(index):
        return
    raise TypeError(
        f"DataCube index must be a DatetimeIndex, TimedeltaIndex, "
        f"or a numeric index, got {type(index).__name__}."
    )


BINARY_INVERSE = {
    operator.sub: operator.add,
    operator.add: operator.sub,
    operator.mul: operator.truediv,
    operator.truediv: operator.mul,
}

BINARY_SYMBOL = {
    operator.add: '+',
    operator.sub: '-',
    operator.mul: '*',
    operator.truediv: '/',
}


def _record(op):
    """Decorator: records the inverse binary op in _ops before executing."""
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, other):
            self._ops.append(('binary', BINARY_INVERSE[op], other))
            return method(self, other)
        return wrapper
    return decorator


def _to_dataframe(component, index: pd.Index, horizon: int = None):
    """Extract a DataFrame from any component type.

    Parameters
    ----------
    component : DataCube, object with reverse_transform, callable, int, or float
    index : pd.Index
        Target index for alignment.
    horizon : int, optional
        If positive, request future values from the component (forecast mode).
        If None, request historical values matching the index length.
    """
    if isinstance(component, DataCube):
        return component.data.reindex(index)
    if hasattr(component, 'reverse_transform'):
        i = horizon if horizon is not None and horizon > 0 else -len(index)
        return component.reverse_transform(i)
    if callable(component):
        return component(index)
    if isinstance(component, (int, float)):
        return component
    raise TypeError(f"Unsupported component type: {type(component)}")


class DataCube:
    """
    Data structure wrapping a DataFrame with a temporal index.

    The index (X axis) must be either a ``DatetimeIndex`` or a numeric
    index (int / float).  A ``TypeError`` is raised otherwise.

    In-place operators (-=, /=, *=, +=) are recorded in ``_ops`` so that
    a :class:`~pasts.core.decomposition.Decomposition` can reverse them.

    Operators +, -, *, / delegate directly to pandas.

    Attributes
    ----------
    data : pd.DataFrame
        The underlying DataFrame.
    _ops : list
        Stack of recorded operations (populated by in-place operators).
    """

    def __init__(self, data: pd.DataFrame):
        _check_temporal_index(data.index)
        self._data = data
        self._ops = []

    @property
    def data(self):
        return self._data

    # --- In-place binary operators (recorded) ---

    @_record(operator.sub)
    def __isub__(self, other):
        self._data -= _to_dataframe(other, self._data.index)
        return self

    @_record(operator.truediv)
    def __itruediv__(self, other):
        self._data /= _to_dataframe(other, self._data.index)
        return self

    @_record(operator.mul)
    def __imul__(self, other):
        self._data *= _to_dataframe(other, self._data.index)
        return self

    @_record(operator.add)
    def __iadd__(self, other):
        self._data += _to_dataframe(other, self._data.index)
        return self

    # --- Unary transformation (recorded) ---

    def apply(self, func, inverse) -> "DataCube":
        """Apply a unary function and record its inverse.

        Parameters
        ----------
        func : callable
            Function to apply to the data (e.g. ``np.log``).
        inverse : callable
            Inverse of *func* (e.g. ``np.exp``), used during recomposition.

        Returns
        -------
        self
        """
        self._ops.append(('unary', func, inverse))
        self._data = func(self._data)
        return self

    # --- Forward binary operators ---

    def __add__(self, other):
        if isinstance(other, DataCube):
            return DataCube(self._data + other._data)
        if isinstance(other, (int, float)):
            return DataCube(self._data + other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, DataCube):
            return DataCube(self._data - other._data)
        if isinstance(other, (int, float)):
            return DataCube(self._data - other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, DataCube):
            return DataCube(self._data * other._data)
        if isinstance(other, (int, float)):
            return DataCube(self._data * other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, DataCube):
            return DataCube(self._data / other._data)
        if isinstance(other, (int, float)):
            return DataCube(self._data / other)
        return NotImplemented

    # --- Reverse binary operators (scalar on the left) ---

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return DataCube(other + self._data)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return DataCube(other - self._data)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return DataCube(other * self._data)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return DataCube(other / self._data)
        return NotImplemented

    # --- Unary ---

    def __neg__(self):
        return DataCube(-self._data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(x._data if isinstance(x, DataCube) else x for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        return DataCube(result)
