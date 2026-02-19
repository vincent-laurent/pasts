# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

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


class DataCube:
    """
    Data structure wrapping a DataFrame with a temporal index.

    The index (X axis) must be either a ``DatetimeIndex`` or a numeric
    index (int / float).  A ``TypeError`` is raised otherwise.

    Operators +, -, *, / delegate directly to pandas.

    Attributes
    ----------
    data : pd.DataFrame
        The underlying DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        _check_temporal_index(data.index)
        self._data = data

    @property
    def data(self):
        return self._data

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
