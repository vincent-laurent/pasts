# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import operator

import numpy as np
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


def _make_future_index(origin_index: pd.Index, i: int) -> pd.Index:
    """Generate a future index of length *i* extending *origin_index*."""
    if isinstance(origin_index, pd.DatetimeIndex):
        freq = origin_index.freq or pd.infer_freq(origin_index)
        return pd.date_range(start=origin_index[-1], periods=i + 1, freq=freq)[1:]
    # Numeric index: extend with constant step
    step = np.mean(np.diff(origin_index.to_numpy(dtype=float)))
    start = float(origin_index[-1])
    return pd.Index(start + np.arange(1, i + 1) * step)


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
        result = component.reverse_transform(i)
        # Force the index to match the target when lengths agree.
        # This is needed when a non-parametric trend was fit on a subset
        # (e.g. train_data only): the stored trend has the correct values
        # but carries the training-period DatetimeIndex instead of the
        # target (e.g. test-period) index.
        if hasattr(result, 'index') and len(result) == len(index):
            result = result.set_axis(index)
        return result
    if callable(component):
        return component(index)
    if isinstance(component, (int, float)):
        return component
    raise TypeError(f"Unsupported component type: {type(component)}")


class Operation:
    """Base class for recording reversible operations.

    In-place operators (``-=``, ``+=``, ``*=``, ``/=``) record each
    operation in ``_ops`` and call a hook (``_apply_binary``) that
    subclasses can override to apply the operation to data.

    Attributes
    ----------
    _ops : list
        Stack of recorded operations.
    """

    def __init__(self):
        self._ops = []

    def _apply_binary(self, op, other):
        """Hook called after recording a binary op. No-op by default."""
        pass

    def _apply_unary(self, func):
        """Hook called after recording a unary op. No-op by default."""
        pass

    def __isub__(self, other):
        self._ops.append(('binary', operator.add, other))
        self._apply_binary(operator.sub, other)
        return self

    def __iadd__(self, other):
        self._ops.append(('binary', operator.sub, other))
        self._apply_binary(operator.add, other)
        return self

    def __imul__(self, other):
        self._ops.append(('binary', operator.truediv, other))
        self._apply_binary(operator.mul, other)
        return self

    def __itruediv__(self, other):
        self._ops.append(('binary', operator.mul, other))
        self._apply_binary(operator.truediv, other)
        return self

    def apply(self, func, inverse) -> "Operation":
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
        self._apply_unary(func)
        return self

    def components(self) -> list:
        """List the components used in binary operations."""
        return [entry[-1] for entry in self._ops if entry[0] == 'binary']

    def __repr__(self):
        expr = "residual"
        for entry in reversed(self._ops):
            if entry[0] == 'unary':
                _, _, inverse = entry
                name = getattr(inverse, '__name__', str(inverse))
                expr = f"{name}({expr})"
            elif entry[0] == 'binary':
                _, inverse_op, comp = entry
                name = getattr(comp, 'name', comp.__class__.__name__)
                sym = BINARY_SYMBOL[inverse_op]
                if sym in ('*', '/'):
                    if '+' in expr or '-' in expr:
                        expr = f"{name} {sym} ({expr})"
                    else:
                        expr = f"{name} {sym} {expr}"
                else:
                    expr = f"{name} {sym} {expr}"
        return expr

class DataCube(Operation, pd.DataFrame):
    """
    Data structure wrapping a DataFrame with a temporal index.

    Inherits from ``pd.DataFrame`` so that forward operators
    (+, -, *, /) work out of the box.

    In-place operators (-=, /=, *=, +=) are recorded in ``_ops`` so that
    a :class:`~pasts.core.decomposition.Decomposition` can reverse them.

    Attributes
    ----------
    _ops : list
        Stack of recorded operations (populated by in-place operators).
    """

    _metadata = ['_ops']

    def __init__(self, data=None, **kwargs):
        Operation.__init__(self)
        if data is not None and isinstance(data, pd.DataFrame):
            _check_temporal_index(data.index)
        pd.DataFrame.__init__(self, data=data, **kwargs)

    @property
    def _constructor(self):
        return _datacube_internal

    @property
    def data(self):
        return pd.DataFrame(self)


    def _apply_binary(self, op, other):
        values = _to_dataframe(other, self.index)
        result = op(pd.DataFrame(self), values)
        self[result.columns] = result

    def _apply_unary(self, func):
        result = func(pd.DataFrame(self))
        pd.DataFrame.__init__(self, data=result)


def _datacube_internal(data=None, **kwargs):
    """Construct a DataCube without index validation.

    Used as ``_constructor`` so that pandas-internal operations (e.g.
    ``.sum()``, ``.mean()``) that produce non-temporal indices don't crash.
    """
    obj = DataCube.__new__(DataCube)
    Operation.__init__(obj)
    pd.DataFrame.__init__(obj, data=data, **kwargs)
    return obj
