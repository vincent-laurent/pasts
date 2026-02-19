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

from pasts.datacube import DataCube


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
    """Decorator: records binary op in _ops before executing."""
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, other):
            self._ops.append(('binary', op, BINARY_INVERSE[op], other))
            return method(self, other)
        return wrapper
    return decorator


def _to_dataframe(component, index: pd.Index):
    """Extract a DataFrame from any component type."""
    if isinstance(component, DataCube):
        return component.data.reindex(index)
    if hasattr(component, 'reverse_transform'):
        return component.reverse_transform(-len(index))
    if callable(component):
        return component(index)
    if isinstance(component, (int, float)):
        return component
    raise TypeError(f"Unsupported component type: {type(component)}")


class Residual(DataCube):
    """DataCube that records all in-place operations (binary and unary).

    Binary operations (``-=``, ``/=``, ``*=``, ``+=``) are recorded via
    the ``@_record`` decorator.  Unary operations are recorded via the
    explicit ``.apply(func, inverse)`` method.
    """

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self._ops = []

    # --- Binary ops ---

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

    # --- Unary ops ---

    def apply(self, func, inverse) -> "Residual":
        """Apply a unary function and record its inverse.

        Parameters
        ----------
        func : callable
            Function to apply to the residual data (e.g. ``np.log``).
        inverse : callable
            Inverse of *func* (e.g. ``np.exp``), used during composition.

        Returns
        -------
        self
        """
        self._ops.append(('unary', func, inverse))
        self._data = func(self._data)
        return self


class Decomposition:
    """Decomposition formula derived from the operations stack.

    Built automatically from a :class:`Residual`'s ``_ops`` list.
    Can reconstruct the original signal from a predicted residual
    via :meth:`compose`.
    """

    def __init__(self, ops: list):
        self._ops = ops

    def compose(self, predicted_residual: DataCube, index=None) -> DataCube:
        """Reconstruct the signal from a predicted residual.

        Walks the operation stack in reverse and applies each inverse.
        """
        idx = index or predicted_residual.data.index
        result = predicted_residual.data.copy()

        for entry in reversed(self._ops):
            if entry[0] == 'unary':
                _, func, inverse = entry
                result = inverse(result)
            elif entry[0] == 'binary':
                _, op, inverse_op, component = entry
                values = _to_dataframe(component, idx)
                result = inverse_op(result, values)

        return DataCube(result)

    def components(self) -> list:
        """List the components used in binary operations."""
        return [entry[-1] for entry in self._ops if entry[0] == 'binary']

    def __repr__(self):
        expr = "residual"
        for entry in reversed(self._ops):
            if entry[0] == 'unary':
                _, func, inverse = entry
                name = getattr(inverse, '__name__', str(inverse))
                expr = f"{name}({expr})"
            elif entry[0] == 'binary':
                _, op, inverse_op, comp = entry
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
