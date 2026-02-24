# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from pasts.core.datacube import BINARY_SYMBOL, DataCube, _to_dataframe


class Decomposition:
    """Decomposition formula derived from the operations stack of a DataCube.

    Built automatically from a DataCube's ``_ops`` list (populated by in-place
    operators ``-=``, ``/=``, ``*=``, ``+=`` and :meth:`~pasts.core.datacube.DataCube.apply`).

    Can reconstruct the original signal from a predicted residual
    via :meth:`compose`.
    """

    def __init__(self, ops: list):
        self._ops = ops

    def compose(self, predicted_residual: DataCube, index=None, horizon: int = None) -> DataCube:
        """Reconstruct the signal from a predicted residual.

        Walks the operation stack in reverse and applies each inverse.

        Parameters
        ----------
        predicted_residual : DataCube
            The predicted residual values.
        index : pd.Index, optional
            Target index. Defaults to the residual's index.
        horizon : int, optional
            If positive, components produce future values (forecast mode).
            If None, components produce historical values.
        """
        idx = index or predicted_residual.data.index
        result = predicted_residual.data.copy()

        for entry in reversed(self._ops):
            if entry[0] == 'unary':
                _, func, inverse = entry
                result = inverse(result)
            elif entry[0] == 'binary':
                _, inverse_op, component = entry
                values = _to_dataframe(component, idx, horizon=horizon)
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
