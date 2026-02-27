# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import copy

from pasts.core.base_model import TimeSeriesModel
from pasts.core.datacube import BINARY_INVERSE, BINARY_SYMBOL, DataCube, _to_dataframe


class Decomposition(TimeSeriesModel):
    """Decomposition formula derived from the operations stack of a DataCube.

    Built automatically from a DataCube's ``_ops`` list (populated by in-place
    operators ``-=``, ``/=``, ``*=``, ``+=`` and :meth:`~pasts.core.datacube.DataCube.apply`).

    Can reconstruct the original signal from a predicted residual
    via :meth:`compose`.  Also usable as a standalone
    :class:`~pasts.core.base_model.TimeSeriesModel`:

    .. code-block:: python

        decomp = Decomposition(signal.residual._ops, model=DartsModel(AutoARIMA()))
        decomp.fit(train_data)
        forecast = decomp.forecast(horizon)

    Parameters
    ----------
    ops : list
        Operation stack (from ``DataCube._ops``).
    model : TimeSeriesModel, optional
        Model to fit on the residual.  Required for
        :meth:`fit` / :meth:`forecast` / :meth:`reverse_transform`.
    """

    nan_safe = True  # delegates NaN handling to sub-models

    def __init__(self, ops: list, model: "TimeSeriesModel" = None):
        self._ops = ops
        self._model = model

    def fit(self, X, covariates=None) -> "Decomposition":
        """Fit the decomposition on training data.

        Replays the recorded operations on *X* to compute the residual.
        Each :class:`TimeSeriesModel` component is deep-copied and refitted
        on the current residual before its values are applied.
        Finally, the residual model (if provided) is fitted on the
        resulting residual.

        Parameters
        ----------
        X : pd.DataFrame
            Training data (original signal space).
        covariates : :class:`~pasts.covariates.Covariates`, optional
            Covariates forwarded to the residual model.

        Returns
        -------
        self
        """
        if isinstance(X, DataCube):
            X = X.data

        residual = X.copy()
        new_ops = []
        for entry in self._ops:
            if entry[0] == 'unary':
                _, func, inverse = entry
                residual = func(residual)
                new_ops.append(('unary', func, inverse))
            elif entry[0] == 'binary':
                _, inverse_op, component = entry
                forward_op = BINARY_INVERSE[inverse_op]
                if isinstance(component, TimeSeriesModel):
                    component = copy.deepcopy(component)
                    component.fit(residual, covariates=covariates)
                values = _to_dataframe(component, residual.index)
                residual = forward_op(residual, values)
                new_ops.append(('binary', inverse_op, component))
        self._ops = new_ops

        if self._model is not None:
            self._model = copy.deepcopy(self._model)
            self._model.fit(residual, covariates=covariates)

        return self

    def reverse_transform(self, i: int):
        """Predict and compose back through the decomposition.

        Parameters
        ----------
        i : int
            Number of steps (positive for forecast, negative for historical).

        Returns
        -------
        pd.DataFrame
        """
        if self._model is None:
            raise ValueError(
                "No residual model set on this Decomposition. "
                "Pass model= to the constructor."
            )
        prediction = self._model.reverse_transform(i)
        horizon = i if i > 0 else None
        return self.compose(DataCube(prediction), horizon=horizon).data

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
                _, _, inverse = entry
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
