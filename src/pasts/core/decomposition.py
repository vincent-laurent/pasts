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

import pandas as pd

from pasts.core.base_model import TimeSeriesModel
from pasts.core.datacube import BINARY_INVERSE, BINARY_SYMBOL, DataCube, Operation, _to_dataframe


class DecompositionModel(Operation, TimeSeriesModel):
    def __init__(self):
        Operation.__init__(self)
        self._residual = None

    def fit(self, X, covariates=None):
        if isinstance(X, DataCube):
            X = X.data
        dc = DataCube(X.copy())
        new_ops = []
        for entry in self._ops:
            if entry[0] == 'unary':
                _, func, _ = entry
                dc._apply_unary(func)
                new_ops.append(entry)
            elif entry[0] == 'binary':
                _, inverse_op, component = entry
                forward_op = BINARY_INVERSE[inverse_op]
                if isinstance(component, TimeSeriesModel):
                    component = copy.deepcopy(component)
                    component.fit(dc.data, covariates=covariates)
                dc._apply_binary(forward_op, component)
                new_ops.append(('binary', inverse_op, component))
        self._ops = new_ops
        self._residual = dc.data
        return self

    @property
    def residual(self):
        if self._residual is None:
            raise ValueError("Call fit() first.")
        return self._residual

    def compose(self, predicted_residual, index=None, horizon=None):
        if isinstance(predicted_residual, DataCube):
            idx = index or predicted_residual.data.index
            result = predicted_residual.data.copy()
        else:
            idx = index or predicted_residual.index
            result = predicted_residual.copy()

        for entry in reversed(self._ops):
            if entry[0] == 'unary':
                _, _, inverse = entry
                result = inverse(result)
            elif entry[0] == 'binary':
                _, inverse_op, component = entry
                values = _to_dataframe(component, idx, horizon=horizon)
                result = inverse_op(result, values)
        return DataCube(result)

    def recompose(self, raw, horizon):
        return self.compose(DataCube(raw), horizon=horizon).data

    def reverse_transform(self, i):
        raise NotImplementedError

    
class Decomposition(DataCube):
    """Reversible preprocessing pipeline backed by a DataCube.

    Inherits all in-place operators (``-=``, ``+=``, ``*=``, ``/=``,
    :meth:`~pasts.core.datacube.DataCube.apply`) from :class:`DataCube`.
    TimeSeriesModel components are automatically fitted on the current
    data when applied via an in-place operator.

    Can be used standalone::

        decomp = Decomposition(data)
        decomp -= LinearTrend()
        model  = DartsModel(AutoARIMA(), decomposition=decomp)

    Or through :class:`~pasts.signal.Signal`::

        signal.decompose("MA_Trend")
        signal.decompositions["MA_Trend"] -= MovingAverageTrend(30)
        signal.decompositions["MA_Trend"].apply_model(XGBModel(lags=250))

    Parameters
    ----------
    data : pd.DataFrame
        Initial signal data.
    signal : Signal, optional
        Parent signal (set by ``Signal.decompose()``).
    name : str
        Decomposition name (used for ``"name__ModelName"`` key).
    """

    _metadata = ['_ops', '_signal', '_name', '_residual']

    def __init__(self, data: pd.DataFrame, signal=None, name: str = "default"):
        super().__init__(data)
        self._signal = signal
        self._name = name
        self._residual = None

    @property
    def _constructor(self):
        return Decomposition

    # -------------------------------------------------------------------
    # Auto-fit components in operators
    # -------------------------------------------------------------------

    def _prepare_component(self, other):
        """Deep-copy and fit a TimeSeriesModel on current data."""
        if isinstance(other, TimeSeriesModel):
            other = copy.deepcopy(other)
            other.fit(pd.DataFrame(self))
            self._auto_register(other)
        return other

    def _auto_register(self, component):
        """Register the component as a model in the parent Signal.

        The component was fitted on full data (for correct ``_data``
        modification).  For the registered model we create a separate
        copy fitted on **train data only**, so its predictions are
        true out-of-sample forecasts covering the test period.
        """
        if self._signal is None or self._signal.train_data is None:
            return
        from pasts.core.model_result import ModelResult
        # Fit a train-only copy for fair evaluation
        train_component = copy.deepcopy(component)
        train_component.fit(self._signal.train_data)
        n_test = len(self._signal.test_data)
        predictions = train_component.reverse_transform(n_test)
        if len(predictions) == n_test:
            predictions.index = self._signal.test_data.index
        self._signal.models[self._name] = ModelResult(
            estimator_on_train=train_component,
            predictions=predictions,
            best_parameters="default",
            _data=self._signal.data,
            _covariates=self._signal._covariates,
        )

    def __isub__(self, other):
        return super().__isub__(self._prepare_component(other))

    def __iadd__(self, other):
        return super().__iadd__(self._prepare_component(other))

    def __imul__(self, other):
        return super().__imul__(self._prepare_component(other))

    def __itruediv__(self, other):
        return super().__itruediv__(self._prepare_component(other))

    # -------------------------------------------------------------------
    # Residual
    # -------------------------------------------------------------------

    @property
    def residual(self) -> pd.DataFrame:
        """The residual obtained after replaying all operations.

        Available only after :meth:`fit` has been called.
        """
        if self._residual is None:
            raise ValueError("Call fit() first.")
        return self._residual

    # -------------------------------------------------------------------
    # Fit / compose
    # -------------------------------------------------------------------

    def fit(self, X, covariates=None) -> "Decomposition":
        """Replay the recorded operations on *X* to compute the residual.

        Each :class:`TimeSeriesModel` component is deep-copied and refitted
        on the current residual before its values are applied.

        Parameters
        ----------
        X : pd.DataFrame
            Training data (original signal space).
        covariates : :class:`~pasts.covariates.Covariates`, optional
            Covariates forwarded to sub-model components.

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
        self._residual = residual

        return self

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

    def recompose(self, raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Recompose raw predictions from residual space to original signal space."""
        return self.compose(DataCube(raw), horizon=horizon).data

    # -------------------------------------------------------------------
    # Signal integration: apply_model
    # -------------------------------------------------------------------

    @property
    def covariates(self):
        """Covariates from the parent Signal (if any)."""
        if self._signal is None:
            return None
        return self._signal._covariates

    def apply_model(self, model, gridsearch=False, parameters=None, save_model=False):
        """Wrap *model* with this decomposition and train via the parent Signal.

        Parameters
        ----------
        model : TimeSeriesModel or Darts model
            The forecasting model to train on the residual.
        gridsearch : bool, optional
            Whether to perform a gridsearch (default ``False``).
        parameters : dict, optional
            Gridsearch parameters.
        save_model : bool, optional
            Whether to persist the result to disk (default ``False``).
        """
        if self._signal is None:
            raise ValueError("apply_model requires a Signal context (use signal.decompose()).")

        from pasts.components.darts_model import DartsModel
        from pasts import persistence

        # Build a fresh Decomposition with the same ops (will be refitted on train data)
        decomp = Decomposition(pd.DataFrame(self).copy())
        decomp._ops = copy.deepcopy(self._ops)

        if not isinstance(model, TimeSeriesModel):
            gridsearch_params = parameters if gridsearch else None
            model = DartsModel(model, gridsearch_params=gridsearch_params, decomposition=decomp)
        else:
            model = copy.deepcopy(model)
            model._decomposition = decomp

        full_name = f"{self._name}__{model.name}"
        result = self._signal._fit_and_predict(model)
        self._signal.models[full_name] = result

        if save_model:
            persistence.save_model(self._signal.path, full_name, result)
            persistence.save_common_data(
                self._signal.path, self._signal.train_data, self._signal.test_data
            )

    # -------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------

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
