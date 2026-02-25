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
import os
import warnings
from typing import Union

import numpy as np
import pandas as pd

from pasts.core.base_model import TimeSeriesModel
from pasts.core.datacube import DataCube
from pasts.core.decomposition import Decomposition
from pasts.core.model_result import ModelResult
from pasts.components.darts_model import DartsModel
from pasts.statistical_tests import StatAccessor
from pasts.validation import ValidationAccessor
from pasts.metrics import Metrics
from pasts import persistence
from pasts.visualization import PlotAccessor

def _build_ci(pred: pd.DataFrame, std_values: dict) -> pd.DataFrame:
    """Build a confidence-interval DataFrame from a predictions DataFrame and per-column std values."""
    df_itv = pd.DataFrame(index=pred.index, columns=pred.columns)
    weights = np.arange(1, len(pred) + 1, dtype=float)
    for ref in pred.columns:
        vals = pred[ref].values
        itv_inf = vals - 1.96 * std_values[ref] * np.sqrt(weights)
        itv_sup = vals + 1.96 * std_values[ref] * np.sqrt(weights)
        df_itv[ref] = list(zip(itv_inf, itv_sup))
    return df_itv


class Signal(DataCube):
    """A class to represent a signal.

    Combines data storage, decomposition, model fitting, scoring, and
    forecasting in a single orchestrator.

    Attributes
    ----------
    models : dict
        keys: model names, values: :class:`ModelResult` instances.
    """

    @staticmethod
    def _profiling(data: pd.DataFrame) -> dict:
        """Compute basic properties of the time series."""
        return {'shape': data.shape,
                'types': data.dtypes,
                'is_univariate': data.shape[1] == 1,
                'nanSum': data.isnull().sum(),
                'quantiles': data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T}

    def __init__(self, data: pd.DataFrame, path: str = None):
        """
        Constructs all the necessary attributes for the signal object.

        Parameters
        ----------
        data : pd.Dataframe
                Dataframe of time series with time as index and one or several entities as columns.
                Index must be of type DatetimeIndex.
        path : str, optional
                The path to the directory where the Signal data will be stored. The directory may or
                may not exist. If it doesn't exist, it will be created automatically.
                If None, no directory is created (e.g. when used as a residual).
        """
        if path and not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        super().__init__(data)
        self._properties = Signal._profiling(data)
        self._tests_stat = {}
        self._validation = ValidationAccessor(self)
        self.models = {}
        self._performance_models = {}
        self._decompositions = {}

    @property
    def train_data(self):
        """Train set as a pandas dataframe (computed on demand from validation split)."""
        ts = self._validation._timestamp
        return None if ts is None else self.data.loc[self.data.index <= ts]

    @property
    def test_data(self):
        """Test set as a pandas dataframe (computed on demand from validation split)."""
        ts = self._validation._timestamp
        return None if ts is None else self.data.loc[self.data.index > ts]

    @property
    def properties(self):
        """Dictionary of properties of the signal"""
        return self._properties

    @property
    def tests_stat(self):
        """Dictionary of statistical tests applied on data"""
        return self._tests_stat

    @property
    def performance_models(self):
        """Dictionary containing a maximum of 2 other dictionaries for unit-wise or time-wise scores. Dictionaries
        contain a dataframe for each scorer, with scores computed for all models."""
        return self._performance_models

    def handle_nan(self, method: str = "drop", **kwargs) -> None:
        """Remove or fill NaN values in the signal data (in place).

        Parameters
        ----------
        method : str
            ``"drop"`` — drop rows containing any NaN (default).
            ``"fill"`` — fill NaN with a value (default 0, override via *value=…*).
            ``"interpolate"`` — ``pandas.DataFrame.interpolate`` + bfill + ffill.
        **kwargs
            Passed to the underlying pandas method (e.g. ``value=0`` for
            ``"fill"``, ``method='linear'`` for ``"interpolate"``).
        """
        if method == "drop":
            self._data = self._data.dropna(**kwargs)
        elif method == "fill":
            self._data = self._data.fillna(kwargs.pop("value", 0), **kwargs)
        elif method == "interpolate":
            self._data = self._data.interpolate(**kwargs).bfill().ffill()
        else:
            raise ValueError(
                f"Unknown method {method!r}. Use 'drop', 'fill', or 'interpolate'."
            )
        self._properties = Signal._profiling(self._data)
        if self._validation._timestamp is not None:
            self._validation.reset()
            warnings.warn(
                "Train/test split has been reset after handle_nan(). "
                "Call validation_split() again.",
                UserWarning,
            )

    def decompose(self, name: str = "default") -> None:
        """Initialize a named residual as a copy of the signal data.

        After calling this method, operate on ``signal.decompositions[name]`` to
        build the decomposition (e.g. ``signal.decompositions["t"] -= Trend().fit(...)``).
        Each named residual is a full :class:`Signal`, so all analysis methods
        (``apply_model``, ``validation_split``, etc.) work on it directly.
        Multiple named decompositions can coexist on the same signal.

        The new decomposition shares the parent's :class:`~pasts.validation.ValidationAccessor`,
        so the train/test boundary is always in sync without any explicit propagation.

        Parameters
        ----------
        name : str, optional
            Name for this decomposition (default ``"default"``).
            Calling ``decompose()`` without a name keeps backward compatibility
            with ``signal.residual``.
        """
        decomp_path = os.path.join(self.path, name) if self.path else None
        decomp = Signal(self.data.copy(), path=decomp_path)
        decomp._validation = self._validation   # share — no propagation needed
        self._decompositions[name] = decomp

    @property
    def plot(self) -> PlotAccessor:
        """Accessor for visualization methods.

        Usage::

            signal.plot()                              # raw signal
            signal.plot.acf()                          # autocorrelation
            signal.plot.predictions()                  # predictions (matplotlib)
            signal.plot.predictions(backend="plotly")  # predictions (plotly)
            signal.plot.forecast()                     # forecast (matplotlib)
            signal.plot.forecast(backend="plotly")     # forecast (plotly)
        """
        return PlotAccessor(self)

    @property
    def stat(self) -> StatAccessor:
        """Accessor for statistical tests.

        Usage::

            signal.stat.test_stationarity()               # ADF per column
            signal.stat.test_stationarity(method='kpss')   # KPSS per column
            signal.stat.test_seasonality()                 # per column
            signal.stat.test_causality()                   # all column pairs
        """
        return StatAccessor(self)

    @property
    def validation(self) -> ValidationAccessor:
        """Accessor for train/test split operations.

        Usage::

            signal.validation.split("2020-01-01")
            signal.train_data                        # .loc slice, computed on demand
            signal.test_data                         # .loc slice, computed on demand
            signal.validation.split("2020-01-01", n_splits_cv=5)
            signal.validation.cv_tseries
        """
        return self._validation

    @property
    def decompositions(self) -> dict:
        """Dict of named residuals (each a :class:`Signal`).

        Access a named decomposition via ``signal.decompositions["name"]``.
        """
        return self._decompositions

    @property
    def residual(self) -> "Signal":
        """Shorthand for the default decomposition (``decompositions["default"]``).

        Backward-compatible with ``signal.decompose()`` (no name).
        """
        return self._decompositions.get("default")

    @residual.setter
    def residual(self, value):
        self._decompositions["default"] = value

    @property
    def decomposition(self) -> "Decomposition":
        """Decomposition formula for the default residual.

        Backward-compatible property. For named decompositions use
        ``signal.get_decomposition(name)``.
        """
        if "default" not in self._decompositions:
            raise AttributeError("No decomposition. Call decompose() first.")
        return Decomposition(self._decompositions["default"]._ops)

    def get_decomposition(self, name: str = "default") -> "Decomposition":
        """Return the :class:`Decomposition` formula for a named residual.

        Parameters
        ----------
        name : str
            Name of the decomposition (as passed to ``decompose(name)``).
        """
        if name not in self._decompositions:
            raise AttributeError(
                f"No decomposition named {name!r}. Call decompose({name!r}) first."
            )
        return Decomposition(self._decompositions[name]._ops)

    @staticmethod
    def _check_nan_for_model(data: pd.DataFrame, model: "TimeSeriesModel", context: str) -> None:
        """Raise ValueError if *data* has NaN and *model* is not nan_safe."""
        if data.isna().any().any() and not model.nan_safe:
            n_nan = int(data.isna().sum().sum())
            cols = list(data.columns[data.isna().any()])
            raise ValueError(
                f"Cannot {context} model '{model.name}': data contains {n_nan} "
                f"NaN value(s) in column(s) {cols}. "
                f"{model.name} does not support NaN (nan_safe=False). "
                f"Handle missing values before fitting "
                f"(e.g. df.fillna(), df.dropna(), df.interpolate())."
            )

    def validation_split(self, timestamp: Union[int, str, pd.Timestamp], n_splits_cv=None) -> None:
        """
        Splits the series between train and test sets.

        Delegates to :meth:`signal.validation.split`.
        Also accessible via ``signal.validation.split(timestamp)``.

        Parameters
        ----------
        timestamp :
                Time index to split between train and test sets
        n_splits_cv : int, optional
                Number of folds for cross-validation
        """
        self._validation.split(timestamp, n_splits_cv)

    def apply_model(self,
                    model: object,
                    gridsearch: bool = False,
                    parameters: dict = None,
                    save_model: bool = False) -> None:
        """
        Applies a model to the series.

        Accepts any :class:`TimeSeriesModel` instance (``DartsModel``, ``LinearTrend``,
        ``AggregatedModel``, …) or a raw Darts model (auto-wrapped in ``DartsModel``).

        Fills the attribute models.

        Parameters
        ----------
        model
                Instance of a TimeSeriesModel or a raw Darts model.
        gridsearch : bool, optional
                Whether to perform a gridsearch (default is False). Only for Darts models.
        parameters : dict, optional
                Parameters to test if a gridsearch is performed (default is None)
        save_model : bool, optional
                Whether to save the model in a file in Signal.path (default is False).

        Returns
        -------
        None
        """
        if not isinstance(model, TimeSeriesModel):
            if gridsearch and parameters is None:
                raise ValueError("Please enter the parameters")
            gridsearch_params = parameters if gridsearch else None
            model = DartsModel(model, gridsearch_params=gridsearch_params)

        model = copy.deepcopy(model)
        self.models[model.name] = self._fit_and_predict(model)
        if save_model:
            persistence.save_model(self.path, model.name, self.models[model.name])
            persistence.save_common_data(self.path, self.train_data, self.test_data)

    def compute_scores(self, list_metrics: list[str] = None, axis=1) -> None:
        """
        Computes scores of models on test data.

        Fills the attribute models with the scores and the attribute performance_models.

        Parameters
        ----------
        list_metrics : list[str]
                List of name of metrics chosen in ['r2', 'mse', 'rmse', 'mape', 'smape', 'mae']
        axis : int, optional (default = 1)
                Whether to compute scores unit-wise (axis=1) or time-wise (axis=0)

        Returns
        -------
        None
        """
        if axis == 1:
            score_type = 'unit_wise'
        elif axis == 0:
            score_type = 'time_wise'
        else:
            raise ValueError('axis must be 0 or 1')
        if list_metrics is None:
            list_metrics = ['r2', 'mse', 'rmse', 'mape', 'smape', 'mae']
        call_metric = Metrics(self, list_metrics)
        for model in self.models.keys():
            self.models[model]['scores'][score_type] = call_metric.compute_scores(model, axis)
        self.performance_models[score_type] = call_metric.scores_comparison(axis)

    def _fit_and_predict(self, model: TimeSeriesModel) -> ModelResult:
        """Fit a TimeSeriesModel on train data and predict on test set."""
        self._check_nan_for_model(self.train_data, model, "fit")
        model.fit(self.train_data)
        predictions = model.reverse_transform(len(self.test_data))
        # Align prediction index with test_data to avoid index mismatch
        # (e.g. Darts may produce slightly different DatetimeIndex values)
        if len(predictions) == len(self.test_data):
            predictions.index = self.test_data.index
        return ModelResult(
            model=model,
            predictions=predictions,
            best_parameters=getattr(model, 'best_params_', None) or "default",
        )

    def _ensure_final_estimator(self, model_name: str) -> None:
        """Lazily refit the model on the full dataset for forecasting."""
        result = self.models[model_name]
        if result.final_estimator is None:
            self._check_nan_for_model(self.data, result.model, "refit")
            warnings.warn(f"Fitting model {model_name} on whole dataset...")
            result.final_estimator = copy.deepcopy(result.model)
            result.final_estimator.fit(self.data)

    def forecast(self, name: str, horizon: int, save_model: bool = False) -> None:
        """
        Generates forecasts for future dates.

        ``forecast(name, horizon)``
            *Direct path* — ``name`` is a model key in ``self.models``
            (registered via :meth:`apply_model`).  The model is refitted on
            the full dataset before forecasting.

        ``forecast("decomp__model", horizon)``
            *Decomposition path* — ``name`` contains ``__`` and is split into
            ``decomp_name`` and ``model_name``.  The model must have been
            trained on ``signal.decompositions[decomp_name]`` via
            :meth:`apply_model`.  The forecast is composed back through the
            decomposition formula and stored in
            ``signal.models["decomp__model"]``.

        Parameters
        ----------
        name : str
            Model key, or ``"decomp_name__model_name"`` for the decomposition
            path.
        horizon : int
            Number of steps to forecast.
        save_model : bool, optional
            Whether to persist the result to disk (default ``False``).
        """
        if '__' in name:
            # --- Decomposition path: forecast("decomp__model", horizon) ---
            decomp_name, model_name = name.split('__', 1)
            if decomp_name not in self._decompositions:
                raise ValueError(
                    f"No decomposition named {decomp_name!r}. "
                    f"Call decompose({decomp_name!r}) first."
                )
            decomp_signal = self._decompositions[decomp_name]
            if model_name not in decomp_signal.models:
                raise ValueError(
                    f"Model {model_name!r} has not been trained on "
                    f"decomposition {decomp_name!r}."
                )
            decomp_signal._ensure_final_estimator(model_name)
            result = decomp_signal.models[model_name]
            decomp = self.get_decomposition(decomp_name)

            # Compose test predictions back to the original signal space
            if result.predictions is not None:
                composed_pred = decomp.compose(DataCube(result.predictions)).data
                if len(composed_pred) == len(self.test_data):
                    composed_pred.index = self.test_data.index
            else:
                composed_pred = None

            # Compose forecast
            residual_forecast = result.final_estimator.reverse_transform(horizon)
            composed_forecast = decomp.compose(
                DataCube(residual_forecast), horizon=horizon).data

            self.models[name] = ModelResult(
                model=result.model,
                predictions=composed_pred,
                best_parameters=result.best_parameters,
                forecast=composed_forecast,
            )
            if save_model:
                persistence.save_model(self.path, name, self.models[name], suffix="final")
                persistence.save_common_data(self.path, self.train_data, self.test_data)

        else:
            # --- Direct path: forecast(model_name, horizon) ---
            if name not in self.models:
                raise ValueError(f'{name} has not been trained.')
            self._ensure_final_estimator(name)
            result = self.models[name]
            result.forecast = result.final_estimator.reverse_transform(horizon)
            if save_model:
                persistence.save_model(self.path, name, result, suffix="final")
                persistence.save_common_data(self.path, self.train_data, self.test_data)

    def _conf_interval_test(self, model_name: str, window_size: int = 6):
        if model_name not in self.models:
            raise AttributeError(f'{model_name} has not been fitted.')
        result = self.models[model_name]
        pred = result.predictions
        df_residuals = pd.DataFrame(index=pred.index, columns=pred.columns)
        std_values = {}
        for ref in pred.columns:
            errors = self.test_data[ref].values - pred[ref].values
            df_residuals[ref] = errors
            std_values[ref] = pd.Series(errors).rolling(window=window_size).std().values
        result.test_residuals = df_residuals
        result.test_confidence_interval = _build_ci(pred, std_values)

    def _conf_interval_forecast(self, model_name: str):
        if model_name not in self.models:
            raise AttributeError(f'{model_name} has not been fitted.')
        result = self.models[model_name]
        if result.forecast is None:
            raise AttributeError(f'No forecasts have been computed with model {model_name}.')
        pred = result.forecast
        std_values = {ref: np.std(result.test_residuals[ref])
                      for ref in pred.columns}
        result.forecast_confidence_interval = _build_ci(pred, std_values)

    def compute_conf_intervals(self, window_size: int = 10, save=False):
        if not self.models:
            raise AttributeError('No predictions have been found.')
        for model_ in self.models.keys():
            self._conf_interval_test(model_, window_size)
            if self.models[model_].forecast is not None:
                self._conf_interval_forecast(model_)
            if save:
                persistence.save_model(self.path, model_, self.models[model_])
                if self.models[model_].forecast is not None:
                    persistence.save_model(self.path, model_, self.models[model_], suffix="final")

    def get_saved_models(self) -> None:
        """Gets previously fitted models saved in joblib files in Signal.path and saves them in attribute models."""
        def set_train(data):
            self._train_data = data
        def set_test(data):
            self._test_data = data
        persistence.load_saved_models(self.path, self.models, set_train, set_test)
