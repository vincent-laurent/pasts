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
from pasts.validation import Validation
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
        self._train_data = None
        self._test_data = None
        self.models = {}
        self._performance_models = {}
        self._residual = None
        self._learned_model = None

    @property
    def train_data(self):
        """Train set as a pandas dataframe"""
        return self._train_data

    @property
    def test_data(self):
        """Test set as a pandas dataframe"""
        return self._test_data

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
        if self._train_data is not None or self._test_data is not None:
            self._train_data = None
            self._test_data = None
            warnings.warn(
                "Train/test split has been reset after handle_nan(). "
                "Call validation_split() again.",
                UserWarning,
            )

    def decompose(self) -> None:
        """Initialize the residual as a copy of the signal data.

        After calling this method, operate on ``signal.residual`` to build
        the decomposition (e.g. ``signal.residual -= LinearTrend().fit(signal.data)``).
        The residual is itself a :class:`Signal`, so all analysis methods
        (``apply_model``, ``validation_split``, etc.) can be used on it directly.
        """
        self._residual = Signal(self.data.copy())

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
    def residual(self) -> "Signal":
        """Current residual (a :class:`Signal`)."""
        return self._residual

    @residual.setter
    def residual(self, value):
        self._residual = value

    @property
    def decomposition(self) -> "Decomposition":
        """Decomposition formula derived from operations applied to the residual."""
        if self._residual is None:
            raise AttributeError("No decomposition. Call decompose() first.")
        return Decomposition(self._residual._ops)

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

    def learn(self, model: object) -> None:
        """Fit a forecasting model on this signal's data.

        Intended for the terminal step of a decomposition workflow:
        after stripping components from the residual, fit a model that
        will predict the residual.

        Parameters
        ----------
        model : TimeSeriesModel or raw Darts model
            The model to fit. Raw Darts models are auto-wrapped in DartsModel.
        """
        if not isinstance(model, TimeSeriesModel):
            model = DartsModel(model)
        model = copy.deepcopy(model)
        self._check_nan_for_model(self.data, model, "learn")
        model.fit(self.data)
        self._learned_model = model

    def validation_split(self, timestamp: Union[int, str, pd.Timestamp], n_splits_cv=None) -> None:
        """
        Splits the series between train and test sets.

        If n_splits_cv is filled, yields train and test indices for cross-validation.

        Fills the attributes train_data and test_data.

        Parameters
        ----------
        timestamp :
                Time index to split between train and test sets
        n_splits_cv : int, optional
                Number of folds for cross-validation

        Returns
        -------
        None
        """
        call_validation = Validation(self.data)
        call_validation.split_cv(timestamp, n_splits_cv)
        if call_validation.train_data.shape[0] < 2:
            raise ValueError("Train set is empty or too small.")
        self._train_data = call_validation.train_data
        self._test_data = call_validation.test_data

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

    def forecast(self, model_name: str = None, horizon: int = None, save_model: bool = False) -> None:
        """
        Generates forecasts for future dates.

        Fills models attribute with a 'forecast' key and a 'final_estimator' key.

        Parameters
        ----------
        model_name : str, optional
                Name of a model previously fitted via apply_model(). If None and a
                model has been learned on the residual (via ``residual.learn()``),
                that model is used automatically with decomposition recomposition.
        horizon : int
                Horizon of prediction.
        save_model : bool, optional
                Whether to save the model in a file in Signal.path (default is False).

        Returns
        -------
        None
        """
        # --- Decomposition path: model learned on the residual ---
        if model_name is None:
            if (self._residual is None
                    or self._residual._learned_model is None):
                raise ValueError(
                    "No model_name provided and no model has been "
                    "learned on the residual. Call residual.learn() first.")
            model = self._residual._learned_model
            forecast_df = model.reverse_transform(horizon)
            forecast_df = self.decomposition.compose(
                DataCube(forecast_df), horizon=horizon).data
            learned_name = model.name
            if learned_name not in self.models:
                self.models[learned_name] = ModelResult(model=model)
            self.models[learned_name].forecast = forecast_df
            return

        # --- Direct path (uniform for all models including AggregatedModel) ---
        if model_name not in self.models:
            raise ValueError(f'{model_name} has not been trained.')
        self._ensure_final_estimator(model_name)
        result = self.models[model_name]
        result.forecast = result.final_estimator.reverse_transform(horizon)
        if save_model:
            persistence.save_model(self.path, model_name, result, suffix="final")
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
