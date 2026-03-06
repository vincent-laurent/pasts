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
from typing import Union

import pandas as pd

from pasts.core.base_model import TimeSeriesModel
from pasts.core.datacube import DataCube, _make_future_index
from pasts.core.decomposition import DecompositionModel
from pasts.core.model_result import ModelResult
from pasts.covariates import Covariates, validate_covariates
from pasts.components.darts_model import DartsModel
from pasts.statistical_tests import StatAccessor
from pasts.validation import ValidationAccessor
from pasts.metrics import Metrics
from pasts import persistence
from pasts.visualization import PlotAccessor
from pasts.prediction_intervals import CIAccessor
from pasts.core.handle_nan import NaNHandler


class Signal(DataCube):
    """A class to represent a signal.

    Combines data storage, model fitting, scoring, and
    forecasting in a single orchestrator.

    Attributes
    ----------
    models : dict
        keys: model names, values: :class:`ModelResult` instances.
    """

    _metadata = [
        '_ops', 'path', '_properties', '_tests_stat',
        '_validation', 'models', '_decompositions',
        '_performance_models', '_covariates',
    ]

    # -------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------

    @staticmethod
    def _profiling(data: pd.DataFrame) -> dict:
        """Compute basic properties of the time series."""
        df = pd.DataFrame(data)
        return {'shape': df.shape,
                'types': df.dtypes,
                'is_univariate': df.shape[1] == 1,
                'nanSum': df.isnull().sum(),
                'quantiles': df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T}

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
                If None, no directory is created.
        """
        if path and not os.path.exists(path):
            os.makedirs(path)
        super().__init__(data)
        self.path = path
        self._properties = Signal._profiling(data)
        self._tests_stat = {}
        self._validation = ValidationAccessor(self)
        self.models = {}
        self._decompositions = {}
        self._performance_models = {}
        self._covariates = Covariates()

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # Data access: train/test, covariates, NaN handling
    # -------------------------------------------------------------------

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
    def covariates(self) -> Covariates:
        """Registered covariates (past, future, static)."""
        return self._covariates

    def set_covariates(
        self,
        past_covariates: pd.DataFrame = None,
        future_covariates: pd.DataFrame = None,
        static_covariates: pd.DataFrame = None,
    ) -> None:
        """Register covariates on the signal.

        Parameters
        ----------
        past_covariates : pd.DataFrame, optional
            Past covariates. Index must cover the signal's DatetimeIndex.
        future_covariates : pd.DataFrame, optional
            Future covariates. Index must cover the signal period and
            extend far enough for forecasting (validated at forecast time).
        static_covariates : pd.DataFrame, optional
            Static (time-invariant) covariates.
        """
        cov = Covariates(
            past=past_covariates,
            future=future_covariates,
            static=static_covariates,
        )
        validate_covariates(self.data.index, cov)
        self._covariates = cov

    @property
    def handle_nan(self) -> NaNHandler:
        """Accessor for NaN handling.

        Usage::

            signal.handle_nan.fill(value=0)
            signal.handle_nan.interpolate(max_consecutive=5)
            signal.handle_nan.extrapolate(method='ffill')
            signal.handle_nan.before_launch(value=0)
            signal.handle_nan.after_stops(value=0)

            # chainable
            signal.handle_nan.before_launch(0).interpolate(max_consecutive=5).after_stops(0)
        """
        return NaNHandler(self)

    # -------------------------------------------------------------------
    # Decomposition
    # -------------------------------------------------------------------

    def decompose(self, name: str = "default") -> None:
        """Create a named decomposition slot.

        The returned :class:`Decomposition` supports in-place operators
        (``-=``, ``+=``, …) and :meth:`~Decomposition.apply_model`.

        Parameters
        ----------
        name : str
            Decomposition name (default ``"default"``).
        """
        self._decompositions[name] = DecompositionModel()

    @property
    def decompositions(self) -> dict:
        """Named decompositions created by :meth:`decompose`."""
        return self._decompositions

    @property
    def residual(self):
        """Shortcut for ``decompositions["default"]``."""
        return self._decompositions.get("default")

    @residual.setter
    def residual(self, value):
        self._decompositions["default"] = value

    # -------------------------------------------------------------------
    # Accessors: plot, stat, validation, ci
    # -------------------------------------------------------------------

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
    def ci(self) -> CIAccessor:
        """Accessor for prediction-interval computation.

        Usage::

            signal.ci.compute()                                     # empirical
            signal.ci.compute(method="bootstrap", random_state=42)  # residual bootstrap
            signal.ci.compute(method="bootstrap_full", n_bootstrap=200)
        """
        return CIAccessor(self)

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # Modeling: fit, predict, aggregate
    # -------------------------------------------------------------------

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

    def _fit_and_predict(self, model: TimeSeriesModel) -> ModelResult:
        """Fit a TimeSeriesModel on train data and predict on test set."""
        self._check_nan_for_model(self.train_data, model, "fit")
        model.fit(self.train_data, covariates=self._covariates)
        predictions = model.forecast(len(self.test_data))
        # Align prediction index with test_data to avoid index mismatch
        # (e.g. Darts may produce slightly different DatetimeIndex values)
        if len(predictions) == len(self.test_data):
            predictions.index = self.test_data.index
        return ModelResult(
            estimator_on_train=model,
            predictions=predictions,
            best_parameters=getattr(model, 'best_params_', None) or "default",
            _data=self.data,
            _covariates=self._covariates,
        )

    def apply_model(self,
                    model: object,
                    gridsearch: bool = False,
                    parameters: dict = None,
                    save_model: bool = False,
                    decomposition: str = None) -> None:
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
        decomposition : str, optional
                Name of a decomposition created by :meth:`decompose`. When set,
                the model is trained on the residual and predictions are
                recomposed back to the original signal space.

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

        if decomposition is not None:
            decomp = self._decompositions[decomposition]
            decomp_copy = DecompositionModel()
            decomp_copy._ops = copy.deepcopy(decomp._ops)
            model._decomposition = decomp_copy
            name = f"{decomposition}__{model.name}"
        else:
            name = model.name

        self.models[name] = self._fit_and_predict(model)
        if save_model:
            if self.path is None:
                raise ValueError(
                    "Cannot save model: no path was provided when creating the Signal. "
                    "Pass a `path` argument to Signal() or set `save_model=False`."
                )
            persistence.save_model(self.path, name, self.models[name])
            persistence.save_common_data(self.path, self.train_data, self.test_data)

    # -------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # Refit & Forecasting
    # -------------------------------------------------------------------

    def refit(self, name: str = None) -> None:
        """Refit model(s) on the full dataset (train + test).

        Must be called before :meth:`forecast`.  Each model is deep-copied
        so that the original (train-only) estimator in
        ``models[name].estimator_on_train`` is preserved.

        Parameters
        ----------
        name : str, optional
            Model key to refit.  If ``None``, all models are refitted.
        """
        names = [name] if name else list(self.models.keys())
        for n in names:
            if n not in self.models:
                raise ValueError(f"Model '{n}' has not been trained.")
            result = self.models[n]
            result.estimator_on_all = copy.deepcopy(result.estimator_on_train)
            result.estimator_on_all.fit(self.data, covariates=self._covariates)

    def forecast(self, name: str, horizon: int, save_model: bool = False) -> pd.DataFrame:
        """Forecast *horizon* steps ahead.

        Before :meth:`refit`: uses the train-only model — the forecast
        starts at the end of *train_data* (covers the test period and
        potentially beyond).

        After :meth:`refit`: uses the full-data model — the forecast
        starts at the end of the full dataset (true future).

        Parameters
        ----------
        name : str
            Model key (as stored in ``signal.models``).
        horizon : int
            Number of steps to forecast.
        save_model : bool, optional
            Whether to persist the result to disk (default ``False``).

        Returns
        -------
        pd.DataFrame
        """
        if not self._covariates.is_empty:
            validate_covariates(self.data.index, self._covariates,
                                forecast_horizon=horizon)
        if name not in self.models:
            raise ValueError(f'{name} has not been trained.')
        result = self.models[name]
        estimator = result.estimator_on_all if result.estimator_on_all is not None else result.estimator_on_train
        result.forecast_data = estimator.forecast(horizon)
        # Correct forecast index — Darts may infer wrong timestamps.
        # Mirrors the alignment done for test predictions in _fit_and_predict.
        anchor_idx = self.data.index if result.estimator_on_all is not None else self.train_data.index
        correct_idx = _make_future_index(anchor_idx, horizon)
        if len(result.forecast_data) == horizon:
            result.forecast_data.index = correct_idx
        if save_model:
            if self.path is None:
                raise ValueError(
                    "Cannot save model: no path was provided when creating the Signal. "
                    "Pass a `path` argument to Signal() or set `save_model=False`."
                )
            persistence.save_model(self.path, name, result, suffix="final")
            persistence.save_common_data(self.path, self.train_data, self.test_data)
        return result.forecast_data

    # -------------------------------------------------------------------
    # Confidence intervals
    # -------------------------------------------------------------------

    def compute_conf_intervals(self, **kwargs):
        """Backward-compatible delegate to :meth:`signal.ci.compute`."""
        self.ci.compute(**kwargs)

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def get_saved_models(self) -> None:
        """Gets previously fitted models saved in joblib files in Signal.path and saves them in attribute models."""
        def set_train(data):
            self._train_data = data
        def set_test(data):
            self._test_data = data
        persistence.load_saved_models(self.path, self.models, set_train, set_test)
