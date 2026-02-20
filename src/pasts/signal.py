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
import warnings
from typing import Union
import joblib
import os
import glob
import re

import numpy as np
import pandas as pd
from darts import TimeSeries
from pasts.core.datacube import DataCube
from pasts.core.decomposition import Decomposition
from pasts.components.aggregated_model import AggregatedModel, _weighted_aggregate
from pasts.statistical_tests import TestStatistics
from pasts.validation import Validation
from pasts.metrics import Metrics

# Maps non-stationary test types to their canonical test name (used as dict key)
_TEST_NAMES = {'seasonality': 'check_seasonality', 'causality': 'grangercausalitytests'}


def profiling(data: pd.DataFrame) -> dict:
    """
    Finds some properties about time series.

    Parameters
    ----------
    data: pd.Dataframe
        Dataframe of time series with time as index and entities as columns.

    Returns
    -------
    Dictionary of properties of passed dataset.
    """
    return {'shape': data.shape,
            'types': data.dtypes,
            'is_univariate': data.shape[1] == 1,
            'nanSum': data.isnull().sum(),
            'quantiles': data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T}


def _build_ci(pred, std_values: dict) -> pd.DataFrame:
    """Build a confidence-interval DataFrame from a predictions TimeSeries and per-column std values."""
    df_itv = pd.DataFrame(index=pred.time_index, columns=pred.columns)
    weights = np.arange(1, len(pred) + 1, dtype=float)
    for ref in pred.columns:
        vals = pred[ref].values()[:, 0]
        itv_inf = vals - 1.96 * std_values[ref] * np.sqrt(weights)
        itv_sup = vals + 1.96 * std_values[ref] * np.sqrt(weights)
        df_itv[ref] = list(zip(itv_inf, itv_sup))
    return df_itv


class Signal(DataCube):
    """
    A class to represent a signal.

    Attributes
    ----------
    models : dict
        keys: models applied on the series
        values: dictionary of predictions and best parameters

    Methods
    -------
    apply_stat_test(type_test, test_stat_name = None, *args, **kwargs):
        Applies statistical test to the univariate or multivariate series.
    validation_split(timestamp, n_splits_cv = None):
        Splits the series between train and test sets.
    apply_model(model, gridsearch = False, parameters = None):
        Applies statistical, machine learning of deep learning model to the series.
    compute_scores(list_metrics: list[str] = None, axis=1)
        Computes scores of models on test data.
    apply_aggregated_model(list_models, refit=False)
        Aggregates a given list of models according to their performance on test data.
    forecast(model_name: str, horizon: int)
        Generates forecasts for future dates.
    """

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
        self.__properties = profiling(data)
        self.__tests_stat = {}
        self.__train_data = None
        self.__test_data = None
        self.models = {}
        self.__performance_models = {}
        self._residual = None

    @property
    def rest_data(self):
        """Residual data: residual if decomposition exists, otherwise raw data."""
        if self._residual is not None:
            return self._residual.data
        return self.data

    @property
    def train_data(self):
        """Train set as a pandas dataframe"""
        return self.__train_data

    @property
    def test_data(self):
        """Test set as a pandas dataframe"""
        return self.__test_data

    @property
    def rest_train_data(self):
        """Residual train data: residual if decomposition exists, otherwise train data."""
        if self._residual is not None:
            return self._residual.data.reindex(self.__train_data.index).dropna()
        return self.__train_data

    @property
    def properties(self):
        """Dictionary of properties of the signal"""
        return self.__properties

    @property
    def tests_stat(self):
        """Dictionary of statistical tests applied on data"""
        return self.__tests_stat

    @property
    def performance_models(self):
        """Dictionary containing a maximum of 2 other dictionaries for unit-wise or time-wise scores. Dictionaries
        contain a dataframe for each scorer, with scores computed for all models."""
        return self.__performance_models

    def decompose(self) -> None:
        """Initialize the residual as a copy of the signal data.

        After calling this method, operate on ``signal.residual`` to build
        the decomposition (e.g. ``signal.residual -= Trend().fit(signal.data)``).
        The residual is itself a :class:`Signal`, so all analysis methods
        (``apply_model``, ``validation_split``, etc.) can be used on it directly.
        """
        self._residual = Signal(self.data.copy())

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

    def apply_stat_test(self, type_test: str, test_stat_name: str = None, *args, **kwargs) -> None:
        """
        Applies statistical test to the univariate or multivariate series.

        Fills the attribute tests_stat.

        Parameters
        ----------
        type_test : str
                Type of test to be applied: stationary or seasonality for univariate series, causality for multivariate
        test_stat_name : str, optional
                adfuller or kpss if type_test is stationary (default is adfuller)
                ignored if type_test is seasonality or causality

        Returns
        -------
        None
        """
        call_test = TestStatistics(self)
        if type_test == 'stationary' and test_stat_name is None:
            test_stat_name = 'adfuller'
        elif type_test != 'stationary':
            test_stat_name = _TEST_NAMES[type_test]
        self.tests_stat[f"{type_test}: {test_stat_name}"] = call_test.apply(type_test, test_stat_name,
                                                                            *args, **kwargs)

    def validation_split(self, timestamp: Union[int, str, pd.Timestamp], n_splits_cv=None) -> None:
        """
        Splits the series between train and test sets.

        If n_splits_cv is filled, yields train and test indices for cross-validation.

        Fills the attributes train_data, test_data and rest_train_data (with train_data per default)

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
        self.__train_data = call_validation.train_data
        self.__test_data = call_validation.test_data

    def apply_model(self,
                    model: object,
                    gridsearch: bool = False,
                    parameters: dict = None,
                    save_model: bool = False) -> None:
        """
        Applies statistical, machine learning of deep learning model to the series.

        Fills the attribute models.

        Parameters
        ----------
        model
                Instance of a model from darts. Will be refitted even if it has already been.
        gridsearch : bool, optional
                Whether to perform a gridsearch (default is False)
        parameters : dict, optional
                Parameters to test if a gridsearch is performed (default is None)
        save_model : bool, optional
                Whether to save the model in a file in Signal.path (default is False).
                When True, also saves data, train and test set, and transformed data (if it exists).
                If the same model with different parameters has previously been saved from the same Signal object,
                the file will be overwritten.

        Returns
        -------
        None
        """
        self.models[model.__class__.__name__] = self._fit_and_predict(copy.deepcopy(model), gridsearch, parameters)
        if save_model:
            joblib.dump(self.models[model.__class__.__name__], os.path.join(self.path,
                                                                            f'{model.__class__.__name__}_train_jlib'))
            joblib.dump(self.test_data, os.path.join(self.path, 'test_data_jlib'))
            joblib.dump(self.train_data, os.path.join(self.path, 'train_data_jlib'))

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

    def apply_aggregated_model(self, list_models: list[object], refit: bool = False, save_model: bool = False) -> None:
        """
        Aggregates a given list of models according to their performance on test data.

        Creates a new model "AggregatedModel" in the attribute models.

        Parameters
        ----------
        list_models :
                List of instances of models.
        refit : bool, optional
                Whether to refit estimators even if they were previously fitted (default is False).
                Ignored for estimators not previously fitted.
        save_model : bool, optional
                Whether to save the model in a file in Signal.path (default is False).
                When True, also saves data, train and test set, and transformed data (if it exists).
                If the same model with different parameters has previously been saved from the same Signal object,
                the file will be overwritten.

        Returns
        -------
        None
        """
        dict_models = {model.__class__.__name__: model for model in list_models}
        if refit:
            for model in dict_models.values():
                self.apply_model(model)
        else:
            for model_name, model in dict_models.items():
                if model_name not in self.models:
                    warnings.warn(f'{model_name} has not yet been fitted. Fitting {model_name}...')
                    self.apply_model(model)
                    if save_model:
                        joblib.dump(self.models[model_name], os.path.join(self.path, f'{model_name}_train_jlib'))
        predictions = {
            m: self.models[m]['predictions'].to_dataframe().copy()
            for m in dict_models
        }
        weights = AggregatedModel.compute_weights(predictions, self.test_data.copy())
        df_ag = _weighted_aggregate(predictions, weights)
        self.models['AggregatedModel'] = {
            'predictions': TimeSeries.from_dataframe(df_ag),
            'weights': weights,
            'models': dict_models,
            'scores': {'unit_wise': {}, 'time_wise': {}},
        }
        if save_model:
            joblib.dump(self.models['AggregatedModel'], os.path.join(self.path, 'AggregatedModel_train_jlib'))
            joblib.dump(self.test_data, os.path.join(self.path, 'test_data_jlib'))
            joblib.dump(self.train_data, os.path.join(self.path, 'train_data_jlib'))

    def _fit_and_predict(self, model, gridsearch=False, parameters=None) -> dict:
        """Fit a Darts model on train data and predict on test set."""
        train_tseries = TimeSeries.from_dataframe(self.rest_train_data)
        if gridsearch:
            if parameters is None:
                raise ValueError("Please enter the parameters")
            print('Performing the gridsearch for', model.__class__.__name__, '...')
            best_model, best_parameters, _ = model.gridsearch(
                parameters=parameters, series=train_tseries,
                start=0.5, forecast_horizon=5
            )
            model = best_model
        else:
            best_parameters = "default"

        model.fit(train_tseries)
        forecast = model.predict(len(self.test_data))
        if self._residual is not None:
            forecast = TimeSeries.from_dataframe(
                self.decomposition.compose(DataCube(forecast.to_dataframe())).data
            )
        return {
            'predictions': forecast,
            'best_parameters': best_parameters,
            'scores': {'unit_wise': {}, 'time_wise': {}},
            'estimator': model,
        }

    def _fit_on_full_data(self, model_name: str):
        """Refit model on full dataset. Requires prior fit on train set."""
        if model_name not in self.models:
            raise AttributeError(f'{model_name} has not been fitted.')
        model = self.models[model_name]['estimator']
        train_temp = TimeSeries.from_dataframe(self.rest_data)
        model.fit(train_temp)
        return model

    def _ensure_final_estimator(self, model_name: str) -> None:
        if 'final_estimator' not in self.models[model_name]:
            warnings.warn(f"Fitting model {model_name} on whole dataset...")
            self.models[model_name]['final_estimator'] = self._fit_on_full_data(model_name)

    def _save_common_data(self) -> None:
        joblib.dump(self.test_data, os.path.join(self.path, 'test_data_jlib'))
        joblib.dump(self.train_data, os.path.join(self.path, 'train_data_jlib'))

    def forecast(self, model_name: str, horizon: int, save_model: bool = False) -> None:
        """
        Generates forecasts for future dates.

        Fills models attribute with a 'forecast' key and a 'final estimator' key.

        Parameters
        ----------
        model_name : str
                Name of a model. If AggregatedModel, forecasts will be computed with the models included in the
                aggregation.
        horizon : int
                Horizon of prediction.
        save_model : bool, optional
                Whether to save the model in a file in Signal.path (default is False).
                When True, also saves data, train and test set, and transformed data (if it exists).
                If the same model with different parameters has previously been saved from the same Signal object,
                the file will be overwritten.

        Returns
        -------
        None
        """
        if model_name != 'AggregatedModel':
            if model_name not in self.models:
                raise ValueError(f'{model_name} has not been trained.')
            self._ensure_final_estimator(model_name)
            self.models[model_name]['forecast'] = self.models[model_name]['final_estimator'].predict(horizon)
            if save_model:
                joblib.dump(self.models[model_name], os.path.join(self.path, f'{model_name}_final_jlib'))
                self._save_common_data()
            return

        if 'AggregatedModel' not in self.models:
            raise ValueError('Aggregated Model has not been trained. Use method apply_aggregated_model first.')
        for model in self.models['AggregatedModel']['models']:
            self._ensure_final_estimator(model)
            self.models[model]['forecast'] = self.models[model]['final_estimator'].predict(horizon)
            if save_model:
                joblib.dump(self.models[model], os.path.join(self.path, f'{model}_final_jlib'))

        dict_models = self.models['AggregatedModel']['models']
        weights = self.models['AggregatedModel']['weights']
        forecast_predictions = {
            m: self.models[m]['forecast'].to_dataframe()
            for m in dict_models
        }
        self.models['AggregatedModel']['forecast'] = TimeSeries.from_dataframe(
            _weighted_aggregate(forecast_predictions, weights)
        )
        if save_model:
            joblib.dump(self.models['AggregatedModel'], os.path.join(self.path, 'AggregatedModel_final_jlib'))
            self._save_common_data()

    def _conf_interval_test(self, model_name: str, window_size: int = 6):
        if model_name not in self.models:
            raise AttributeError(f'{model_name} has not been fitted.')
        pred = self.models[model_name]['predictions']
        df_residuals = pd.DataFrame(index=pred.time_index, columns=pred.columns)
        std_values = {}
        for ref in pred.columns:
            errors = self.test_data[ref].values - pred[ref].values()[:, 0]
            df_residuals[ref] = errors
            std_values[ref] = pd.Series(errors).rolling(window=window_size).std().values
        self.models[model_name]['test_residuals'] = df_residuals
        self.models[model_name]['test_confidence_interval'] = _build_ci(pred, std_values)

    def _conf_interval_forecast(self, model_name: str):
        if model_name not in self.models:
            raise AttributeError(f'{model_name} has not been fitted.')
        if 'forecast' not in self.models[model_name]:
            raise AttributeError(f'No forecasts have been computed with model {model_name}.')
        pred = self.models[model_name]['forecast']
        std_values = {ref: np.std(self.models[model_name]['test_residuals'][ref])
                      for ref in pred.columns}
        self.models[model_name]['forecast_confidence_interval'] = _build_ci(pred, std_values)

    def compute_conf_intervals(self, window_size: int = 10, save=False):
        if not self.models:
            raise AttributeError('No predictions have been found.')
        for model_ in self.models.keys():
            self._conf_interval_test(model_, window_size)
            if 'forecast' in self.models[model_]:
                self._conf_interval_forecast(model_)
            if save:
                joblib.dump(self.models[model_], os.path.join(self.path, f'{model_}_train_jlib'))
                if 'forecast' in self.models[model_]:
                    joblib.dump(self.models[model_], os.path.join(self.path, f'{model_}_final_jlib'))

    def _load_saved_file(self, file: str, filename: str) -> None:
        match_data = re.search(r'(.+)_data_jlib', filename)
        if match_data:
            name = match_data.group(1)
            if name == 'test':
                self.__test_data = joblib.load(file)
            elif name == 'train':
                self.__train_data = joblib.load(file)
            return

        match_final = re.search(r'(.+)_final_jlib', filename)
        if match_final:
            self.models[match_final.group(1)] = joblib.load(file)
            return

        match_train = re.search(r'(.+)_train_jlib', filename)
        if match_train:
            name = match_train.group(1)
            if name not in self.models or 'forecast' not in self.models[name]:
                self.models[name] = joblib.load(file)
            return

        warnings.warn(f"File {filename} does not correspond to a saved model.")

    def get_saved_models(self) -> None:
        """
        Gets previously fitted models saved in joblib files in Signal.path and saves them in attribute models.
        """
        files = glob.glob(os.path.join(self.path, "*jlib"))
        if not files:
            warnings.warn("No saved models were found.")
            return
        for file in files:
            self._load_saved_file(file, os.path.basename(file))
