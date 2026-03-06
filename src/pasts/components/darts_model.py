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

import pandas as pd

from pasts.core.base_model import TimeSeriesModel


def _filter_covariates_for_model(darts_model, covariates):
    """Build Darts covariate kwargs, warning when a type is unsupported.

    Parameters
    ----------
    darts_model : darts model instance
        The underlying Darts estimator.
    covariates : :class:`~pasts.covariates.Covariates` or None
        Covariates to filter.

    Returns
    -------
    dict
        Keyword arguments for Darts ``fit`` / ``predict`` / ``historical_forecasts``.
        Keys are ``'past_covariates'`` and/or ``'future_covariates'``
        (only those supported by *darts_model*).
    """
    if covariates is None or covariates.is_empty:
        return {}

    from darts import TimeSeries

    kwargs = {}
    if covariates.past is not None:
        if darts_model.supports_past_covariates:
            kwargs["past_covariates"] = TimeSeries.from_dataframe(covariates.past)
        else:
            warnings.warn(
                f"{darts_model.__class__.__name__} does not support past "
                f"covariates; they will be ignored.",
                UserWarning,
                stacklevel=3,
            )

    if covariates.future is not None:
        if darts_model.supports_future_covariates:
            kwargs["future_covariates"] = TimeSeries.from_dataframe(covariates.future)
        else:
            warnings.warn(
                f"{darts_model.__class__.__name__} does not support future "
                f"covariates; they will be ignored.",
                UserWarning,
                stacklevel=3,
            )

    return kwargs


class DartsModel(TimeSeriesModel):
    """A time series model backed by a Darts ML/DL estimator.

    Unlike parametric trend components (e.g. :class:`~pasts.components.trend.LinearTrend`),
    a ``DartsModel`` learns patterns from data rather than fitting a
    closed-form function of time.  It can be applied to any
    :class:`~pasts.core.datacube.DataCube` or ``pd.DataFrame`` —
    the raw signal, a residual after detrending, or any intermediate series.

    Parameters
    ----------
    model : darts.models base class instance
        Any fitted or unfitted Darts model
        (e.g. ``NaiveSeasonal()``, ``ExponentialSmoothing()``).
    gridsearch_params : dict, optional
        Parameter grid for gridsearch. When provided, :meth:`fit` runs
        ``model.gridsearch(...)`` before the normal fit. The best
        parameters are stored in :attr:`best_params_`.
    decomposition : :class:`~pasts.core.decomposition.Decomposition`, optional
        When provided, :meth:`fit` replays the decomposition on the
        training data first and fits the Darts model on the residual.
        :meth:`forecast` then recomposes predictions back to the
        original signal space.

    Notes
    -----
    ``reverse_transform(i > 0)`` calls ``model.predict(i)`` for future values.
    ``reverse_transform(i < 0)`` calls ``model.historical_forecasts`` to
    retrieve the |i| last in-sample one-step-ahead predictions.
    """

    def __init__(self, model, gridsearch_params=None, decomposition=None):
        self._model = model
        self._gridsearch_params = gridsearch_params
        self.best_params_ = None
        self._decomposition = decomposition

    @property
    def name(self) -> str:
        """Return the wrapped Darts model's class name."""
        return self._model.__class__.__name__

    def fit(self, X, covariates=None) -> "DartsModel":
        """Fit the Darts model on *X*.

        Parameters
        ----------
        X : pd.DataFrame or :class:`~pasts.core.datacube.DataCube`
            Training data. Both the original signal and any residual are valid.
        covariates : :class:`~pasts.covariates.Covariates`, optional
            Past, future and/or static covariates. Unsupported types are
            silently ignored (with a warning).

        Returns
        -------
        self
        """
        from pasts.core.datacube import DataCube
        if isinstance(X, DataCube):
            X = X.data

        if self._decomposition is not None:
            self._decomposition.fit(X, covariates=covariates)
            X = self._decomposition.residual

        if X.isna().any().any():
            n_nan = int(X.isna().sum().sum())
            cols = list(X.columns[X.isna().any()])
            raise ValueError(
                f"Training data contains {n_nan} NaN value(s) in column(s) {cols}. "
                f"Handle missing values before fitting (e.g. df.fillna(0), df.dropna(), "
                f"df.interpolate())."
            )
        if isinstance(X.index, pd.DatetimeIndex) and X.index.freq is None:
            inferred = pd.infer_freq(X.index)
            if inferred is not None:
                X = X.asfreq(inferred)

        freq_arg = X.index.freq if isinstance(X.index, pd.DatetimeIndex) else None
        from darts import TimeSeries
        self._train_series = TimeSeries.from_dataframe(X, fill_missing_dates=True, freq=freq_arg)

        # Attach static covariates to the TimeSeries object
        if covariates is not None and covariates.static is not None:
            if self._model.supports_static_covariates:
                self._train_series = self._train_series.with_static_covariates(
                    covariates.static
                )
            else:
                warnings.warn(
                    f"{self.name} does not support static covariates; "
                    f"they will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )

        # Build temporal covariate kwargs and store for predict/historical_forecasts
        self._cov_kwargs = _filter_covariates_for_model(self._model, covariates)

        self._per_column = False
        try:
            if self._gridsearch_params:
                print('Performing the gridsearch for', self.name, '...')
                best_model, self.best_params_, _ = self._model.gridsearch(
                    parameters=self._gridsearch_params,
                    series=self._train_series,
                    start=0.5,
                    forecast_horizon=5,
                    **self._cov_kwargs,
                )
                self._model = best_model
            self._model.fit(self._train_series, **self._cov_kwargs)
        except ValueError as e:
            if "univariate" in str(e).lower() and X.shape[1] > 1:
                warnings.warn(
                    f"{self.name} only supports univariate series; "
                    f"fitting one model per column ({X.shape[1]} columns).",
                    UserWarning,
                    stacklevel=2,
                )
                self._fit_per_column(X)
            else:
                raise
        self._n_train = len(self._train_series)
        return self

    def _fit_per_column(self, X):
        """Fit a separate model clone per column for univariate-only models."""
        from darts import TimeSeries

        self._per_column = True
        self._column_models = {}
        self._column_train_series = {}

        for col in X.columns:
            col_model = copy.deepcopy(self._model)
            col_series = TimeSeries.from_dataframe(X[[col]])

            if self._gridsearch_params:
                best_model, _, _ = col_model.gridsearch(
                    parameters=self._gridsearch_params,
                    series=col_series,
                    start=0.5,
                    forecast_horizon=5,
                    **self._cov_kwargs,
                )
                col_model = best_model

            col_model.fit(col_series, **self._cov_kwargs)
            self._column_models[col] = col_model
            self._column_train_series[col] = col_series

    def compute_historical_residuals(self, train_data):
        """Compute one-step-ahead historical forecast residuals on training data.

        Uses Darts' ``historical_forecasts()`` starting at 50 % of the series.
        Local models (e.g. ExponentialSmoothing) require ``retrain=True``;
        global models can use ``retrain=False`` for speed.

        Covariates stored during :meth:`fit` are forwarded automatically.

        Residuals are always returned in the original signal space.
        When a decomposition is attached, the Darts historical forecasts
        (in residual space) are recomposed back to original space before
        computing residuals against *train_data*.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training data in original signal space.

        Returns
        -------
        pd.DataFrame
            Residuals in original signal space, aligned to the covered
            portion of *train_data*.
        """
        if getattr(self, '_per_column', False):
            return self._compute_historical_residuals_per_column(train_data)

        from darts.models.forecasting.forecasting_model import (
            LocalForecastingModel,
        )
        retrain = isinstance(self._model, LocalForecastingModel)
        cov_kwargs = getattr(self, "_cov_kwargs", {})
        hf = self._model.historical_forecasts(
            self._train_series,
            start=0.5,
            forecast_horizon=1,
            retrain=retrain,
            **cov_kwargs,
        )
        hf_df = hf.to_dataframe()

        if self._decomposition is not None:
            hf_df = self._decomposition.recompose(hf_df, -len(hf_df))

        common_index = train_data.index.intersection(hf_df.index)
        if len(common_index) == 0:
            n_hf = len(hf_df)
            hf_df.index = train_data.index[-n_hf:]
            common_index = hf_df.index

        return train_data.loc[common_index] - hf_df.loc[common_index]

    def _compute_historical_residuals_per_column(self, train_data):
        """Per-column historical residuals for univariate-only models."""
        from darts.models.forecasting.forecasting_model import (
            LocalForecastingModel,
        )
        cov_kwargs = getattr(self, "_cov_kwargs", {})
        results = []
        for col in self._column_models:
            model = self._column_models[col]
            series = self._column_train_series[col]
            retrain = isinstance(model, LocalForecastingModel)
            hf = model.historical_forecasts(
                series,
                start=0.5,
                forecast_horizon=1,
                retrain=retrain,
                **cov_kwargs,
            )
            results.append(hf.to_dataframe())
        hf_df = pd.concat(results, axis=1)

        if self._decomposition is not None:
            hf_df = self._decomposition.recompose(hf_df, -len(hf_df))

        common_index = train_data.index.intersection(hf_df.index)
        if len(common_index) == 0:
            n_hf = len(hf_df)
            hf_df.index = train_data.index[-n_hf:]
            common_index = hf_df.index

        return train_data.loc[common_index] - hf_df.loc[common_index]

    def reverse_transform(self, i: int):
        """Return component values for recomposition.

        Covariates stored during :meth:`fit` are forwarded automatically.

        Parameters
        ----------
        i : int
            i > 0 : forecast the next *i* steps.
            i < 0 : return the last *|i|* one-step-ahead in-sample predictions.
        """
        if getattr(self, '_per_column', False):
            return self._reverse_transform_per_column(i)
        cov_kwargs = getattr(self, "_cov_kwargs", {})
        if i > 0:
            return self._model.predict(i, **cov_kwargs).to_dataframe()
        hf = self._model.historical_forecasts(
            self._train_series,
            start=self._n_train + i,
            forecast_horizon=1,
            retrain=False,
            **cov_kwargs,
        )
        return hf.to_dataframe()

    def _reverse_transform_per_column(self, i: int):
        """Per-column reverse_transform for univariate-only models."""
        cov_kwargs = getattr(self, "_cov_kwargs", {})
        results = []
        for col in self._column_models:
            model = self._column_models[col]
            if i > 0:
                col_df = model.predict(i, **cov_kwargs).to_dataframe()
            else:
                series = self._column_train_series[col]
                hf = model.historical_forecasts(
                    series,
                    start=self._n_train + i,
                    forecast_horizon=1,
                    retrain=False,
                    **cov_kwargs,
                )
                col_df = hf.to_dataframe()
            results.append(col_df)
        return pd.concat(results, axis=1)
