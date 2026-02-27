# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import warnings

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

    Notes
    -----
    ``reverse_transform(i > 0)`` calls ``model.predict(i)`` for future values.
    ``reverse_transform(i < 0)`` calls ``model.historical_forecasts`` to
    retrieve the |i| last in-sample one-step-ahead predictions.
    """

    def __init__(self, model, gridsearch_params=None):
        self._model = model
        self._gridsearch_params = gridsearch_params
        self.best_params_ = None

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
        if X.isna().any().any():
            n_nan = int(X.isna().sum().sum())
            cols = list(X.columns[X.isna().any()])
            raise ValueError(
                f"Training data contains {n_nan} NaN value(s) in column(s) {cols}. "
                f"Handle missing values before fitting (e.g. df.fillna(0), df.dropna(), "
                f"df.interpolate())."
            )
        from darts import TimeSeries
        self._train_series = TimeSeries.from_dataframe(X)

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
        self._n_train = len(self._train_series)
        return self

    def compute_historical_residuals(self, train_data):
        """Compute one-step-ahead historical forecast residuals on training data.

        Uses Darts' ``historical_forecasts()`` starting at 50 % of the series.
        Local models (e.g. ExponentialSmoothing) require ``retrain=True``;
        global models can use ``retrain=False`` for speed.

        Covariates stored during :meth:`fit` are forwarded automatically.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training data (same as passed to :meth:`fit`).

        Returns
        -------
        pd.DataFrame
            Residuals aligned to the covered portion of *train_data*.
        """
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
