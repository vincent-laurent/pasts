# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from abc import ABC, abstractmethod

import pandas as pd


class TimeSeriesModel(ABC):
    """Abstract base class for time series decomposition models.

    A model extracts a signal component (trend, seasonality, …) from a time
    series and provides the inverse operation for recomposition.

    Two families exist:

    - Trend components (e.g. :class:`LinearTrend`, :class:`NonParametricTrend`)
      — extract and restore signal components (trend, etc.).
    - :class:`DartsModel` — component backed by a Darts ML/DL estimator;
      applicable to any :class:`~pasts.core.datacube.DataCube` or DataFrame,
      whether it is the raw signal or a residual.

    Subclasses must implement :meth:`fit` and :meth:`reverse_transform`.
    :meth:`transform` is provided as a default (negation of ``reverse_transform``).

    Attributes
    ----------
    nan_safe : bool
        Whether ``fit()`` can handle NaN values in the input data.
        Defaults to ``False``.  Subclasses that handle NaN internally
        (e.g. via interpolation) should set this to ``True``.
    """

    nan_safe: bool = False
    _decomposition = None

    @property
    def name(self) -> str:
        """Model name used as key in ``Signal.models``."""
        return self.__class__.__name__

    @abstractmethod
    def fit(self, X: pd.DataFrame, covariates=None) -> "TimeSeriesModel":
        """Fit the model to the time series *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Target time series.
        covariates : :class:`~pasts.covariates.Covariates`, optional
            Past, future and/or static covariates (default ``None``).
        """

    @abstractmethod
    def reverse_transform(self, i: int) -> pd.DataFrame:
        """Return values to add back the component for recomposition.

        Parameters
        ----------
        i : int
            Length of output (negative for past, positive for future).
        """

    def compute_historical_residuals(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """Compute one-step-ahead historical residuals on the training set.

        Residuals are always returned in the same space as *train_data*
        (original signal space).  When a decomposition is attached, fitted
        values are recomposed before computing residuals.

        Default implementation: ``train_data - forecast(-n)``.
        This is exact for deterministic models (trends).  Stochastic models
        (e.g. :class:`DartsModel`) should override this to use proper
        one-step-ahead historical forecasts.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training data used during :meth:`fit`.

        Returns
        -------
        pd.DataFrame
            Residuals (may be shorter than *train_data* if the model cannot
            produce fitted values for all training points).
        """
        n = len(train_data)
        fitted = self.forecast(-n)
        if len(fitted) < n:
            train_data = train_data.iloc[-len(fitted):]
        if len(fitted) == len(train_data):
            fitted.index = train_data.index
        return train_data - fitted

    def forecast(self, horizon: int) -> pd.DataFrame:
        """Forecast the next *horizon* steps after the training data.

        When a decomposition is attached, the raw prediction (in residual
        space) is recomposed back to the original signal space.

        Parameters
        ----------
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        pd.DataFrame
        """
        raw = self.reverse_transform(horizon)
        if self._decomposition is not None:
            return self._decomposition.recompose(raw, horizon)
        return raw

    def transform(self, i: int) -> pd.DataFrame:
        """Return values to remove the component (negation of ``reverse_transform``).

        Parameters
        ----------
        i : int
            Length of output (negative for past, positive for future).
        """
        return -self.reverse_transform(i)
