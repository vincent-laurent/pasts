# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pasts.core.base_model import TimeSeriesModel


class ParametricModel(TimeSeriesModel):
    """Abstract base for components defined as a parametric function of time f(t).

    After :meth:`fit`, a ``ParametricModel`` can evaluate its component for
    any time index — past or future — without re-training.

    Examples: :class:`Trend`, polynomial trend, Fourier seasonality.
    """


class Trend(ParametricModel):
    """
    A parametric model that removes and restores a linear trend.

    Attributes
    ----------
    time_index: numpy array
        TimeIndex converted to float index.
    features_: list[str]
        Columns of passed series.
    coef_: np.ndarray
        Trend coefficients, shape (n_columns, 1).
    intercept_: np.ndarray
        Trend intercepts, shape (n_columns,).
    t0_: float
        Last index value.

    Methods
    -------
    fit(X):
        Extracts trend from passed X series.
    transform(i):
        Computes DataFrame to remove trend from a series.
    reverse_transform(i):
        Computes DataFrame to add trend to a series.
    """

    def __init__(self):
        ...

    def fit(self, X: pd.DataFrame) -> "Trend":
        """
        Finds linear trend in passed series.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to detrend. Index must be a TimeIndex.

        Returns
        -------
        self
        """
        self.__origin_index = X.index
        self.time_index = X.index.to_numpy(dtype="float64") * 1e-17
        self.features_ = X.columns
        estimator = LinearRegression()
        estimator.fit(self.time_index.reshape(-1, 1), X.values)
        self.coef_ = estimator.coef_                    # shape (n_columns, 1)
        self.intercept_ = estimator.intercept_          # shape (n_columns,)
        self.t0_ = self.time_index[-1]
        return self

    def _from_i_to_vector(self, i: int) -> np.ndarray:
        """Return past (i<0) or future (i>0) float time vector of length |i|."""
        if i > 0:
            return self.t0_ + np.arange(i) * np.mean(np.diff(self.time_index))
        return self.time_index[i:]

    def _from_i_to_time_index(self, i: int) -> pd.DatetimeIndex:
        """Return past (i<0) or future (i>0) DatetimeIndex of length |i|."""
        if i > 0:
            return pd.date_range(
                start=self.__origin_index[-1], periods=i + 1, freq=self.__origin_index.freq
            )[1:]
        return self.__origin_index[i:]

    def _trend_values(self, i: int) -> np.ndarray:
        """Compute trend values; shape (n_columns, |i|)."""
        t = self._from_i_to_vector(i)
        return self.coef_ * t + self.intercept_.reshape(-1, 1)

    def _make_frame(self, i: int, sign: int) -> pd.DataFrame:
        """Build a signed trend DataFrame with time as index and features as columns."""
        vals = self._trend_values(i)
        idx = self._from_i_to_time_index(i)
        return pd.DataFrame(sign * vals.T, index=idx, columns=self.features_)

    def reverse_transform(self, i: int) -> pd.DataFrame:
        """
        Computes DataFrame to add trend to a series.
        Adds trend when added to a DataFrame of same shape.

        Parameters
        ----------
        i : int
            Length of output (negative for past, positive for future).
        """
        return self._make_frame(i, sign=1)
