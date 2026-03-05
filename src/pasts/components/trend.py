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
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL

from pasts.core.base_model import TimeSeriesModel
from pasts.core.datacube import _make_future_index


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _handle_nan(X: pd.DataFrame):
    """Interpolate NaN for algorithms that cannot handle them.

    Returns the filled DataFrame and the boolean mask of original NaN positions.
    """
    mask = X.isna()
    if not mask.any().any():
        return X, mask
    filled = X.interpolate(method='linear').bfill().ffill()
    return filled, mask


def _restore_nan(trend: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Put NaN back where the original data had NaN."""
    if mask.any().any():
        trend = trend.copy()
        trend[mask] = np.nan
    return trend


# ---------------------------------------------------------------------------
# LinearTrend — parametric (OLS)
# ---------------------------------------------------------------------------

class LinearTrend(TimeSeriesModel):
    """
    A parametric model that removes and restores a linear trend.

    NaN values in the input are interpolated before fitting the OLS
    regression (via ``_handle_nan``).  This is safe because a few
    interpolated points do not significantly affect a linear fit.

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

    nan_safe = True

    def __init__(self, lags: int = None):
        self.lags = lags

    def fit(self, X: pd.DataFrame, covariates=None) -> "LinearTrend":
        """
        Finds linear trend in passed series.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to detrend. Index must be a TimeIndex.
        covariates : ignored
            Accepted for interface compatibility.

        Returns
        -------
        self
        """
        self.__origin_index = X.index
        self.time_index = X.index.to_numpy(dtype="float64") * 1e-17
        self.features_ = X.columns
        x_filled, _ = _handle_nan(X)

        if self.lags is not None:
            fit_t = self.time_index[-self.lags:]
            fit_v = x_filled.values[-self.lags:]
        else:
            fit_t = self.time_index
            fit_v = x_filled.values

        estimator = LinearRegression()
        estimator.fit(fit_t.reshape(-1, 1), fit_v)
        self.coef_ = estimator.coef_                    # shape (n_columns, 1)
        self.intercept_ = estimator.intercept_          # shape (n_columns,)
        self.t0_ = self.time_index[-1]
        return self

    def _from_i_to_vector(self, i: int) -> np.ndarray:
        """Return past (i<0) or future (i>0) float time vector of length |i|."""
        if i > 0:
            return self.t0_ + np.arange(1, i + 1) * np.mean(np.diff(self.time_index))
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


# ---------------------------------------------------------------------------
# NonParametricTrend — base for non-parametric methods
# ---------------------------------------------------------------------------

class NonParametricTrend(TimeSeriesModel):
    """Base class for non-parametric trend extraction methods.

    Subclasses must call :meth:`_store_trend` at the end of their ``fit()``
    implementation.  The base class provides a generic ``reverse_transform``
    that returns historical trend values (``i < 0``) or extrapolated future
    values (``i > 0``).

    NaN values are handled via ``_handle_nan`` / ``_restore_nan``:
    input is interpolated for the algorithm, then NaN positions are
    restored in the output trend.

    Parameters
    ----------
    extrapolation : str
        Strategy for future values: ``'constant'`` (repeat last trend value)
        or ``'linear'`` (linear extrapolation from stored trend).
        Ignored when *forecast_model* is provided.
    forecast_model : object, optional
        A forecasting model used to extrapolate the trend into the future.
        Accepts any Darts model (e.g. ``ExponentialSmoothing()``) or a
        :class:`TimeSeriesModel` instance.  When provided, the model is
        fitted on the estimated trend and used for future extrapolation
        instead of the simple ``extrapolation`` strategy.
    """

    nan_safe = True

    def __init__(self, extrapolation: str = 'constant', forecast_model=None):
        self._extrapolation = extrapolation
        self._forecast_model = forecast_model
        self._fitted_forecast_model = None

    def _fit_forecast_model(self, trend: pd.DataFrame):
        """Fit the forecast model on the estimated trend."""
        import copy
        from pasts.components.darts_model import DartsModel

        model = copy.deepcopy(self._forecast_model)
        if not isinstance(model, TimeSeriesModel):
            model = DartsModel(model)
        trend_clean = trend.interpolate(method='linear').bfill().ffill()
        model.fit(trend_clean)
        return model

    def _store_trend(self, X: pd.DataFrame, trend_values: pd.DataFrame):
        """Store the extracted trend and metadata after fitting."""
        self._trend = trend_values
        self._origin_index = X.index
        self._features = X.columns
        if self._forecast_model is not None:
            self._fitted_forecast_model = self._fit_forecast_model(trend_values)

    def reverse_transform(self, i: int) -> pd.DataFrame:
        if i < 0:
            return self._trend.iloc[i:]
        # Future extrapolation via forecast model
        if self._fitted_forecast_model is not None:
            forecast = self._fitted_forecast_model.reverse_transform(i)
            # Anchor to last training trend value for continuity
            offset = self._trend.iloc[-1].values - forecast.iloc[0].values
            return forecast + offset
        # Fallback: simple strategies
        future_idx = _make_future_index(self._origin_index, i)
        if self._extrapolation == 'constant':
            last = self._trend.iloc[-1].values
            arr = np.tile(last, (i, 1))
            return pd.DataFrame(arr, index=future_idx, columns=self._features)
        if self._extrapolation == 'linear':
            future_t = np.arange(len(self._trend), len(self._trend) + i, dtype=float).reshape(-1, 1)
            arr = np.empty((i, len(self._features)))
            for j, col in enumerate(self._features):
                vals = self._trend[col].dropna().values
                t_valid = np.arange(len(vals), dtype=float).reshape(-1, 1)
                reg = LinearRegression().fit(t_valid, vals)
                arr[:, j] = reg.predict(future_t)
            return pd.DataFrame(arr, index=future_idx, columns=self._features)
        raise ValueError(f"Unknown extrapolation strategy: {self._extrapolation!r}")


# ---------------------------------------------------------------------------
# Concrete non-parametric trend methods
# ---------------------------------------------------------------------------

class MovingAverageTrend(NonParametricTrend):
    """Trend extraction by rolling mean.

    Parameters
    ----------
    window : int
        Window size for the rolling mean.
    center : bool
        Whether to center the rolling window (default ``True``).
    extrapolation : str
        Future extrapolation strategy (``'constant'`` or ``'linear'``).
    forecast_model : object, optional
        Forecasting model for trend extrapolation (see :class:`NonParametricTrend`).
    """

    def __init__(self, window: int, center: bool = True, extrapolation: str = 'constant',
                 forecast_model=None):
        super().__init__(extrapolation, forecast_model)
        self.window = window
        self.center = center

    def fit(self, X: pd.DataFrame, covariates=None) -> "MovingAverageTrend":
        trend = X.rolling(window=self.window, center=self.center, min_periods=1).mean()
        self._store_trend(X, trend)
        return self


class HPFilterTrend(NonParametricTrend):
    """Trend extraction via the Hodrick-Prescott filter.

    Parameters
    ----------
    lamb : float
        Smoothing parameter (default 1600, standard for quarterly data).
    extrapolation : str
        Future extrapolation strategy (``'constant'`` or ``'linear'``).
    forecast_model : object, optional
        Forecasting model for trend extrapolation (see :class:`NonParametricTrend`).
    """

    def __init__(self, lamb: float = 1600, extrapolation: str = 'constant',
                 forecast_model=None):
        super().__init__(extrapolation, forecast_model)
        self.lamb = lamb

    def fit(self, X: pd.DataFrame, covariates=None) -> "HPFilterTrend":
        x_filled, mask = _handle_nan(X)
        trend_data = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        for col in X.columns:
            _, trend_col = hpfilter(x_filled[col].values, lamb=self.lamb)
            trend_data[col] = trend_col
        trend_data = _restore_nan(trend_data, mask)
        self._store_trend(X, trend_data)
        return self


class STLTrend(NonParametricTrend):
    """Trend extraction via STL (Seasonal and Trend decomposition using Loess).

    Parameters
    ----------
    period : int
        Seasonal period of the data.
    extrapolation : str
        Future extrapolation strategy (``'constant'`` or ``'linear'``).
    forecast_model : object, optional
        Forecasting model for trend extrapolation (see :class:`NonParametricTrend`).
    """

    def __init__(self, period: int, extrapolation: str = 'constant',
                 forecast_model=None):
        super().__init__(extrapolation, forecast_model)
        self.period = period

    def fit(self, X: pd.DataFrame, covariates=None) -> "STLTrend":
        x_filled, mask = _handle_nan(X)
        trend_data = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        for col in X.columns:
            stl = STL(x_filled[col], period=self.period)
            res = stl.fit()
            trend_data[col] = res.trend
        trend_data = _restore_nan(trend_data, mask)
        self._store_trend(X, trend_data)
        return self


class EMDTrend(NonParametricTrend):
    """Trend extraction via Empirical Mode Decomposition.

    The last IMF (Intrinsic Mode Function) is taken as the trend.

    Requires the ``PyEMD`` package (``pip install EMD-signal``).

    Parameters
    ----------
    extrapolation : str
        Future extrapolation strategy (``'constant'`` or ``'linear'``).
    forecast_model : object, optional
        Forecasting model for trend extrapolation (see :class:`NonParametricTrend`).
    """

    def __init__(self, extrapolation: str = 'constant', forecast_model=None):
        super().__init__(extrapolation, forecast_model)

    def fit(self, X: pd.DataFrame, covariates=None) -> "EMDTrend":
        try:
            from PyEMD import EMD
        except ImportError:
            raise ImportError(
                "EMDTrend requires the PyEMD package. "
                "Install it with: pip install EMD-signal"
            )
        x_filled, mask = _handle_nan(X)
        trend_data = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        emd = EMD()
        for col in X.columns:
            imfs = emd.emd(x_filled[col].values)
            trend_data[col] = imfs[-1]
        trend_data = _restore_nan(trend_data, mask)
        self._store_trend(X, trend_data)
        return self


class HighPassFilterTrend(NonParametricTrend):
    """Trend extraction by Butterworth high-pass filter.

    The trend is defined as ``signal - high_pass_filtered(signal)``,
    i.e. the low-frequency content.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency for the high-pass filter (in the same units as *fs*).
    fs : float
        Sampling frequency of the signal.
    order : int
        Order of the Butterworth filter (default 2).
    extrapolation : str
        Future extrapolation strategy (``'constant'`` or ``'linear'``).
    forecast_model : object, optional
        Forecasting model for trend extrapolation (see :class:`NonParametricTrend`).
    """

    def __init__(self, cutoff: float, fs: float, order: int = 2, extrapolation: str = 'linear',
                 forecast_model=None):
        super().__init__(extrapolation, forecast_model)
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

    def fit(self, X: pd.DataFrame, covariates=None) -> "HighPassFilterTrend":
        x_filled, mask = _handle_nan(X)
        b, a = butter(N=self.order, Wn=self.cutoff, btype='high', fs=self.fs)
        trend_data = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        for col in X.columns:
            filtered = filtfilt(b, a, x_filled[col].values)
            trend_data[col] = x_filled[col].values - filtered
        trend_data = _restore_nan(trend_data, mask)
        self._store_trend(X, trend_data)
        return self


# ---------------------------------------------------------------------------
# Transform-based detrending
# ---------------------------------------------------------------------------

class Differencing:
    """Detrend by differencing — used with ``DataCube.apply()``.

    Unlike the component-based methods, differencing is a sequential
    transformation whose inverse (cumulative sum) depends on previous
    values.  Use it via::

        diff = Differencing(order=1).fit(signal.data)
        signal.residual.apply(diff.forward, diff.inverse)

    Parameters
    ----------
    order : int
        Differencing order (default 1).  Order 2 applies differencing twice.
    """

    def __init__(self, order: int = 1):
        self.order = order
        self._anchors = []

    def fit(self, X: pd.DataFrame) -> "Differencing":
        """Store the anchor values needed to invert the differencing.

        For each differencing level ``d`` (0 .. order-1), the anchor is the
        last value of the ``d``-th differenced series.
        """
        self._features = X.columns
        self._anchors = []
        temp = X.copy()
        for _ in range(self.order):
            self._anchors.append(temp.iloc[-1].copy())
            temp = temp.diff().iloc[1:]
        return self

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply differencing (forward transform)."""
        self.fit(df)  # Update anchors to match the data being transformed
        result = df.copy()
        for _ in range(self.order):
            result = result.diff()
        return result.iloc[self.order:]

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Invert differencing via cumulative sum + anchors."""
        result = df.copy()
        for d in range(self.order - 1, -1, -1):
            result = result.cumsum()
            result = result.add(self._anchors[d])
        return result
