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

import numpy as np
import pandas as pd


class NaNHandler:
    """Accessor for NaN handling on a Signal.

    Obtained via ``signal.handle_nan``. All methods operate **in place** on the
    signal data and return ``self`` so that calls can be chained::

        signal.handle_nan.before_launch(0).interpolate(max_consecutive=5).after_stops(0)

    Methods
    -------
    fill(value=0)
        Replace every NaN with a scalar value.
    interpolate(max_consecutive=None)
        Interpolate **internal** gaps only (never touches leading/trailing NaN).
    extrapolate(method='ffill')
        Fill NaN at the **edges** of each column (before first / after last valid value).
    before_launch(value=0)
        Fill NaN before the first valid observation in each column.
    after_stops(value=0)
        Fill NaN after the last valid observation in each column.
    """

    def __init__(self, signal):
        self._signal = signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply(self, result: pd.DataFrame) -> "NaNHandler":
        """Commit *result* back into the signal and reset metadata."""
        # Import here to avoid circular imports at module level
        from pasts.signal import Signal  # noqa: F401 – used for _profiling only

        pd.DataFrame.__init__(self._signal, data=result)
        self._signal._properties = type(self._signal)._profiling(self._signal)
        if self._signal._validation._timestamp is not None:
            self._signal._validation.reset()
            warnings.warn(
                "Train/test split has been reset after handle_nan(). "
                "Call validation_split() again.",
                UserWarning,
                stacklevel=3,
            )
        return self

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fill(self, value: float = 0) -> "NaNHandler":
        """Replace all NaN values with *value*.

        Parameters
        ----------
        value : float, optional
            Fill value (default 0).
        """
        result = pd.DataFrame(self._signal).fillna(value)
        return self._apply(result)

    def interpolate(self, max_consecutive: int = None) -> "NaNHandler":
        """Interpolate **internal** NaN gaps with linear interpolation.

        Only gaps that are surrounded by valid values on both sides are filled
        (``limit_area='inside'``). Leading and trailing NaN are left untouched.

        Parameters
        ----------
        max_consecutive : int, optional
            Maximum number of consecutive NaN values to interpolate. Gaps
            longer than this value are left as NaN. If ``None`` (default),
            all internal gaps are interpolated regardless of length.
        """
        df = pd.DataFrame(self._signal)
        result = df.interpolate(
            method='linear',
            limit=max_consecutive,
            limit_area='inside',
        )
        return self._apply(result)

    def extrapolate(self, method: str = 'ffill') -> "NaNHandler":
        """Fill NaN at the **edges** of each column.

        Targets only leading NaN (before first valid value) and trailing NaN
        (after last valid value). Internal NaN are left untouched.

        Parameters
        ----------
        method : str, optional
            ``'ffill'`` — forward fill (repeat last known value).
            ``'bfill'`` — backward fill (repeat first known value).
            ``'linear'`` — linear extrapolation using the 2 nearest valid points.
            Default is ``'ffill'``.
        """
        if method not in ('ffill', 'bfill', 'linear'):
            raise ValueError(
                f"Unknown extrapolation method {method!r}. "
                "Use 'ffill', 'bfill', or 'linear'."
            )
        df = pd.DataFrame(self._signal).copy()

        if method == 'ffill':
            result = df.ffill()
        elif method == 'bfill':
            result = df.bfill()
        else:  # linear
            result = df.copy()
            for col in df.columns:
                series = df[col]
                valid = series.dropna()
                if len(valid) < 2:
                    continue
                # Forward extrapolation (trailing NaN)
                if pd.isna(series.iloc[-1]):
                    x0, x1 = valid.index[-2], valid.index[-1]
                    y0, y1 = valid.iloc[-2], valid.iloc[-1]
                    slope = _linear_slope(x0, x1, y0, y1)
                    for idx in series.index[series.index > x1]:
                        gap = _index_diff(x1, idx, series.index)
                        result.loc[idx, col] = y1 + slope * gap
                # Backward extrapolation (leading NaN)
                if pd.isna(series.iloc[0]):
                    x0, x1 = valid.index[0], valid.index[1]
                    y0, y1 = valid.iloc[0], valid.iloc[1]
                    slope = _linear_slope(x0, x1, y0, y1)
                    for idx in series.index[series.index < x0]:
                        gap = _index_diff(x0, idx, series.index)
                        result.loc[idx, col] = y0 + slope * gap

        return self._apply(result)

    def before_launch(self, value: float = 0) -> "NaNHandler":
        """Fill NaN **before** the first valid observation in each column.

        Useful for products that were not yet launched at the start of the
        series — their leading NaN values are replaced with *value*.

        Parameters
        ----------
        value : float, optional
            Fill value (default 0).
        """
        df = pd.DataFrame(self._signal).copy()
        for col in df.columns:
            first_valid = df[col].first_valid_index()
            if first_valid is not None:
                mask = df.index < first_valid
                df.loc[mask, col] = df.loc[mask, col].fillna(value)
        return self._apply(df)

    def after_stops(self, value: float = 0) -> "NaNHandler":
        """Fill NaN **after** the last valid observation in each column.

        Useful for products that have been discontinued — their trailing NaN
        values are replaced with *value* (typically 0).

        Parameters
        ----------
        value : float, optional
            Fill value (default 0).
        """
        df = pd.DataFrame(self._signal).copy()
        for col in df.columns:
            last_valid = df[col].last_valid_index()
            if last_valid is not None:
                mask = df.index > last_valid
                df.loc[mask, col] = df.loc[mask, col].fillna(value)
        return self._apply(df)


# ------------------------------------------------------------------
# Private helpers for linear extrapolation
# ------------------------------------------------------------------

def _linear_slope(x0, x1, y0, y1):
    """Slope between two points. Works for both numeric and datetime indices."""
    dx = _index_diff(x0, x1, None)
    if dx == 0:
        return 0.0
    return (y1 - y0) / dx


def _index_diff(ref, target, index):
    """Signed numeric distance from *ref* to *target* (1 unit = 1 step)."""
    if isinstance(ref, (pd.Timestamp, np.datetime64)):
        # Use nanoseconds then normalise to avoid floating-point issues
        ref_ns = pd.Timestamp(ref).value
        tgt_ns = pd.Timestamp(target).value
        return tgt_ns - ref_ns
    return float(target) - float(ref)
