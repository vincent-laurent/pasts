# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import pandas as pd
from darts import TimeSeries
from darts.utils.statistics import check_seasonality
from scipy.stats import shapiro, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

if TYPE_CHECKING:
    from pasts.signal import Signal

_STATIONARITY_METHODS = {'adfuller', 'kpss'}
_NORMALITY_METHODS = {'shapiro': shapiro, 'jarque_bera': jarque_bera}


def _check_nan(data: pd.DataFrame) -> None:
    """Raise ValueError if data contains NaN values."""
    if data.isna().any().any():
        n = int(data.isna().sum().sum())
        cols = list(data.columns[data.isna().any()])
        raise ValueError(
            f"Data contains {n} NaN value(s) in column(s) {cols}. "
            f"Statistical tests require NaN-free data. "
            f"Handle missing values before running tests "
            f"(e.g. df.fillna(), df.dropna(), df.interpolate())."
        )


class StatAccessor:
    """Accessor for statistical tests on a Signal.

    Returned by ``signal.stat``. Each method runs a specific test
    family and returns a :class:`pd.DataFrame` with structured results.

    Usage::

        signal.stat.test_stationarity()               # ADF per column
        signal.stat.test_stationarity(method='kpss')   # KPSS per column
        signal.stat.test_seasonality()                 # per column
        signal.stat.test_causality()                   # all column pairs
    """

    def __init__(self, signal: Signal):
        self._signal = signal

    def test_stationarity(self, method: str = 'adfuller',
                          alpha: float = 0.05, **kwargs) -> pd.DataFrame:
        """Per-column stationarity test using ADF or KPSS.

        Parameters
        ----------
        method : str
            ``'adfuller'`` (default) or ``'kpss'``.
        alpha : float
            Significance level (default 0.05).
        **kwargs
            Passed to the underlying statsmodels function.

        Returns
        -------
        pd.DataFrame
            Columns: ``statistic``, ``p_value``, ``n_lags``, ``n_obs``, ``stationary``.
            Indexed by signal column names.
        """
        if method not in _STATIONARITY_METHODS:
            raise ValueError(
                f"Unknown method {method!r}. Choose from {sorted(_STATIONARITY_METHODS)}."
            )
        _check_nan(self._signal.data)
        rows = []
        for col in self._signal.data.columns:
            series = self._signal.data[col]
            if method == 'adfuller':
                result = adfuller(series, **kwargs)
                rows.append({
                    'statistic': result[0],
                    'p_value': result[1],
                    'n_lags': result[2],
                    'n_obs': result[3],
                    'stationary': result[1] <= alpha,
                })
            else:  # kpss
                result = kpss(series, **kwargs)
                rows.append({
                    'statistic': result[0],
                    'p_value': result[1],
                    'n_lags': result[2],
                    'n_obs': len(series),
                    'stationary': result[1] > alpha,
                })
        df = pd.DataFrame(rows, index=self._signal.data.columns)
        self._signal._tests_stat['stationarity'] = df
        return df

    def test_seasonality(self, **kwargs) -> pd.DataFrame:
        """Per-column seasonality detection.

        Parameters
        ----------
        **kwargs
            Passed to ``darts.utils.statistics.check_seasonality``.

        Returns
        -------
        pd.DataFrame
            Columns: ``seasonal``, ``period``.
            Indexed by signal column names.
        """
        _check_nan(self._signal.data)
        rows = []
        for col in self._signal.data.columns:
            ts = TimeSeries.from_dataframe(self._signal.data[[col]])
            is_seasonal, period = check_seasonality(ts, **kwargs)
            rows.append({
                'seasonal': is_seasonal,
                'period': period,
            })
        df = pd.DataFrame(rows, index=self._signal.data.columns)
        self._signal._tests_stat['seasonality'] = df
        return df

    def test_causality(self, maxlag: int = 1,
                       alpha: float = 0.05, **kwargs) -> pd.DataFrame:
        """Granger causality test for all column pairs.

        Parameters
        ----------
        maxlag : int
            Maximum lag to test (default 1).
        alpha : float
            Significance level (default 0.05).
        **kwargs
            Passed to ``statsmodels.tsa.stattools.grangercausalitytests``.

        Returns
        -------
        pd.DataFrame
            Columns: ``statistic``, ``p_value``, ``causal``.
            Indexed by directed pair strings (e.g. ``"A-->B"``).

        Raises
        ------
        ValueError
            If the signal is univariate (causality requires at least 2 columns).
        """
        _check_nan(self._signal.data)
        if self._signal.properties['is_univariate']:
            raise ValueError(
                "Causality tests require multivariate data (at least 2 columns)."
            )
        names = self._signal.data.columns
        rows = []
        index = []
        for s1, s2 in combinations(names, 2):
            result_12 = grangercausalitytests(
                self._signal.data[[s1, s2]], maxlag, **kwargs
            )
            result_21 = grangercausalitytests(
                self._signal.data[[s2, s1]], maxlag, **kwargs
            )
            stat12 = result_12[1][0]['ssr_ftest'][0]
            pv12 = result_12[1][0]['ssr_ftest'][1]
            stat21 = result_21[1][0]['ssr_ftest'][0]
            pv21 = result_21[1][0]['ssr_ftest'][1]

            rows.append({'statistic': stat12, 'p_value': pv12, 'causal': pv12 <= alpha})
            index.append(f"{s1}-->{s2}")
            rows.append({'statistic': stat21, 'p_value': pv21, 'causal': pv21 <= alpha})
            index.append(f"{s2}-->{s1}")

        df = pd.DataFrame(rows, index=index)
        self._signal._tests_stat['causality'] = df
        return df

    def test_normality(self, method: str = 'shapiro',
                       alpha: float = 0.05, **kwargs) -> pd.DataFrame:
        """Per-column normality test.

        Parameters
        ----------
        method : str
            ``'shapiro'`` (default) or ``'jarque_bera'``.
        alpha : float
            Significance level (default 0.05).
        **kwargs
            Passed to the underlying scipy function.

        Returns
        -------
        pd.DataFrame
            Columns: ``statistic``, ``p_value``, ``normal``.
            Indexed by signal column names.
        """
        if method not in _NORMALITY_METHODS:
            raise ValueError(
                f"Unknown method {method!r}. Choose from {sorted(_NORMALITY_METHODS)}."
            )
        _check_nan(self._signal.data)
        func = _NORMALITY_METHODS[method]
        rows = []
        for col in self._signal.data.columns:
            result = func(self._signal.data[col], **kwargs)
            rows.append({
                'statistic': result.statistic,
                'p_value': result.pvalue,
                'normal': result.pvalue > alpha,
            })
        df = pd.DataFrame(rows, index=self._signal.data.columns)
        self._signal._tests_stat['normality'] = df
        return df

    def test_whitenoise(self, lags: int = 10,
                        alpha: float = 0.05, **kwargs) -> pd.DataFrame:
        """Per-column Ljung-Box test for white noise (no autocorrelation).

        Parameters
        ----------
        lags : int
            Number of lags to test (default 10).
        alpha : float
            Significance level (default 0.05).
        **kwargs
            Passed to ``statsmodels.stats.diagnostic.acorr_ljungbox``.

        Returns
        -------
        pd.DataFrame
            Columns: ``statistic``, ``p_value``, ``white_noise``.
            Indexed by signal column names.
        """
        _check_nan(self._signal.data)
        rows = []
        for col in self._signal.data.columns:
            lb = acorr_ljungbox(self._signal.data[col], lags=lags,
                                return_df=True, **kwargs)
            # Use the last lag (most conservative)
            last = lb.iloc[-1]
            rows.append({
                'statistic': last['lb_stat'],
                'p_value': last['lb_pvalue'],
                'white_noise': last['lb_pvalue'] > alpha,
            })
        df = pd.DataFrame(rows, index=self._signal.data.columns)
        self._signal._tests_stat['whitenoise'] = df
        return df

    def test_heteroscedasticity(self, nlags: int = 5,
                                alpha: float = 0.05,
                                **kwargs) -> pd.DataFrame:
        """Per-column ARCH test for heteroscedasticity.

        Parameters
        ----------
        nlags : int
            Number of lags for the ARCH test (default 5).
        alpha : float
            Significance level (default 0.05).
        **kwargs
            Passed to ``statsmodels.stats.diagnostic.het_arch``.

        Returns
        -------
        pd.DataFrame
            Columns: ``statistic``, ``p_value``, ``homoscedastic``.
            Indexed by signal column names.
        """
        _check_nan(self._signal.data)
        rows = []
        for col in self._signal.data.columns:
            result = het_arch(self._signal.data[col], nlags=nlags, **kwargs)
            rows.append({
                'statistic': result[0],
                'p_value': result[1],
                'homoscedastic': result[1] > alpha,
            })
        df = pd.DataFrame(rows, index=self._signal.data.columns)
        self._signal._tests_stat['heteroscedasticity'] = df
        return df

    def all(self, alpha: float = 0.05, **kwargs) -> dict[str, pd.DataFrame]:
        """Run all applicable tests and return results.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05).
        **kwargs
            Passed to individual test methods.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys are test names, values are result DataFrames.
        """
        results = {
            'stationarity': self.test_stationarity(alpha=alpha),
            'seasonality': self.test_seasonality(**kwargs),
            'normality': self.test_normality(alpha=alpha),
            'whitenoise': self.test_whitenoise(alpha=alpha),
            'heteroscedasticity': self.test_heteroscedasticity(alpha=alpha),
        }
        if not self._signal.properties['is_univariate']:
            results['causality'] = self.test_causality(alpha=alpha)
        return results
