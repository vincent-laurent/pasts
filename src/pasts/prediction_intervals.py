# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Prediction intervals for time-series forecasts.

Three methods are available:

- **empirical** (``empirical_pi``): Gaussian assumption ``pred ± z × σ``.
- **bootstrap** (``bootstrap_pi``): resample historical residuals (no refit).
- **bootstrap_full** (``bootstrap_full_pi``): block-bootstrap training data,
  refit the model *B* times, take quantiles of the *B* prediction paths.

References
----------
Chatfield, C. (1993). Calculating interval forecasts. *Journal of Business &
    Economic Statistics*, 11(2), 121–135.

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
    Practice* (3rd ed.), OTexts. https://otexts.com/fpp3/prediction-intervals.html

Stine, R. A. (1985). Bootstrap prediction intervals for regression.
    *Journal of the American Statistical Association*, 80(392), 1026–1031.
"""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pasts.signal import Signal


_SCALINGS = ("constant", "sqrt_h")


def empirical_pi(
    pred: pd.DataFrame,
    residuals,
    scaling: str = "constant",
    z: float = 1.96,
) -> pd.DataFrame:
    """Compute empirical prediction intervals column-wise.

    Parameters
    ----------
    pred : pd.DataFrame
        Point forecasts or test predictions.
        Shape: (n_steps, n_series). Index is preserved in the output.
    residuals : array-like or pd.DataFrame
        Observed forecast errors used to estimate the base standard deviation.
        If a DataFrame, columns must match *pred*. If a 1-D array or Series,
        the same residuals are used for every column of *pred*.
    scaling : {"constant", "sqrt_h"}
        Growth assumption for the interval width:

        ``"constant"``
            ``CI(h) = pred(h) ± z × σ``

            Statistically appropriate when no strong horizon-growth assumption
            can be made (e.g., single train/test split). Recommended default.

        ``"sqrt_h"``
            ``CI(h) = pred(h) ± z × σ × √h``

            Assumes errors compound like a random walk (variance ∝ h).
            Valid for naïve / random-walk models; tends to over-widen for
            other model classes.

    z : float, optional
        Gaussian quantile multiplier. Default 1.96 (≈ 95 % coverage).

    Returns
    -------
    pd.DataFrame
        Same shape and index as *pred*. Each cell is a tuple ``(lower, upper)``.

    Notes
    -----
    The formula ``std(errors / √h) × √h`` is algebraically identical to
    ``std(errors)``; normalising by the horizon before computing the std does
    not improve the estimator.  To get a proper one-step-ahead σ₁ estimate,
    one-step-ahead historical forecasts (in-sample) should be used instead of
    multi-step test errors.
    """
    if scaling not in _SCALINGS:
        raise ValueError(
            f"scaling must be one of {_SCALINGS!r}, got {scaling!r}."
        )
    if scaling == "sqrt_h":
        warnings.warn(
            "scaling='sqrt_h' assumes a random-walk error structure "
            "(variance ∝ horizon). This may over-widen intervals for "
            "non-naïve models (ETS, ARIMA, …). "
            "See Chatfield (1993) and Hyndman & Athanasopoulos (2021).",
            UserWarning,
            stacklevel=2,
        )

    n_steps = len(pred)
    horizons = np.arange(1, n_steps + 1, dtype=float)

    df_ci = pd.DataFrame(index=pred.index, columns=pred.columns)

    for col in pred.columns:
        # Extract residuals for this column
        if isinstance(residuals, pd.DataFrame):
            res = np.asarray(residuals[col], dtype=float)
        else:
            res = np.asarray(residuals, dtype=float)

        sigma = np.std(res)

        vals = pred[col].values.astype(float)
        if scaling == "constant":
            half_width = z * sigma * np.ones(n_steps)
        else:  # "sqrt_h"
            half_width = z * sigma * np.sqrt(horizons)

        df_ci[col] = list(zip(vals - half_width, vals + half_width))

    return df_ci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_residual_pool(historical_residuals, col):
    """Return a clean 1-D float array of residuals for *col*."""
    if isinstance(historical_residuals, pd.DataFrame):
        if col in historical_residuals.columns:
            pool = historical_residuals[col].dropna().values.astype(float)
        else:
            pool = historical_residuals.iloc[:, 0].dropna().values.astype(float)
    else:
        pool = np.asarray(historical_residuals, dtype=float)
        pool = pool[~np.isnan(pool)]
    if len(pool) == 0:
        raise ValueError(
            f"No valid historical residuals for column '{col}'."
        )
    return pool


def _block_bootstrap(residuals_df, block_size, n_obs, rng):
    """Block-resample a residual DataFrame preserving cross-column structure.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Historical residuals (rows = time, columns = series).
    block_size : int
        Length of each block.
    n_obs : int
        Number of rows required in the output.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Shape ``(n_obs, n_columns)``.
    """
    values = residuals_df.values.astype(float)
    n = len(values)
    if block_size >= n:
        # Not enough data for blocks — fall back to iid row resampling
        idx = rng.integers(0, n, size=n_obs)
        return values[idx]

    n_blocks = int(np.ceil(n_obs / block_size))
    max_start = n - block_size
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    blocks = [values[s: s + block_size] for s in starts]
    return np.concatenate(blocks, axis=0)[:n_obs]


# ---------------------------------------------------------------------------
# Bootstrap — residual resampling (no refit)
# ---------------------------------------------------------------------------

def bootstrap_pi(
    pred: pd.DataFrame,
    historical_residuals: pd.DataFrame,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    block_size: int = None,
    random_state=None,
) -> pd.DataFrame:
    """Non-parametric bootstrap prediction intervals (no model refit).

    For each bootstrap iteration, residuals are resampled with replacement
    and added to the point predictions.  The bounds are the ``alpha/2``
    and ``1 - alpha/2`` percentiles across iterations.

    By default, block resampling is used to preserve temporal correlation
    in the residuals (block size defaults to ``int(sqrt(n))``).  Set
    ``block_size=1`` to recover i.i.d. resampling.

    Parameters
    ----------
    pred : pd.DataFrame
        Point forecasts or test predictions (n_steps, n_series).
    historical_residuals : pd.DataFrame
        One-step-ahead historical residuals from the training set.
    n_bootstrap : int
        Number of bootstrap iterations (default 1000).
    alpha : float
        Significance level (default 0.05 → 95 % CI).
    block_size : int or None
        Block length for block bootstrap.  ``None`` (default) uses
        ``int(sqrt(n_residuals))``.  Set to ``1`` for i.i.d. resampling.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Same shape/index as *pred*.  Each cell is a ``(lower, upper)`` tuple.
    """
    rng = np.random.default_rng(random_state)
    n_steps = len(pred)
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    # Ensure historical_residuals is a DataFrame for _block_bootstrap
    if not isinstance(historical_residuals, pd.DataFrame):
        historical_residuals = pd.DataFrame(historical_residuals)

    n_res = len(historical_residuals)
    if block_size is None:
        block_size = max(1, int(np.sqrt(n_res)))

    df_ci = pd.DataFrame(index=pred.index, columns=pred.columns)

    for col in pred.columns:
        pool = _extract_residual_pool(historical_residuals, col)
        vals = pred[col].values.astype(float)

        # Block-resample residuals for each bootstrap iteration
        pool_df = pd.DataFrame(pool, columns=["res"])
        samples = np.empty((n_bootstrap, n_steps))
        for b in range(n_bootstrap):
            resampled = _block_bootstrap(pool_df, block_size, n_steps, rng)
            samples[b] = vals + resampled[:, 0]

        lower = np.percentile(samples, lower_q * 100, axis=0)
        upper = np.percentile(samples, upper_q * 100, axis=0)
        df_ci[col] = list(zip(lower, upper))

    return df_ci


# ---------------------------------------------------------------------------
# Bootstrap — full refit (block bootstrap)
# ---------------------------------------------------------------------------

def bootstrap_full_pi(
    model,
    train_data: pd.DataFrame,
    pred: pd.DataFrame,
    historical_residuals: pd.DataFrame,
    covariates=None,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    block_size: int = None,
    random_state=None,
) -> pd.DataFrame:
    """Bootstrap prediction intervals with model refit (block bootstrap).

    For each iteration the training residuals are block-resampled, a
    synthetic training series is built (fitted + resampled residuals),
    the model is deep-copied and refitted, and predictions of the same
    length as *pred* are generated.  The bounds are the ``alpha/2`` and
    ``1 - alpha/2`` percentiles across all *B* prediction paths.

    Parameters
    ----------
    model : TimeSeriesModel
        Already-fitted model instance.  A deep copy is created for each
        bootstrap iteration.
    train_data : pd.DataFrame
        Training data used during the original fit.
    pred : pd.DataFrame
        Point predictions (used for shape and index of the output).
    historical_residuals : pd.DataFrame
        One-step-ahead historical residuals on *train_data*.
    covariates : :class:`~pasts.covariates.Covariates`, optional
        Covariates forwarded to each bootstrap refit.
    n_bootstrap : int
        Number of bootstrap iterations (default 200).
    alpha : float
        Significance level (default 0.05 → 95 % CI).
    block_size : int or None
        Block length for the block bootstrap.  ``None`` uses
        ``int(sqrt(n_residuals))`` as default.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Same shape/index as *pred*.  Each cell is ``(lower, upper)``.
    """
    rng = np.random.default_rng(random_state)
    pred_length = len(pred)

    # Align train_data to the portion covered by historical_residuals
    aligned_train = train_data.loc[historical_residuals.index]
    fitted = aligned_train - historical_residuals
    n_res = len(historical_residuals)

    if block_size is None:
        block_size = max(1, int(np.sqrt(n_res)))

    n_cols = len(fitted.columns)
    all_preds = np.empty((n_bootstrap, pred_length, n_cols))

    for b in range(n_bootstrap):
        resampled_res = _block_bootstrap(
            historical_residuals, block_size, n_res, rng
        )
        bootstrap_train = pd.DataFrame(
            fitted.values + resampled_res,
            index=fitted.index,
            columns=fitted.columns,
        )
        model_copy = copy.deepcopy(model)
        model_copy.fit(bootstrap_train, covariates=covariates)
        preds_b = model_copy.reverse_transform(pred_length)
        all_preds[b] = preds_b.values[:pred_length]

    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    lower = np.percentile(all_preds, lower_q * 100, axis=0)
    upper = np.percentile(all_preds, upper_q * 100, axis=0)

    df_ci = pd.DataFrame(index=pred.index, columns=pred.columns)
    for j, col in enumerate(pred.columns):
        df_ci[col] = list(zip(lower[:, j], upper[:, j]))

    return df_ci


# ---------------------------------------------------------------------------
# CIAccessor — prediction-interval accessor for Signal
# ---------------------------------------------------------------------------

_METHODS = ("empirical", "bootstrap", "bootstrap_full")


class CIAccessor:
    """Accessor for prediction-interval computation on a :class:`~pasts.signal.Signal`.

    Accessed via ``signal.ci``.  Follows the same pattern as
    :class:`~pasts.statistical_tests.StatAccessor` and
    :class:`~pasts.visualization.PlotAccessor`.

    Usage::

        signal.ci.compute()                                     # block bootstrap (default)
        signal.ci.compute(method="empirical")                   # Gaussian assumption
        signal.ci.compute(method="bootstrap_full", n_bootstrap=200)  # full refit
    """

    def __init__(self, signal: "Signal"):
        self._signal = signal

    # -- internal helpers ---------------------------------------------------

    def _resolve_bootstrap_context(self, model_name, result):
        """Return ``(model, train_data)`` for bootstrap residual computation.

        For decomposition entries (``"decomp__model"``), the model lives in
        the residual space, so we use the decomposition's training data.
        """
        sig = self._signal
        if '__' in model_name and model_name in sig.models:
            decomp_name, sub_model_name = model_name.split('__', 1)
            if decomp_name in sig._decompositions:
                decomp_signal = sig._decompositions[decomp_name]
                decomp_result = decomp_signal.models.get(sub_model_name)
                if decomp_result is not None:
                    return decomp_result.model, decomp_signal.train_data
        return result.model, sig.train_data

    def _ensure_historical_residuals(self, model_name, result):
        """Lazily compute and cache historical residuals on *result*."""
        if result.historical_residuals is None:
            model, train_data = self._resolve_bootstrap_context(
                model_name, result
            )
            result.historical_residuals = model.compute_historical_residuals(
                train_data
            )
        return result.historical_residuals

    @staticmethod
    def _compute_test_residuals(result, test_data):
        """Compute and store test residuals (actual - predicted)."""
        pred = result.predictions
        df_residuals = pd.DataFrame(index=pred.index, columns=pred.columns)
        for ref in pred.columns:
            df_residuals[ref] = test_data[ref].values - pred[ref].values
        result.test_residuals = df_residuals
        return df_residuals

    # -- per-model interval computation ------------------------------------

    def _interval_test(self, model_name, *, scaling, method,
                       n_bootstrap, alpha, block_size, random_state):
        sig = self._signal
        if model_name not in sig.models:
            raise AttributeError(f'{model_name} has not been fitted.')
        result = sig.models[model_name]
        pred = result.predictions
        self._compute_test_residuals(result, sig.test_data)

        if method == "empirical":
            result.test_confidence_interval = empirical_pi(
                pred, result.test_residuals, scaling=scaling
            )
        elif method == "bootstrap":
            hist_res = self._ensure_historical_residuals(model_name, result)
            result.test_confidence_interval = bootstrap_pi(
                pred, hist_res,
                n_bootstrap=n_bootstrap, alpha=alpha,
                block_size=block_size, random_state=random_state,
            )
        elif method == "bootstrap_full":
            hist_res = self._ensure_historical_residuals(model_name, result)
            model, train_data = self._resolve_bootstrap_context(
                model_name, result
            )
            result.test_confidence_interval = bootstrap_full_pi(
                model, train_data, pred, hist_res,
                covariates=sig._covariates,
                n_bootstrap=n_bootstrap, alpha=alpha,
                block_size=block_size, random_state=random_state,
            )

    def _interval_forecast(self, model_name, *, scaling, method,
                           n_bootstrap, alpha, block_size, random_state):
        sig = self._signal
        if model_name not in sig.models:
            raise AttributeError(f'{model_name} has not been fitted.')
        result = sig.models[model_name]
        if result.forecast_data is None:
            raise AttributeError(
                f'No forecasts have been computed with model {model_name}.'
            )

        if method == "empirical":
            result.forecast_confidence_interval = empirical_pi(
                result.forecast_data, result.test_residuals, scaling=scaling
            )
        elif method == "bootstrap":
            hist_res = self._ensure_historical_residuals(model_name, result)
            result.forecast_confidence_interval = bootstrap_pi(
                result.forecast_data, hist_res,
                n_bootstrap=n_bootstrap, alpha=alpha,
                block_size=block_size, random_state=random_state,
            )
        elif method == "bootstrap_full":
            result._ensure_final_estimator()
            final_est = result.final_estimator
            full_hist_res = final_est.compute_historical_residuals(sig.data)
            result.forecast_confidence_interval = bootstrap_full_pi(
                final_est, sig.data, result.forecast_data, full_hist_res,
                covariates=sig._covariates,
                n_bootstrap=n_bootstrap, alpha=alpha,
                block_size=block_size, random_state=random_state,
            )

    # -- public API --------------------------------------------------------

    def compute(self, scaling_test: str = "constant",
                scaling_forecast: str = "sqrt_h",
                method: str = "bootstrap",
                n_bootstrap: int = 200,
                alpha: float = 0.05,
                block_size: int = None,
                random_state=None,
                save: bool = False):
        """Compute confidence intervals for all fitted models.

        Parameters
        ----------
        scaling_test : {"constant", "sqrt_h"}, optional
            Width scaling for test-period intervals (default ``"constant"``).
            Only used when ``method="empirical"``.
        scaling_forecast : {"constant", "sqrt_h"}, optional
            Width scaling for forecast intervals (default ``"sqrt_h"``).
            Only used when ``method="empirical"``.
        method : {"empirical", "bootstrap", "bootstrap_full"}, optional
            Interval computation method (default ``"bootstrap"``).

            ``"empirical"``
                Gaussian: ``pred ± z × σ``.  Uses test-set residuals.
            ``"bootstrap"``
                Block-resample historical residuals (no refit).
                Preserves temporal correlation via block bootstrap.
            ``"bootstrap_full"``
                Block-bootstrap training data, refit *B* times, take
                quantiles of the *B* prediction paths.
        n_bootstrap : int, optional
            Number of bootstrap iterations (default 1000).
            Ignored when ``method="empirical"``.
        alpha : float, optional
            Significance level (default 0.05 → 95 % CI).
            Ignored when ``method="empirical"``.
        block_size : int or None, optional
            Block length for ``"bootstrap"`` and ``"bootstrap_full"``
            (default ``sqrt(n)``).  Set to ``1`` for i.i.d. resampling.
        random_state : int or None, optional
            Seed for bootstrap reproducibility.
        save : bool, optional
            Persist results to disk (default ``False``).
        """
        if method not in _METHODS:
            raise ValueError(
                f"method must be one of {_METHODS!r}, got {method!r}."
            )

        sig = self._signal
        if not sig.models:
            raise AttributeError('No predictions have been found.')

        from pasts import persistence

        kw = {"method": method, "n_bootstrap": n_bootstrap, "alpha": alpha,
              "block_size": block_size, "random_state": random_state}

        for model_ in sig.models.keys():
            self._interval_test(model_, scaling=scaling_test, **kw)
            if sig.models[model_].forecast_data is not None:
                self._interval_forecast(model_, scaling=scaling_forecast, **kw)
            if save:
                persistence.save_model(sig.path, model_, sig.models[model_])
                if sig.models[model_].forecast_data is not None:
                    persistence.save_model(
                        sig.path, model_, sig.models[model_], suffix="final"
                    )
