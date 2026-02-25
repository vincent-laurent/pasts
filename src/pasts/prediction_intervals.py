# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Empirical prediction intervals for time-series forecasts.

References
----------
Chatfield, C. (1993). Calculating interval forecasts. *Journal of Business &
    Economic Statistics*, 11(2), 121–135.

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
    Practice* (3rd ed.), OTexts. https://otexts.com/fpp3/prediction-intervals.html
"""

import warnings

import numpy as np
import pandas as pd


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
