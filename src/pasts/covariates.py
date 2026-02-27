"""Covariate container and validation for time-series models.

Provides the :class:`Covariates` dataclass that bundles past, future, and
static covariates into a single immutable object, plus a validation helper
that checks index alignment with the target signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Covariates:
    """Immutable container for the three types of covariates.

    Parameters
    ----------
    past : pd.DataFrame or None
        Past covariates — known only up to the present.
    future : pd.DataFrame or None
        Future covariates — known ahead of time (e.g. calendar features).
    static : pd.DataFrame or None
        Static (time-invariant) covariates.
    """

    past: Optional[pd.DataFrame] = None
    future: Optional[pd.DataFrame] = None
    static: Optional[pd.DataFrame] = None

    @property
    def is_empty(self) -> bool:
        """Return True when no covariates have been set."""
        return self.past is None and self.future is None and self.static is None


def validate_covariates(
    signal_index: pd.Index,
    covariates: Covariates,
    forecast_horizon: int = 0,
) -> None:
    """Validate covariate alignment with the signal index.

    Parameters
    ----------
    signal_index : pd.Index
        DatetimeIndex (or numeric) of the target signal.
    covariates : Covariates
        Covariates to validate.
    forecast_horizon : int, optional
        When > 0, checks that future covariates extend far enough for
        forecasting.

    Raises
    ------
    ValueError
        If temporal covariate indices do not cover *signal_index*, or if
        future covariates are too short for the requested horizon.
    """
    if covariates is None or covariates.is_empty:
        return

    if covariates.past is not None:
        past_idx = covariates.past.index
        if not signal_index.isin(past_idx).all():
            missing = signal_index.difference(past_idx)
            raise ValueError(
                f"Past covariates do not cover {len(missing)} signal "
                f"timestamps. First missing: {missing[0]}."
            )

    if covariates.future is not None:
        future_idx = covariates.future.index
        if not signal_index.isin(future_idx).all():
            missing = signal_index.difference(future_idx)
            raise ValueError(
                f"Future covariates do not cover {len(missing)} signal "
                f"timestamps. First missing: {missing[0]}."
            )
        if forecast_horizon > 0:
            last_signal = signal_index[-1]
            n_future = len(future_idx[future_idx > last_signal])
            if n_future < forecast_horizon:
                raise ValueError(
                    f"Future covariates extend only {n_future} steps beyond "
                    f"the signal, but forecast_horizon={forecast_horizon}."
                )
