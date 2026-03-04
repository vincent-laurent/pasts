"""Tests for covariate support (past, future, static)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pasts.covariates import Covariates, validate_covariates
from pasts.signal import Signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def signal_data():
    """Univariate signal with 120 monthly observations."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01-01", periods=120, freq="MS")
    return pd.DataFrame({"target": np.cumsum(rng.standard_normal(120))}, index=dates)


@pytest.fixture(scope="module")
def past_cov(signal_data):
    """Past covariates aligned with signal_data."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"gdp": rng.standard_normal(len(signal_data))},
        index=signal_data.index,
    )


@pytest.fixture(scope="module")
def future_cov(signal_data):
    """Future covariates extending 24 steps beyond signal_data."""
    rng = np.random.default_rng(1)
    n = len(signal_data) + 24
    dates = pd.date_range("2010-01-01", periods=n, freq="MS")
    return pd.DataFrame({"temp": rng.standard_normal(n)}, index=dates)


@pytest.fixture(scope="module")
def static_cov():
    """Static covariates (one row)."""
    return pd.DataFrame({"category": [1.0]})


# ---------------------------------------------------------------------------
# Covariates dataclass
# ---------------------------------------------------------------------------

class TestCovariatesDataclass:
    def test_empty(self):
        cov = Covariates()
        assert cov.is_empty

    def test_not_empty_past(self, past_cov):
        cov = Covariates(past=past_cov)
        assert not cov.is_empty

    def test_not_empty_future(self, future_cov):
        cov = Covariates(future=future_cov)
        assert not cov.is_empty

    def test_not_empty_static(self, static_cov):
        cov = Covariates(static=static_cov)
        assert not cov.is_empty

    def test_frozen(self, past_cov):
        cov = Covariates(past=past_cov)
        with pytest.raises(AttributeError):
            cov.past = None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_covariates(self, signal_data, past_cov, future_cov, static_cov):
        cov = Covariates(past=past_cov, future=future_cov, static=static_cov)
        validate_covariates(signal_data.index, cov)  # should not raise

    def test_empty_covariates(self, signal_data):
        validate_covariates(signal_data.index, Covariates())  # should not raise
        validate_covariates(signal_data.index, None)  # should not raise

    def test_past_missing_timestamps(self, signal_data):
        short = signal_data.iloc[:50]
        cov = Covariates(past=short)
        with pytest.raises(ValueError, match="Past covariates do not cover"):
            validate_covariates(signal_data.index, cov)

    def test_future_missing_timestamps(self, signal_data):
        short = signal_data.iloc[:50]
        cov = Covariates(future=short)
        with pytest.raises(ValueError, match="Future covariates do not cover"):
            validate_covariates(signal_data.index, cov)

    def test_future_insufficient_horizon(self, signal_data, past_cov):
        # future covariates that cover signal but don't extend beyond
        cov = Covariates(future=past_cov)  # same length as signal
        with pytest.raises(ValueError, match="Future covariates extend only"):
            validate_covariates(signal_data.index, cov, forecast_horizon=12)

    def test_future_sufficient_horizon(self, signal_data, future_cov):
        cov = Covariates(future=future_cov)
        validate_covariates(signal_data.index, cov, forecast_horizon=24)


# ---------------------------------------------------------------------------
# Signal.set_covariates
# ---------------------------------------------------------------------------

class TestSignalSetCovariates:
    def test_set_and_read(self, signal_data, past_cov, future_cov, static_cov):
        sig = Signal(signal_data.copy())
        sig.set_covariates(
            past_covariates=past_cov,
            future_covariates=future_cov,
            static_covariates=static_cov,
        )
        assert not sig.covariates.is_empty
        assert sig.covariates.past is past_cov
        assert sig.covariates.future is future_cov
        assert sig.covariates.static is static_cov

    def test_default_empty(self, signal_data):
        sig = Signal(signal_data.copy())
        assert sig.covariates.is_empty

    def test_validation_error(self, signal_data):
        sig = Signal(signal_data.copy())
        short = signal_data.iloc[:10]
        with pytest.raises(ValueError, match="Past covariates"):
            sig.set_covariates(past_covariates=short)

    def test_decomposition_creates_decomposition_model(self, signal_data, past_cov):
        sig = Signal(signal_data.copy())
        sig.set_covariates(past_covariates=past_cov)
        sig.decompose("test")
        from pasts.core.decomposition import DecompositionModel
        assert isinstance(sig.decompositions["test"], DecompositionModel)


# ---------------------------------------------------------------------------
# Integration with DartsModel (no covariates — backward compat)
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_apply_model_no_covariates(self, signal_data):
        """Models work without covariates (existing behaviour)."""
        from darts.models import ExponentialSmoothing
        sig = Signal(signal_data.copy())
        sig.validation_split("2018-01-01")
        sig.apply_model(ExponentialSmoothing())
        assert "ExponentialSmoothing" in sig.models
        assert sig.models["ExponentialSmoothing"].predictions is not None


# ---------------------------------------------------------------------------
# Integration with DartsModel + covariates
# ---------------------------------------------------------------------------

class TestDartsModelWithCovariates:
    def test_apply_model_with_future_covariates(self, signal_data, future_cov):
        """A model that supports future covariates uses them."""
        from darts.models import LinearRegressionModel
        sig = Signal(signal_data.copy())
        sig.set_covariates(future_covariates=future_cov)
        sig.validation_split("2018-01-01")
        sig.apply_model(
            LinearRegressionModel(lags=12, lags_future_covariates=[0, 1, 2])
        )
        assert "LinearRegressionModel" in sig.models
        assert sig.models["LinearRegressionModel"].predictions is not None

    def test_forecast_with_future_covariates(self, signal_data, future_cov):
        from darts.models import LinearRegressionModel
        sig = Signal(signal_data.copy())
        sig.set_covariates(future_covariates=future_cov)
        sig.validation_split("2018-01-01")
        sig.apply_model(
            LinearRegressionModel(lags=12, lags_future_covariates=[0, 1, 2])
        )
        sig.refit("LinearRegressionModel")
        sig.forecast("LinearRegressionModel", horizon=12)
        assert sig.models["LinearRegressionModel"].forecast_data is not None
        assert len(sig.models["LinearRegressionModel"].forecast_data) == 12

    def test_warning_unsupported_covariates(self, signal_data, future_cov):
        """ExponentialSmoothing does not support covariates — warning, not error."""
        from darts.models import ExponentialSmoothing
        sig = Signal(signal_data.copy())
        sig.set_covariates(future_covariates=future_cov)
        sig.validation_split("2018-01-01")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sig.apply_model(ExponentialSmoothing())
        warning_msgs = [str(x.message) for x in w]
        assert any("does not support future covariates" in m for m in warning_msgs)
        assert "ExponentialSmoothing" in sig.models
