import numpy as np
import pandas as pd
import pytest

from pasts.components.trend import (
    Differencing,
    HPFilterTrend,
    HighPassFilterTrend,
    LinearTrend,
    MovingAverageTrend,
    STLTrend,
)
from pasts.signal import Signal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def univariate_ts():
    """Univariate series with a clear linear trend + seasonality."""
    n = 120
    t = np.arange(n, dtype=float)
    trend = 0.5 * t + 10
    season = 5 * np.sin(2 * np.pi * t / 12)
    values = trend + season
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    return pd.DataFrame({"value": values}, index=idx)


@pytest.fixture
def multivariate_ts():
    """Multivariate (3-column) series with different trends."""
    n = 120
    t = np.arange(n, dtype=float)
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    return pd.DataFrame({
        "a": 0.3 * t + 5 + 2 * np.sin(2 * np.pi * t / 12),
        "b": -0.2 * t + 50 + 3 * np.cos(2 * np.pi * t / 6),
        "c": 0.1 * t ** 0.5 + 20,
    }, index=idx)


@pytest.fixture
def ts_with_nan(univariate_ts):
    """Univariate series with some NaN values."""
    df = univariate_ts.copy()
    df.iloc[5] = np.nan
    df.iloc[30] = np.nan
    df.iloc[60] = np.nan
    return df


@pytest.fixture
def numeric_index_ts():
    """Series with a plain numeric index (not DatetimeIndex)."""
    n = 100
    t = np.arange(n, dtype=float)
    values = 2 * t + 10 + np.random.default_rng(42).normal(0, 1, n)
    return pd.DataFrame({"x": values}, index=pd.Index(t))


# ---------------------------------------------------------------------------
# LinearTrend
# ---------------------------------------------------------------------------

class TestLinearTrend:
    def test_fit_reverse_univariate(self, univariate_ts):
        m = LinearTrend().fit(univariate_ts)
        trend = m.reverse_transform(-len(univariate_ts))
        assert trend.shape == univariate_ts.shape
        assert list(trend.columns) == list(univariate_ts.columns)

    def test_fit_reverse_multivariate(self, multivariate_ts):
        m = LinearTrend().fit(multivariate_ts)
        trend = m.reverse_transform(-len(multivariate_ts))
        assert trend.shape == multivariate_ts.shape

    def test_future(self, univariate_ts):
        m = LinearTrend().fit(univariate_ts)
        future = m.reverse_transform(10)
        assert len(future) == 10
        assert future.index[0] > univariate_ts.index[-1]

    def test_lags(self, univariate_ts):
        m_full = LinearTrend().fit(univariate_ts)
        m_lags = LinearTrend(lags=60).fit(univariate_ts)
        # Both produce correct shape
        trend_full = m_full.reverse_transform(-len(univariate_ts))
        trend_lags = m_lags.reverse_transform(-len(univariate_ts))
        assert trend_full.shape == trend_lags.shape
        # Coefficients differ (different fitting window)
        assert not np.allclose(m_full.coef_, m_lags.coef_)

    def test_lags_future(self, univariate_ts):
        m = LinearTrend(lags=60).fit(univariate_ts)
        future = m.reverse_transform(12)
        assert len(future) == 12
        assert future.index[0] > univariate_ts.index[-1]


# ---------------------------------------------------------------------------
# MovingAverageTrend
# ---------------------------------------------------------------------------

class TestMovingAverageTrend:
    def test_fit_reverse_univariate(self, univariate_ts):
        m = MovingAverageTrend(window=12).fit(univariate_ts)
        trend = m.reverse_transform(-len(univariate_ts))
        assert trend.shape == univariate_ts.shape
        assert list(trend.columns) == list(univariate_ts.columns)
        assert trend.index.equals(univariate_ts.index)

    def test_fit_reverse_multivariate(self, multivariate_ts):
        m = MovingAverageTrend(window=12).fit(multivariate_ts)
        trend = m.reverse_transform(-len(multivariate_ts))
        assert trend.shape == multivariate_ts.shape
        assert list(trend.columns) == list(multivariate_ts.columns)

    def test_future_constant(self, univariate_ts):
        m = MovingAverageTrend(window=12, extrapolation='constant').fit(univariate_ts)
        future = m.reverse_transform(10)
        assert len(future) == 10
        # Constant extrapolation: all rows identical
        assert np.allclose(future.values, future.iloc[0].values)
        # Future index starts after last historical index
        assert future.index[0] > univariate_ts.index[-1]

    def test_future_linear(self, univariate_ts):
        m = MovingAverageTrend(window=12, extrapolation='linear').fit(univariate_ts)
        future = m.reverse_transform(10)
        assert len(future) == 10
        assert future.index[0] > univariate_ts.index[-1]

    def test_nan_handling(self, ts_with_nan):
        m = MovingAverageTrend(window=12).fit(ts_with_nan)
        trend = m.reverse_transform(-len(ts_with_nan))
        assert trend.shape == ts_with_nan.shape

    def test_numeric_index(self, numeric_index_ts):
        m = MovingAverageTrend(window=10).fit(numeric_index_ts)
        future = m.reverse_transform(5)
        assert len(future) == 5
        assert future.index[0] > numeric_index_ts.index[-1]

    def test_decomposition_workflow(self, univariate_ts):
        signal = Signal(univariate_ts)
        signal.decompose()
        signal.residual -= MovingAverageTrend(window=12)
        assert len(signal.residual._ops) == 1
        # Verify fit + residual works
        signal.residual.fit(univariate_ts)
        residual = signal.residual.residual
        assert residual.std().iloc[0] < univariate_ts.std().iloc[0]


# ---------------------------------------------------------------------------
# HPFilterTrend
# ---------------------------------------------------------------------------

class TestHPFilterTrend:
    def test_fit_reverse_univariate(self, univariate_ts):
        m = HPFilterTrend(lamb=1600).fit(univariate_ts)
        trend = m.reverse_transform(-len(univariate_ts))
        assert trend.shape == univariate_ts.shape

    def test_fit_reverse_multivariate(self, multivariate_ts):
        m = HPFilterTrend(lamb=1600).fit(multivariate_ts)
        trend = m.reverse_transform(-len(multivariate_ts))
        assert trend.shape == multivariate_ts.shape
        assert list(trend.columns) == list(multivariate_ts.columns)

    def test_nan_handling(self, ts_with_nan):
        m = HPFilterTrend(lamb=1600).fit(ts_with_nan)
        trend = m.reverse_transform(-len(ts_with_nan))
        # NaN positions should be preserved
        assert trend.iloc[5].isna().all()
        assert trend.iloc[30].isna().all()

    def test_future(self, univariate_ts):
        m = HPFilterTrend(lamb=1600).fit(univariate_ts)
        future = m.reverse_transform(12)
        assert len(future) == 12
        assert future.index[0] > univariate_ts.index[-1]


# ---------------------------------------------------------------------------
# STLTrend
# ---------------------------------------------------------------------------

class TestSTLTrend:
    def test_fit_reverse_univariate(self, univariate_ts):
        m = STLTrend(period=12).fit(univariate_ts)
        trend = m.reverse_transform(-len(univariate_ts))
        assert trend.shape == univariate_ts.shape

    def test_fit_reverse_multivariate(self, multivariate_ts):
        m = STLTrend(period=12).fit(multivariate_ts)
        trend = m.reverse_transform(-len(multivariate_ts))
        assert trend.shape == multivariate_ts.shape

    def test_nan_handling(self, ts_with_nan):
        m = STLTrend(period=12).fit(ts_with_nan)
        trend = m.reverse_transform(-len(ts_with_nan))
        assert trend.iloc[5].isna().all()

    def test_future(self, univariate_ts):
        m = STLTrend(period=12).fit(univariate_ts)
        future = m.reverse_transform(12)
        assert len(future) == 12

    def test_decomposition_workflow(self, univariate_ts):
        signal = Signal(univariate_ts)
        signal.decompose()
        signal.residual -= STLTrend(period=12)
        assert len(signal.residual._ops) == 1
        signal.residual.fit(univariate_ts)
        residual = signal.residual.residual
        assert residual.std().iloc[0] < univariate_ts.std().iloc[0]


# ---------------------------------------------------------------------------
# HighPassFilterTrend
# ---------------------------------------------------------------------------

class TestHighPassFilterTrend:
    def test_fit_reverse_univariate(self, univariate_ts):
        # fs=12 (monthly), cutoff=0.5 (remove frequencies below 0.5/year)
        m = HighPassFilterTrend(cutoff=0.5, fs=12).fit(univariate_ts)
        trend = m.reverse_transform(-len(univariate_ts))
        assert trend.shape == univariate_ts.shape

    def test_fit_reverse_multivariate(self, multivariate_ts):
        m = HighPassFilterTrend(cutoff=0.5, fs=12).fit(multivariate_ts)
        trend = m.reverse_transform(-len(multivariate_ts))
        assert trend.shape == multivariate_ts.shape

    def test_nan_handling(self, ts_with_nan):
        m = HighPassFilterTrend(cutoff=0.5, fs=12).fit(ts_with_nan)
        trend = m.reverse_transform(-len(ts_with_nan))
        assert trend.iloc[5].isna().all()

    def test_future(self, univariate_ts):
        m = HighPassFilterTrend(cutoff=0.5, fs=12).fit(univariate_ts)
        future = m.reverse_transform(12)
        assert len(future) == 12


# ---------------------------------------------------------------------------
# Differencing
# ---------------------------------------------------------------------------

class TestDifferencing:
    def test_forward_produces_correct_diffs(self, univariate_ts):
        diff = Differencing(order=1).fit(univariate_ts)
        forward = diff.forward(univariate_ts)
        assert len(forward) == len(univariate_ts) - 1
        # forward values should equal manual diff
        expected = univariate_ts.diff().iloc[1:]
        pd.testing.assert_frame_equal(forward, expected, atol=1e-10)

    def test_forward_order2(self, univariate_ts):
        diff = Differencing(order=2).fit(univariate_ts)
        forward = diff.forward(univariate_ts)
        assert len(forward) == len(univariate_ts) - 2
        expected = univariate_ts.diff().diff().iloc[2:]
        pd.testing.assert_frame_equal(forward, expected, atol=1e-10)

    def test_multivariate(self, multivariate_ts):
        diff = Differencing(order=1).fit(multivariate_ts)
        forward = diff.forward(multivariate_ts)
        assert forward.shape == (len(multivariate_ts) - 1, 3)
        expected = multivariate_ts.diff().iloc[1:]
        pd.testing.assert_frame_equal(forward, expected, atol=1e-10)

    def test_future_inverse(self, univariate_ts):
        """Verify that inverse correctly reconstructs from future diffs."""
        diff = Differencing(order=1).fit(univariate_ts)
        # Simulate future diffs (constant increments)
        future_idx = pd.date_range(
            start=univariate_ts.index[-1], periods=6, freq="MS"
        )[1:]
        future_diffs = pd.DataFrame(
            {"value": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=future_idx
        )
        result = diff.inverse(future_diffs)
        # First value should be last_anchor + 1
        last_val = univariate_ts.iloc[-1, 0]
        expected = last_val + np.cumsum([1.0, 1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(result["value"].values, expected, atol=1e-10)

    def test_with_datacube_apply(self, univariate_ts):
        signal = Signal(univariate_ts)
        signal.decompose()
        diff = Differencing(order=1).fit(signal.data)
        signal.residual.apply(diff.forward, diff.inverse)
        assert len(signal.residual._ops) == 1
        assert signal.residual._ops[0][0] == 'unary'

    def test_nan_propagation(self, ts_with_nan):
        diff = Differencing(order=1).fit(ts_with_nan)
        forward = diff.forward(ts_with_nan)
        # NaN should propagate: positions adjacent to NaN in original become NaN
        assert forward.isna().any().any()


# ---------------------------------------------------------------------------
# EMDTrend (skip if PyEMD not installed)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# forecast_model parameter
# ---------------------------------------------------------------------------

class TestForecastModel:
    def test_moving_average_with_forecast_model(self, univariate_ts):
        from darts.models import ExponentialSmoothing
        m = MovingAverageTrend(window=12, forecast_model=ExponentialSmoothing())
        m.fit(univariate_ts)
        future = m.reverse_transform(10)
        assert len(future) == 10
        assert future.index[0] > univariate_ts.index[-1]
        # Should not be constant (unlike 'constant' strategy)
        assert not np.allclose(future.values, future.iloc[0].values)

    def test_hp_filter_with_forecast_model(self, univariate_ts):
        from darts.models import ExponentialSmoothing
        m = HPFilterTrend(lamb=1600, forecast_model=ExponentialSmoothing())
        m.fit(univariate_ts)
        future = m.reverse_transform(12)
        assert len(future) == 12
        assert future.index[0] > univariate_ts.index[-1]

    def test_stl_with_forecast_model(self, univariate_ts):
        from darts.models import ExponentialSmoothing
        m = STLTrend(period=12, forecast_model=ExponentialSmoothing())
        m.fit(univariate_ts)
        future = m.reverse_transform(10)
        assert len(future) == 10

    def test_forecast_model_multivariate(self, multivariate_ts):
        from darts.models import NaiveDrift
        m = MovingAverageTrend(window=12, forecast_model=NaiveDrift())
        m.fit(multivariate_ts)
        future = m.reverse_transform(10)
        assert future.shape == (10, 3)

    def test_backward_compat_no_forecast_model(self, univariate_ts):
        """Without forecast_model, behaviour is unchanged."""
        m = MovingAverageTrend(window=12, extrapolation='constant')
        m.fit(univariate_ts)
        future = m.reverse_transform(10)
        assert np.allclose(future.values, future.iloc[0].values)


# ---------------------------------------------------------------------------
# EMDTrend (skip if PyEMD not installed)
# ---------------------------------------------------------------------------

class TestEMDTrend:
    @pytest.fixture(autouse=True)
    def _skip_if_no_pyemd(self):
        pytest.importorskip("PyEMD")

    def test_fit_reverse_univariate(self, univariate_ts):
        from pasts.components.trend import EMDTrend
        m = EMDTrend().fit(univariate_ts)
        trend = m.reverse_transform(-len(univariate_ts))
        assert trend.shape == univariate_ts.shape

    def test_fit_reverse_multivariate(self, multivariate_ts):
        from pasts.components.trend import EMDTrend
        m = EMDTrend().fit(multivariate_ts)
        trend = m.reverse_transform(-len(multivariate_ts))
        assert trend.shape == multivariate_ts.shape
