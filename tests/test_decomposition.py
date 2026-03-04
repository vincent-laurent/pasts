import numpy as np
import pandas as pd
import pytest

from pasts.core.datacube import DataCube, _to_dataframe
from pasts.core.decomposition import Decomposition, DecompositionModel
from pasts.components import LinearTrend
from pasts.signal import Signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    """A small DataFrame with a DatetimeIndex."""
    index = pd.date_range("2020-01-01", periods=50, freq="MS")
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {"value": np.arange(50, dtype=float) * 2 + 100 + rng.randn(50) * 3},
        index=index,
    )


@pytest.fixture
def alpha_dc(simple_df):
    """A constant-amplitude DataCube (same shape as simple_df)."""
    return DataCube(
        pd.DataFrame(2.0, index=simple_df.index, columns=simple_df.columns)
    )


# ---------------------------------------------------------------------------
# Residual — binary ops recording
# ---------------------------------------------------------------------------

class TestDataCubeBinary:

    def test_isub_records(self, simple_df):
        r = DataCube(simple_df.copy())
        comp = DataCube(pd.DataFrame(1.0, index=simple_df.index, columns=simple_df.columns))
        r -= comp
        assert len(r._ops) == 1
        assert r._ops[0][0] == 'binary'

    def test_itruediv_records(self, simple_df):
        r = DataCube(simple_df.copy())
        r /= 2
        assert len(r._ops) == 1
        assert r._ops[0][0] == 'binary'

    def test_imul_records(self, simple_df):
        r = DataCube(simple_df.copy())
        r *= 3
        assert len(r._ops) == 1

    def test_iadd_records(self, simple_df):
        r = DataCube(simple_df.copy())
        r += 5
        assert len(r._ops) == 1

    def test_chained_ops(self, simple_df):
        r = DataCube(simple_df.copy())
        r -= 10
        r /= 2
        r *= 3
        assert len(r._ops) == 3

    def test_isub_mutates_data(self, simple_df):
        r = DataCube(simple_df.copy())
        before = r.data.copy()
        r -= 10
        assert (r.data == before - 10).all().all()


# ---------------------------------------------------------------------------
# Residual — unary ops recording
# ---------------------------------------------------------------------------

class TestDataCubeUnary:

    def test_apply_records(self, simple_df):
        r = DataCube(simple_df.copy())
        r.apply(np.log, np.exp)
        assert len(r._ops) == 1
        assert r._ops[0][0] == 'unary'

    def test_apply_mutates(self, simple_df):
        r = DataCube(simple_df.copy())
        expected = np.log(simple_df)
        r.apply(np.log, np.exp)
        pd.testing.assert_frame_equal(r.data, expected)

    def test_apply_returns_self(self, simple_df):
        r = DataCube(simple_df.copy())
        ret = r.apply(np.log, np.exp)
        assert ret is r


# ---------------------------------------------------------------------------
# Decomposition — auto-fit in operators
# ---------------------------------------------------------------------------

class TestDecompositionAutoFit:

    def test_isub_auto_fits_trend(self, simple_df):
        """Decomposition -= LinearTrend() auto-fits the trend."""
        decomp = Decomposition(simple_df.copy())
        decomp -= LinearTrend()
        # Ops should be recorded
        assert len(decomp._ops) == 1
        # The trend in _ops should be fitted (has coef_)
        component = decomp._ops[0][2]
        assert hasattr(component, 'coef_')

    def test_isub_records_ops(self, simple_df):
        decomp = Decomposition(simple_df.copy())
        decomp -= 10
        assert len(decomp._ops) == 1
        assert decomp._ops[0][0] == 'binary'


# ---------------------------------------------------------------------------
# Decomposition — compose roundtrip
# ---------------------------------------------------------------------------

class TestDecompositionCompose:

    def test_roundtrip_sub(self, simple_df):
        """signal -= C  →  compose should add C back."""
        original = simple_df.copy()
        decomp = Decomposition(simple_df.copy())
        comp = DataCube(pd.DataFrame(10.0, index=simple_df.index, columns=simple_df.columns))
        decomp -= comp
        reconstructed = decomp.compose(decomp)
        np.testing.assert_allclose(reconstructed.data.values, original.values)

    def test_roundtrip_sub_div(self, simple_df, alpha_dc):
        """signal -= T; signal /= alpha  →  roundtrip."""
        original = simple_df.copy()
        decomp = Decomposition(simple_df.copy())
        decomp -= LinearTrend()
        decomp /= alpha_dc
        reconstructed = decomp.compose(decomp)
        np.testing.assert_allclose(reconstructed.data.values, original.values, atol=1e-10)

    def test_roundtrip_with_log(self, simple_df):
        """signal -= offset; apply(log, exp)  →  roundtrip."""
        original = simple_df.copy()
        decomp = Decomposition(simple_df.copy())
        min_val = simple_df.min().min()
        offset = DataCube(pd.DataFrame(
            min_val - 1.0, index=simple_df.index, columns=simple_df.columns
        ))
        decomp -= offset
        decomp.apply(np.log, np.exp)
        reconstructed = decomp.compose(decomp)
        np.testing.assert_allclose(reconstructed.data.values, original.values, atol=1e-10)

    def test_roundtrip_scalar(self, simple_df):
        """Operations with scalars."""
        original = simple_df.copy()
        decomp = Decomposition(simple_df.copy())
        decomp -= 50
        decomp /= 2
        reconstructed = decomp.compose(decomp)
        np.testing.assert_allclose(reconstructed.data.values, original.values, atol=1e-10)


# ---------------------------------------------------------------------------
# Decomposition — repr
# ---------------------------------------------------------------------------

class TestDecompositionRepr:

    def test_repr_simple(self, simple_df):
        decomp = Decomposition(simple_df.copy())
        decomp -= LinearTrend()
        text = repr(decomp)
        assert "residual" in text
        assert "LinearTrend" in text

    def test_repr_with_unary(self, simple_df):
        decomp = Decomposition(simple_df.copy())
        offset = DataCube(pd.DataFrame(
            simple_df.min().min() - 1, index=simple_df.index, columns=simple_df.columns
        ))
        decomp -= offset
        decomp.apply(np.log, np.exp)
        text = repr(decomp)
        assert "exp" in text
        assert "residual" in text


# ---------------------------------------------------------------------------
# Decomposition — components
# ---------------------------------------------------------------------------

class TestDecompositionComponents:

    def test_components_list(self, simple_df, alpha_dc):
        decomp = Decomposition(simple_df.copy())
        decomp -= LinearTrend()
        decomp /= alpha_dc
        comps = decomp.components()
        assert len(comps) == 2


# ---------------------------------------------------------------------------
# Decomposition — residual property
# ---------------------------------------------------------------------------

class TestDecompositionResidual:

    def test_residual_before_fit_raises(self, simple_df):
        decomp = Decomposition(simple_df.copy())
        decomp -= 10
        with pytest.raises(ValueError, match="Call fit"):
            _ = decomp.residual

    def test_residual_after_fit(self, simple_df):
        decomp = Decomposition(simple_df.copy())
        decomp -= 10
        decomp.fit(simple_df)
        residual = decomp.residual
        assert isinstance(residual, pd.DataFrame)
        np.testing.assert_allclose(residual.values, (simple_df - 10).values)

    def test_residual_with_trend(self, simple_df):
        """fit() deep-copies and refits trend components, storing final residual."""
        decomp = Decomposition(simple_df.copy())
        decomp -= LinearTrend()
        decomp.fit(simple_df)
        residual = decomp.residual
        assert isinstance(residual, pd.DataFrame)
        assert len(residual) == len(simple_df)


# ---------------------------------------------------------------------------
# Decomposition — is a DataCube, not a TimeSeriesModel
# ---------------------------------------------------------------------------

class TestDecompositionIsNotModel:

    def test_decomposition_is_not_timeseriesmodel(self):
        from pasts.core.base_model import TimeSeriesModel
        assert not issubclass(Decomposition, TimeSeriesModel)

    def test_decomposition_is_datacube(self):
        assert issubclass(Decomposition, DataCube)


# ---------------------------------------------------------------------------
# Signal decomposition API
# ---------------------------------------------------------------------------

class TestSignalDecomposition:

    def test_decompose_creates_slot(self, simple_df):
        sig = Signal(simple_df.copy())
        sig.decompose()
        assert sig.residual is not None
        assert isinstance(sig.residual, DecompositionModel)

    def test_decompose_named(self, simple_df):
        sig = Signal(simple_df.copy())
        sig.decompose("trend")
        assert "trend" in sig.decompositions
        assert isinstance(sig.decompositions["trend"], DecompositionModel)

    def test_residual_operators(self, simple_df):
        sig = Signal(simple_df.copy())
        sig.decompose()
        sig.residual -= LinearTrend()
        assert len(sig.residual._ops) == 1


# ---------------------------------------------------------------------------
# _to_dataframe
# ---------------------------------------------------------------------------

class TestToDataframe:

    def test_datacube(self, simple_df):
        dc = DataCube(simple_df)
        result = _to_dataframe(dc, simple_df.index)
        pd.testing.assert_frame_equal(result, simple_df)

    def test_fitter(self, simple_df):
        trend = LinearTrend().fit(simple_df)
        result = _to_dataframe(trend, simple_df.index)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(simple_df)

    def test_scalar(self, simple_df):
        result = _to_dataframe(3.14, simple_df.index)
        assert result == 3.14

    def test_callable(self, simple_df):
        func = lambda idx: pd.DataFrame(1.0, index=idx, columns=simple_df.columns)
        result = _to_dataframe(func, simple_df.index)
        assert isinstance(result, pd.DataFrame)

    def test_unsupported_type(self, simple_df):
        with pytest.raises(TypeError):
            _to_dataframe("bad", simple_df.index)


# ---------------------------------------------------------------------------
# DataCube operator changes
# ---------------------------------------------------------------------------

class TestDataCubeOperators:

    def test_add_scalar(self, simple_df):
        dc = DataCube(simple_df)
        result = dc + 1
        assert isinstance(result, DataCube)
        np.testing.assert_allclose(result.data.values, simple_df.values + 1)

    def test_radd_scalar(self, simple_df):
        dc = DataCube(simple_df)
        result = 1 + dc
        assert isinstance(result, DataCube)

    def test_mul_scalar(self, simple_df):
        dc = DataCube(simple_df)
        result = dc * 2
        assert isinstance(result, DataCube)
        np.testing.assert_allclose(result.data.values, simple_df.values * 2)

    def test_rmul_scalar(self, simple_df):
        dc = DataCube(simple_df)
        result = 2 * dc
        assert isinstance(result, DataCube)

    def test_unknown_type_raises(self, simple_df):
        dc = DataCube(simple_df)
        with pytest.raises(Exception):
            dc + "string"
        with pytest.raises(Exception):
            dc * "string"


# ---------------------------------------------------------------------------
# Standalone model API: model.fit() / model.forecast()
# ---------------------------------------------------------------------------

class TestStandaloneModelAPI:

    def test_linear_trend_forecast(self, simple_df):
        """LinearTrend.fit(X).forecast(h) returns a DataFrame."""
        trend = LinearTrend()
        trend.fit(simple_df)
        forecast = trend.forecast(10)
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 10

    def test_darts_model_forecast(self, simple_df):
        """DartsModel.fit(X).forecast(h) works standalone."""
        from darts.models import ExponentialSmoothing
        from pasts.components.darts_model import DartsModel

        model = DartsModel(ExponentialSmoothing())
        model.fit(simple_df)
        forecast = model.forecast(5)
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 5

    def test_darts_model_with_decomposition_fit_forecast(self, simple_df):
        """DartsModel with decomposition: fit applies decomposition, forecast recomposes."""
        from darts.models import ExponentialSmoothing
        from pasts.components.darts_model import DartsModel

        decomp = Decomposition(simple_df.copy())
        decomp -= LinearTrend()

        model = DartsModel(ExponentialSmoothing(), decomposition=decomp)
        model.fit(simple_df)
        forecast = model.forecast(5)
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 5

    def test_darts_model_with_decomposition_reverse_transform_is_residual_space(self, simple_df):
        """reverse_transform stays in residual space (no recomposition)."""
        from darts.models import ExponentialSmoothing
        from pasts.components.darts_model import DartsModel

        decomp = Decomposition(simple_df.copy())
        decomp -= LinearTrend()

        model = DartsModel(ExponentialSmoothing(), decomposition=decomp)
        model.fit(simple_df)

        raw = model.reverse_transform(5)
        recomposed = model.forecast(5)
        # raw is in residual space, recomposed is in original space — they should differ
        assert not np.allclose(raw.values, recomposed.values)

    def test_darts_model_without_decomposition_forecast_equals_reverse_transform(self, simple_df):
        """Without decomposition, forecast == reverse_transform."""
        from darts.models import ExponentialSmoothing
        from pasts.components.darts_model import DartsModel

        model = DartsModel(ExponentialSmoothing())
        model.fit(simple_df)
        raw = model.reverse_transform(5)
        forecast = model.forecast(5)
        pd.testing.assert_frame_equal(raw, forecast)
