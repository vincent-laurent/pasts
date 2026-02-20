import numpy as np
import pandas as pd
import pytest

from pasts.core.datacube import DataCube, _to_dataframe
from pasts.core.decomposition import Decomposition
from pasts.components import Trend


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
# Decomposition — compose roundtrip
# ---------------------------------------------------------------------------

class TestDecompositionCompose:

    def test_roundtrip_sub(self, simple_df):
        """signal -= C  →  compose should add C back."""
        original = simple_df.copy()
        r = DataCube(simple_df.copy())
        comp = DataCube(pd.DataFrame(10.0, index=simple_df.index, columns=simple_df.columns))
        r -= comp
        decomp = Decomposition(r._ops)
        reconstructed = decomp.compose(r)
        np.testing.assert_allclose(reconstructed.data.values, original.values)

    def test_roundtrip_sub_div(self, simple_df, alpha_dc):
        """signal -= T; signal /= alpha  →  roundtrip."""
        original = simple_df.copy()
        r = DataCube(simple_df.copy())
        trend = Trend().fit(simple_df)
        r -= trend
        r /= alpha_dc
        decomp = Decomposition(r._ops)
        reconstructed = decomp.compose(r)
        np.testing.assert_allclose(reconstructed.data.values, original.values, atol=1e-10)

    def test_roundtrip_with_log(self, simple_df):
        """signal -= offset; apply(log, exp)  →  roundtrip."""
        original = simple_df.copy()
        r = DataCube(simple_df.copy())
        # Ensure positive values before log
        min_val = simple_df.min().min()
        offset = DataCube(pd.DataFrame(
            min_val - 1.0, index=simple_df.index, columns=simple_df.columns
        ))
        r -= offset
        r.apply(np.log, np.exp)
        decomp = Decomposition(r._ops)
        reconstructed = decomp.compose(r)
        np.testing.assert_allclose(reconstructed.data.values, original.values, atol=1e-10)

    def test_roundtrip_scalar(self, simple_df):
        """Operations with scalars."""
        original = simple_df.copy()
        r = DataCube(simple_df.copy())
        r -= 50
        r /= 2
        decomp = Decomposition(r._ops)
        reconstructed = decomp.compose(r)
        np.testing.assert_allclose(reconstructed.data.values, original.values, atol=1e-10)


# ---------------------------------------------------------------------------
# Decomposition — repr
# ---------------------------------------------------------------------------

class TestDecompositionRepr:

    def test_repr_simple(self, simple_df):
        r = DataCube(simple_df.copy())
        trend = Trend().fit(simple_df)
        r -= trend
        decomp = Decomposition(r._ops)
        text = repr(decomp)
        assert "residual" in text
        assert "Trend" in text

    def test_repr_with_unary(self, simple_df):
        r = DataCube(simple_df.copy())
        offset = DataCube(pd.DataFrame(
            simple_df.min().min() - 1, index=simple_df.index, columns=simple_df.columns
        ))
        r -= offset
        r.apply(np.log, np.exp)
        decomp = Decomposition(r._ops)
        text = repr(decomp)
        assert "exp" in text
        assert "residual" in text


# ---------------------------------------------------------------------------
# Decomposition — components
# ---------------------------------------------------------------------------

class TestDecompositionComponents:

    def test_components_list(self, simple_df, alpha_dc):
        r = DataCube(simple_df.copy())
        trend = Trend().fit(simple_df)
        r -= trend
        r /= alpha_dc
        decomp = Decomposition(r._ops)
        comps = decomp.components()
        assert len(comps) == 2
        assert comps[0] is trend
        assert comps[1] is alpha_dc


# ---------------------------------------------------------------------------
# _to_dataframe
# ---------------------------------------------------------------------------

class TestToDataframe:

    def test_datacube(self, simple_df):
        dc = DataCube(simple_df)
        result = _to_dataframe(dc, simple_df.index)
        pd.testing.assert_frame_equal(result, simple_df)

    def test_fitter(self, simple_df):
        trend = Trend().fit(simple_df)
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

    def test_unknown_type_returns_not_implemented(self, simple_df):
        dc = DataCube(simple_df)
        assert dc.__add__("string") is NotImplemented
        assert dc.__mul__("string") is NotImplemented


# ---------------------------------------------------------------------------
# Signal integration
# ---------------------------------------------------------------------------

class TestSignalDecomposition:

    def test_decompose_creates_residual(self, simple_df, tmp_path):
        from pasts.signal import Signal
        signal = Signal(simple_df, path=str(tmp_path))
        signal.decompose()
        assert signal.residual is not None
        from pasts.signal import Signal
        assert isinstance(signal.residual, Signal)
        pd.testing.assert_frame_equal(signal.residual.data, simple_df)

    def test_decomposition_property(self, simple_df, tmp_path):
        from pasts.signal import Signal
        signal = Signal(simple_df, path=str(tmp_path))
        signal.decompose()
        trend = Trend().fit(simple_df)
        signal.residual -= trend
        decomp = signal.decomposition
        assert isinstance(decomp, Decomposition)
        assert len(decomp.components()) == 1

    def test_decomposition_raises_without_decompose(self, simple_df, tmp_path):
        from pasts.signal import Signal
        signal = Signal(simple_df, path=str(tmp_path))
        with pytest.raises(AttributeError):
            _ = signal.decomposition

    def test_full_roundtrip_via_signal(self, simple_df, tmp_path):
        from pasts.signal import Signal
        signal = Signal(simple_df, path=str(tmp_path))
        signal.decompose()
        trend = Trend().fit(simple_df)
        signal.residual -= trend
        signal.residual /= 2
        R = DataCube(signal.residual.data.copy())
        reconstructed = signal.decomposition.compose(R)
        np.testing.assert_allclose(reconstructed.data.values, simple_df.values, atol=1e-10)
