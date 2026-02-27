import pandas as pd
import pytest
from darts.models import ExponentialSmoothing, AutoARIMA
from darts.utils.utils import ModelMode, SeasonalityMode
import math

from pasts.signal import Signal
from pasts.components.darts_model import DartsModel
from pasts.components.aggregated_model import AggregatedModel


def test_validation_split(get_univariate_data):
    signal = Signal(get_univariate_data, 'tests')
    tstamp = '1949-01-01'
    with pytest.raises(ValueError):
        signal.validation_split(tstamp)
    tstamp2 = '1958-12-01'
    signal.validation_split(tstamp2, n_splits_cv=5)
    assert signal.test_data.shape[0] == 24
    assert signal.train_data.shape[0] == 120


def test_apply_model(get_univariate_data):
    signal = Signal(get_univariate_data, 'tests')
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_model(ExponentialSmoothing())
    signal.apply_model(AutoARIMA())
    assert len(signal.models) == 2
    for model in signal.models.keys():
        assert len(signal.models[model]) == 4
        assert len(signal.models[model]['predictions']) == 24
        assert signal.models[model]['best_parameters'] == "default"
    assert isinstance(signal.models['ExponentialSmoothing']['model'], DartsModel)
    assert isinstance(signal.models['ExponentialSmoothing']['model']._model, ExponentialSmoothing)
    assert isinstance(signal.models['AutoARIMA']['model'], DartsModel)
    assert isinstance(signal.models['AutoARIMA']['model']._model, AutoARIMA)


def test_apply_model_grid(get_univariate_data):
    signal = Signal(get_univariate_data, 'tests')
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    with pytest.raises(Exception):
        signal.apply_model(model=ExponentialSmoothing(), gridsearch=True)
    param_grid = {'trend': [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
                  'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE],
                  }
    signal.apply_model(ExponentialSmoothing(), gridsearch=True, parameters=param_grid)
    assert len(signal.models) == 1
    assert len(signal.models['ExponentialSmoothing']) == 4
    assert len(signal.models['ExponentialSmoothing']['predictions']) == 24
    assert signal.models['ExponentialSmoothing']['best_parameters'] != "default"
    assert len(signal.models['ExponentialSmoothing']['best_parameters']) == 2
    assert isinstance(signal.models['ExponentialSmoothing']['model'], DartsModel)


def test_aggregated_model(get_univariate_data):
    signal = Signal(get_univariate_data, 'tests')
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_model(AggregatedModel(
        {'AutoARIMA': AutoARIMA(), 'ExponentialSmoothing': ExponentialSmoothing()},
    ))
    assert 'AggregatedModel' in signal.models
    assert len(signal.models['AggregatedModel']['predictions']) == 24
    assert signal.models['AggregatedModel']['weights'].shape == (1, 2)
    assert len(signal.models['AggregatedModel']['models']) == 2
    assert round(signal.models['AggregatedModel']['weights'].sum(axis=1)[0], 0) == 1


def test_scores_unit(get_univariate_data):
    signal = Signal(get_univariate_data, 'tests')
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_model(ExponentialSmoothing())
    signal.apply_model(AutoARIMA())
    signal.apply_model(AggregatedModel(
        {'AutoARIMA': AutoARIMA(), 'ExponentialSmoothing': ExponentialSmoothing()},
    ))

    signal.compute_scores(['r2', 'mse', 'mape'])
    assert signal.models['ExponentialSmoothing']['scores']['unit_wise'].shape[1] == 3
    assert signal.models['AutoARIMA']['scores']['unit_wise'].shape[1] == 3
    assert signal.models['AggregatedModel']['scores']['unit_wise'].shape[1] == 3
    assert len(signal.performance_models) == 1
    assert len(signal.performance_models['unit_wise']) == 3

    signal.compute_scores()
    for model in signal.models.keys():
        assert not signal.models[model]['scores']['time_wise']
        assert signal.models[model]['scores']['unit_wise'].shape[1] == 6
        assert signal.models[model]['scores']['unit_wise'].loc['passengers', 'rmse']**2 - 0.1 < \
               signal.models[model]['scores']['unit_wise'].loc['passengers', 'mse'] <\
               signal.models[model]['scores']['unit_wise'].loc['passengers', 'rmse']**2 + 0.1
    assert len(signal.performance_models) == 1
    assert len(signal.performance_models['unit_wise']) == 6
    for metric in signal.performance_models['unit_wise'].keys():
        assert signal.performance_models['unit_wise'][metric].shape == (1, 3)


def test_scores_time(get_univariate_data):
    signal = Signal(get_univariate_data, 'tests')
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_model(ExponentialSmoothing())
    signal.apply_model(AutoARIMA())
    signal.apply_model(AggregatedModel(
        {'AutoARIMA': AutoARIMA(), 'ExponentialSmoothing': ExponentialSmoothing()},
    ))

    signal.compute_scores(['r2', 'mse', 'mape'], axis=0)
    for model in signal.models.keys():
        assert signal.models[model]['scores']['time_wise'].shape[1] == 1
    assert len(signal.performance_models) == 1
    assert len(signal.performance_models['time_wise']) == 1

    signal.compute_scores(axis=0)
    for model in signal.models.keys():
        assert not signal.models[model]['scores']['unit_wise']
        assert signal.models[model]['scores']['time_wise'].shape[1] == 2
    assert len(signal.performance_models) == 1
    assert len(signal.performance_models['time_wise']) == 2
    for metric in signal.performance_models['time_wise'].keys():
        assert signal.performance_models['time_wise'][metric].shape == (24, 3)


def test_forecast(get_univariate_data):
    signal = Signal(get_univariate_data, 'tests')
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    with pytest.raises(Exception):
        signal.forecast('AggregatedModel', 12)
    signal.apply_model(AggregatedModel(
        {'AutoARIMA': AutoARIMA(), 'ExponentialSmoothing': ExponentialSmoothing()},
    ))
    signal.forecast('AggregatedModel', 12)
    assert len(signal.models['AggregatedModel']['forecast']) == 12
    assert signal.models['AggregatedModel']['forecast'].index[0] > signal.data.index[-1]


def test_decomposition_learn_forecast(get_univariate_data):
    from pasts.components.trend import LinearTrend

    signal = Signal(get_univariate_data, 'tests')
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.decompose()

    # Subtract trend from residual (auto-fitted on residual data)
    signal.residual -= LinearTrend()

    # Fit a model on the residual via apply_model
    signal.residual.apply_model(ExponentialSmoothing())
    assert 'ExponentialSmoothing' in signal.residual.models

    # Forecast through decomposition using the unified API
    signal.forecast('default__ExponentialSmoothing', 12)

    assert 'default__ExponentialSmoothing' in signal.models
    assert 'forecast' in signal.models['default__ExponentialSmoothing']
    assert len(signal.models['default__ExponentialSmoothing']['forecast']) == 12
    # Forecast index should be in the future
    assert signal.models['default__ExponentialSmoothing']['forecast'].index[0] > signal.data.index[-1]


def test_named_decompositions(get_univariate_data):
    from pasts.components.trend import LinearTrend, MovingAverageTrend

    signal = Signal(get_univariate_data, 'tests')
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)

    # Create two named decompositions
    signal.decompose("linear")
    signal.decompose("ma")

    assert "linear" in signal.decompositions
    assert "ma" in signal.decompositions
    assert isinstance(signal.decompositions["linear"], Signal)
    assert isinstance(signal.decompositions["ma"], Signal)

    # Residual property still points to "default" (None if not created)
    assert signal.residual is None

    # Apply different trends to each named residual
    signal.decompositions["linear"] -= LinearTrend()
    signal.decompositions["ma"] -= MovingAverageTrend(12)

    # Each residual has its own ops stack
    assert len(signal.decompositions["linear"]._ops) == 1
    assert len(signal.decompositions["ma"]._ops) == 1

    # get_decomposition() returns the correct Decomposition
    from pasts.core.decomposition import Decomposition
    assert isinstance(signal.get_decomposition("linear"), Decomposition)
    assert isinstance(signal.get_decomposition("ma"), Decomposition)
    with pytest.raises(AttributeError):
        signal.get_decomposition("nonexistent")

    # Split is already propagated from parent signal — no need to call
    # validation_split again on each decomposition.
    signal.decompositions["linear"].apply_model(ExponentialSmoothing())
    signal.decompositions["linear"].compute_scores()

    signal.decompositions["ma"].apply_model(ExponentialSmoothing())
    signal.decompositions["ma"].compute_scores()

    # Forecast through named decomposition: forecast("decomp__model", horizon)
    # → also composes test predictions into signal.models
    signal.forecast("linear__ExponentialSmoothing", 12)
    assert "linear__ExponentialSmoothing" in signal.models

    entry = signal.models["linear__ExponentialSmoothing"]
    # Forecast in future
    assert len(entry["forecast"]) == 12
    assert entry["forecast"].index[0] > signal.data.index[-1]
    # Predictions composed back to original signal space
    assert entry["predictions"] is not None
    assert len(entry["predictions"]) == 24  # test set size
    assert list(entry["predictions"].index) == list(signal.test_data.index)

    # compute_scores works on composed predictions vs signal.test_data
    signal.compute_scores()
    assert signal.models["linear__ExponentialSmoothing"]["scores"]["unit_wise"] is not None

    signal.forecast("ma__ExponentialSmoothing", 12)
    assert "ma__ExponentialSmoothing" in signal.models

    # Both forecasts coexist in signal.models
    assert len([k for k in signal.models if "__" in k]) == 2

    # Error when decomp_name or model_name not found
    with pytest.raises(ValueError, match="No decomposition"):
        signal.forecast("missing_decomp__ExponentialSmoothing", 12)
    with pytest.raises(ValueError, match="has not been trained"):
        signal.forecast("linear__AutoARIMA", 12)


def test_multistep_decomposition_autofit(get_univariate_data):
    """Multi-step decomposition: second model is fitted on the residual
    after the first step, not on the original signal."""
    from pasts.components.trend import LinearTrend, MovingAverageTrend

    signal = Signal(get_univariate_data, 'tests')
    signal.validation_split('1958-12-01')
    signal.decompose("multi")

    # Step 1: remove linear trend
    signal.decompositions["multi"] -= LinearTrend()
    residual_after_step1 = signal.decompositions["multi"].data.copy()

    # Step 2: remove MA trend from residual (NOT from original signal)
    signal.decompositions["multi"] -= MovingAverageTrend(12)

    # The ops stack should have 2 entries
    assert len(signal.decompositions["multi"]._ops) == 2

    # Verify the second model was fitted on the post-step1 residual
    second_model = signal.decompositions["multi"]._ops[1][2]
    expected = MovingAverageTrend(12).fit(residual_after_step1)
    expected_trend = expected.reverse_transform(-len(residual_after_step1))
    actual_trend = second_model.reverse_transform(-len(residual_after_step1))
    pd.testing.assert_frame_equal(actual_trend, expected_trend)


def test_properties(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data, 'tests')
    assert signal.properties['shape'] == (144, 1)
    assert signal.properties['is_univariate'] == True
    assert len(signal.properties) == 5
    signal_m = Signal(get_multivariate_data, 'tests')
    assert signal_m.properties['is_univariate'] == False


def test_stationarity_adfuller(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data, 'tests')
    result = signal.stat.test_stationarity(method='adfuller')
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['statistic', 'p_value', 'n_lags', 'n_obs', 'stationary']
    assert list(result.index) == ['passengers']
    assert result.loc['passengers', 'stationary'] == False
    assert result.loc['passengers', 'p_value'] > 0.95
    assert 'stationarity' in signal.tests_stat

    # Multivariate: works per-column
    signal_m = Signal(get_multivariate_data, 'tests')
    result_m = signal_m.stat.test_stationarity(method='adfuller')
    assert isinstance(result_m, pd.DataFrame)
    assert len(result_m) == 3
    assert set(result_m.index) == {'Hol', 'VFR', 'Oth'}


def test_stationarity_kpss(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data, 'tests')
    result = signal.stat.test_stationarity(method='kpss')
    assert isinstance(result, pd.DataFrame)
    assert result.loc['passengers', 'stationary'] == False
    assert result.loc['passengers', 'p_value'] < 0.05

    # Multivariate: works per-column
    signal_m = Signal(get_multivariate_data, 'tests')
    result_m = signal_m.stat.test_stationarity(method='kpss')
    assert len(result_m) == 3


def test_stationarity_invalid_method(get_univariate_data):
    signal = Signal(get_univariate_data, 'tests')
    with pytest.raises(ValueError, match="Unknown method"):
        signal.stat.test_stationarity(method='invalid')


def test_seasonality(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data, 'tests')
    result = signal.stat.test_seasonality()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['seasonal', 'period']
    assert result.loc['passengers', 'seasonal'] == True
    assert result.loc['passengers', 'period'] == 12
    assert 'seasonality' in signal.tests_stat

    # Multivariate: works per-column
    signal_m = Signal(get_multivariate_data, 'tests')
    result_m = signal_m.stat.test_seasonality()
    assert isinstance(result_m, pd.DataFrame)
    assert len(result_m) == 3


def test_causality(get_univariate_data, get_multivariate_data):
    # Univariate raises ValueError
    signal = Signal(get_univariate_data, 'tests')
    with pytest.raises(ValueError):
        signal.stat.test_causality()

    # Multivariate
    signal_m = Signal(get_multivariate_data, 'tests')
    result = signal_m.stat.test_causality()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['statistic', 'p_value', 'causal']
    assert len(result) == math.perm(3, 2)
    # Consistency: causal iff p_value <= 0.05
    for pair in result.index:
        if result.loc[pair, 'causal']:
            assert result.loc[pair, 'p_value'] <= 0.05
        else:
            assert result.loc[pair, 'p_value'] > 0.05
    assert 'causality' in signal_m.tests_stat


def test_normality(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data, 'tests')
    result = signal.stat.test_normality()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['statistic', 'p_value', 'normal']
    assert list(result.index) == ['passengers']
    assert 'normality' in signal.tests_stat

    # jarque_bera method
    result_jb = signal.stat.test_normality(method='jarque_bera')
    assert isinstance(result_jb, pd.DataFrame)
    assert list(result_jb.columns) == ['statistic', 'p_value', 'normal']

    # Invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        signal.stat.test_normality(method='invalid')

    # Multivariate: works per-column
    signal_m = Signal(get_multivariate_data, 'tests')
    result_m = signal_m.stat.test_normality()
    assert len(result_m) == 3


def test_whitenoise(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data, 'tests')
    result = signal.stat.test_whitenoise()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['statistic', 'p_value', 'white_noise']
    assert list(result.index) == ['passengers']
    # Air Passengers has strong autocorrelation
    assert result.loc['passengers', 'white_noise'] == False
    assert 'whitenoise' in signal.tests_stat

    # Multivariate: works per-column
    signal_m = Signal(get_multivariate_data, 'tests')
    result_m = signal_m.stat.test_whitenoise()
    assert len(result_m) == 3


def test_heteroscedasticity(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data, 'tests')
    result = signal.stat.test_heteroscedasticity()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['statistic', 'p_value', 'homoscedastic']
    assert list(result.index) == ['passengers']
    assert 'heteroscedasticity' in signal.tests_stat

    # Multivariate: works per-column
    signal_m = Signal(get_multivariate_data, 'tests')
    result_m = signal_m.stat.test_heteroscedasticity()
    assert len(result_m) == 3


def test_all(get_univariate_data, get_multivariate_data):
    # Univariate: 5 keys (no causality)
    signal = Signal(get_univariate_data, 'tests')
    results = signal.stat.all()
    assert isinstance(results, dict)
    expected_keys = {'stationarity', 'seasonality', 'normality', 'whitenoise', 'heteroscedasticity'}
    assert set(results.keys()) == expected_keys
    for key, df in results.items():
        assert isinstance(df, pd.DataFrame)
        assert key in signal.tests_stat

    # Multivariate: 6 keys (includes causality)
    signal_m = Signal(get_multivariate_data, 'tests')
    results_m = signal_m.stat.all()
    assert set(results_m.keys()) == expected_keys | {'causality'}
    for key, df in results_m.items():
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# ValidationAccessor + split propagation
# ---------------------------------------------------------------------------

def test_validation_accessor(get_univariate_data):
    """signal.validation.split() is equivalent to signal.validation_split()."""
    from pasts.validation import ValidationAccessor

    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    signal.validation.split(tstamp)

    # train_data / test_data are properties on Signal, not on the accessor
    assert signal.train_data is not None
    assert signal.test_data is not None
    assert signal.train_data.shape[0] == 120
    assert signal.test_data.shape[0] == 24
    assert isinstance(signal.validation, ValidationAccessor)
    assert signal.validation._timestamp == tstamp


def test_validation_accessor_cv(get_univariate_data):
    """cv_tseries is stored in the accessor and accessible via signal.validation.cv_tseries."""
    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    assert signal.validation.cv_tseries is None
    signal.validation.split(tstamp, n_splits_cv=3)
    assert signal.validation.cv_tseries is not None


def test_split_propagates_to_decompose_after(get_univariate_data):
    """decompose() after validation_split() propagates the split automatically."""
    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.decompose("t")

    decomp = signal.decompositions["t"]
    assert decomp.train_data is not None
    assert decomp.test_data is not None
    assert decomp.train_data.shape[0] == signal.train_data.shape[0]
    assert decomp.test_data.shape[0] == signal.test_data.shape[0]


def test_split_propagates_to_decompose_before(get_univariate_data):
    """decompose() before validation_split(): the shared accessor means the
    decomposition sees the split as soon as it is set on the parent."""
    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    signal.decompose("t")
    assert signal.decompositions["t"].train_data is None  # not yet split

    signal.validation_split(tstamp)
    # Shared accessor — decomposition immediately reflects the split
    decomp = signal.decompositions["t"]
    assert decomp.train_data is not None
    assert decomp.train_data.shape[0] == 120
    assert decomp.test_data.shape[0] == 24


# ---- Confidence interval tests -----------------------------------------

def _setup_signal_with_model(data, tstamp='1958-12-01'):
    """Helper: create a Signal, split, fit ExponentialSmoothing."""
    signal = Signal(data, 'tests')
    signal.validation_split(tstamp)
    signal.apply_model(ExponentialSmoothing())
    return signal


def _assert_ci_valid(ci_df, ref_df):
    """Assert a CI DataFrame has the right shape and (lower, upper) tuples."""
    assert ci_df is not None
    assert ci_df.shape == ref_df.shape
    for col in ci_df.columns:
        cell = ci_df[col].iloc[0]
        assert isinstance(cell, tuple), f"Expected tuple, got {type(cell)}"
        assert len(cell) == 2
        assert cell[0] < cell[1], f"lower >= upper: {cell}"


def test_conf_intervals_empirical(get_univariate_data):
    """Existing empirical method still works (regression test)."""
    signal = _setup_signal_with_model(get_univariate_data)
    signal.forecast('ExponentialSmoothing', 12)
    signal.compute_conf_intervals(method="empirical")

    result = signal.models['ExponentialSmoothing']
    _assert_ci_valid(result.test_confidence_interval, result.predictions)
    _assert_ci_valid(result.forecast_confidence_interval, result.forecast_data)
    # Empirical should not compute historical residuals
    assert result.historical_residuals is None


def test_conf_intervals_default_is_bootstrap_full(get_univariate_data):
    """Default (no method arg) is bootstrap_full."""
    signal = _setup_signal_with_model(get_univariate_data)
    signal.compute_conf_intervals(n_bootstrap=10, random_state=42)

    result = signal.models['ExponentialSmoothing']
    _assert_ci_valid(result.test_confidence_interval, result.predictions)
    assert result.historical_residuals is not None


def test_conf_intervals_bootstrap(get_univariate_data):
    """Bootstrap residual method produces valid intervals."""
    signal = _setup_signal_with_model(get_univariate_data)
    signal.forecast('ExponentialSmoothing', 12)
    signal.compute_conf_intervals(
        method="bootstrap", n_bootstrap=500, alpha=0.05, random_state=42
    )

    result = signal.models['ExponentialSmoothing']
    _assert_ci_valid(result.test_confidence_interval, result.predictions)
    _assert_ci_valid(result.forecast_confidence_interval, result.forecast_data)
    assert result.historical_residuals is not None
    assert len(result.historical_residuals) > 0


def test_conf_intervals_bootstrap_full(get_univariate_data):
    """Bootstrap full (refit) method produces valid intervals."""
    signal = _setup_signal_with_model(get_univariate_data)
    signal.compute_conf_intervals(
        method="bootstrap_full", n_bootstrap=10, alpha=0.05, random_state=42
    )

    result = signal.models['ExponentialSmoothing']
    _assert_ci_valid(result.test_confidence_interval, result.predictions)
    assert result.historical_residuals is not None


def test_conf_intervals_bootstrap_reproducible(get_univariate_data):
    """Same random_state produces identical results."""
    signal1 = _setup_signal_with_model(get_univariate_data)
    signal1.compute_conf_intervals(method="bootstrap", random_state=123)

    signal2 = _setup_signal_with_model(get_univariate_data)
    signal2.compute_conf_intervals(method="bootstrap", random_state=123)

    ci1 = signal1.models['ExponentialSmoothing'].test_confidence_interval
    ci2 = signal2.models['ExponentialSmoothing'].test_confidence_interval
    for col in ci1.columns:
        for i in range(len(ci1)):
            assert ci1[col].iloc[i] == ci2[col].iloc[i]


def test_conf_intervals_bootstrap_other_dataset(get_univariate_data):
    """Bootstrap CI with forecast on univariate data."""
    signal = _setup_signal_with_model(get_univariate_data)
    signal.forecast('ExponentialSmoothing', 6)
    signal.ci.compute(method="bootstrap", n_bootstrap=200, random_state=42)

    result = signal.models['ExponentialSmoothing']
    _assert_ci_valid(result.test_confidence_interval, result.predictions)
    _assert_ci_valid(result.forecast_confidence_interval, result.forecast_data)


def test_conf_intervals_invalid_method(get_univariate_data):
    """Invalid method raises ValueError."""
    signal = _setup_signal_with_model(get_univariate_data)
    with pytest.raises(ValueError, match="method must be"):
        signal.compute_conf_intervals(method="invalid")


def test_ci_accessor_compute(get_univariate_data):
    """signal.ci.compute() accessor works like compute_conf_intervals()."""
    signal = _setup_signal_with_model(get_univariate_data)
    signal.ci.compute(method="bootstrap", n_bootstrap=200, random_state=42)

    result = signal.models['ExponentialSmoothing']
    _assert_ci_valid(result.test_confidence_interval, result.predictions)
    assert result.historical_residuals is not None


def test_ci_accessor_default_is_bootstrap_full(get_univariate_data):
    """signal.ci.compute() defaults to bootstrap_full."""
    signal = _setup_signal_with_model(get_univariate_data)
    signal.ci.compute(n_bootstrap=10, random_state=42)

    result = signal.models['ExponentialSmoothing']
    _assert_ci_valid(result.test_confidence_interval, result.predictions)
    assert result.historical_residuals is not None


