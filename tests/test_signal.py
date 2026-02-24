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
        test_data=signal.test_data,
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
        test_data=signal.test_data,
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
        test_data=signal.test_data,
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
        test_data=signal.test_data,
    ))
    signal.forecast('AggregatedModel', 12)
    assert len(signal.models['AggregatedModel']['forecast']) == 12
    assert signal.models['AggregatedModel']['forecast'].index[0] > signal.data.index[-1]


def test_decomposition_learn_forecast(get_univariate_data):
    from pasts.components.trend import LinearTrend

    signal = Signal(get_univariate_data, 'tests')
    signal.decompose()

    # Subtract trend from residual
    trend = LinearTrend().fit(signal.data)
    signal.residual -= trend

    # Learn a model on the residual
    signal.residual.learn(ExponentialSmoothing())
    assert signal.residual._learned_model is not None

    # Forecast through decomposition
    signal.forecast(horizon=12)

    model_name = signal.residual._learned_model.name
    assert model_name in signal.models
    assert 'forecast' in signal.models[model_name]
    assert len(signal.models[model_name]['forecast']) == 12
    # Forecast index should be in the future
    assert signal.models[model_name]['forecast'].index[0] > signal.data.index[-1]


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
