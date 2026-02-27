import matplotlib.figure
import pytest
from matplotlib import pyplot as plt
from darts.models import ExponentialSmoothing, AutoARIMA

from pasts.signal import Signal
from pasts.components import LinearTrend
from pasts.components.aggregated_model import AggregatedModel


def test_errors_visualization(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data, 'tests')
    signal_m = Signal(get_multivariate_data, 'tests')
    with pytest.raises(ValueError):
        signal_m.plot.acf()
    with pytest.raises(ValueError):
        signal.plot.predictions()


def test_plot_signal(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data, 'tests')
    signal_m = Signal(get_multivariate_data, 'tests')

    fig = signal_m.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    # With decomposition
    signal.decompose()
    signal.residual -= LinearTrend()
    fig = signal.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_acf_plot(get_univariate_data):
    signal = Signal(get_univariate_data, 'tests')
    fig = signal.plot.acf()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_show_predictions(get_univariate_data):
    signal = Signal(get_univariate_data, 'tests')
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_model(ExponentialSmoothing())

    fig = signal.plot.predictions()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_show_forecast(get_univariate_data):
    signal = Signal(get_univariate_data, 'tests')
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_model(AggregatedModel(
        {'AutoARIMA': AutoARIMA(), 'ExponentialSmoothing': ExponentialSmoothing()},
    ))
    signal.forecast('AggregatedModel', 12)

    fig = signal.plot.forecast()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
