import numpy as np
from darts.datasets import AirPassengersDataset

from pasts.components import DartsModel, LinearTrend, NonParametricTrend
from pasts.components.aggregated_model import AggregatedModel
from pasts.core.base_model import TimeSeriesModel


def test_trend():
    trend = LinearTrend()
    series = AirPassengersDataset().load()
    dataframe = series.to_dataframe()
    dataframe["#Passengers2"] = dataframe["#Passengers"]
    dataframe["#Passengers2"] *= 100
    i = 10
    trend.fit(X=dataframe)
    assert trend.transform(i).shape[0] == i
    assert np.abs(trend.coef_[0] * 100 - trend.coef_[1]) < 0.001
    assert len(trend.coef_) == dataframe.shape[1]
    assert len(trend.intercept_) == dataframe.shape[1]
    assert all(dataframe.columns == trend.transform(10).columns)


def test_hierarchy():
    assert issubclass(LinearTrend, TimeSeriesModel)
    assert issubclass(NonParametricTrend, TimeSeriesModel)
    assert issubclass(DartsModel, TimeSeriesModel)
    assert issubclass(AggregatedModel, TimeSeriesModel)
