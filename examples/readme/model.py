import os

import pandas as pd

from darts.datasets import ETTh1Dataset
from darts.models import XGBModel, RandomForestModel

from pasts.signal import Signal
from pasts.components.aggregated_model import AggregatedModel
from pasts.components import MovingAverageTrend

IMG_DIR = "examples/readme"

#----- Load ETTh1 dataset -----
# Electricity Transformer Temperature: 7 variables (6 power loads + oil temperature)
# Hourly data from 2016-07 to 2018-06 (~17,400 observations)
series = ETTh1Dataset().load()
df_all = pd.DataFrame(series.values())
df_all.columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
df_all.index = series.time_index

# Resample to daily frequency for manageable computation
df_daily = df_all.resample('D').mean()

#----- Univariate -----
# Focus on Oil Temperature (OT), the main target variable

dt = df_daily[['OT']]
signal = Signal(dt, path='examples/readme/ETTh1_univariate')

#--- Analyze data ---
print(signal.properties)
fig = signal.plot()
fig.savefig(os.path.join(IMG_DIR, 'plot1.png'), dpi=150, bbox_inches='tight')

fig = signal.plot.acf()
fig.savefig(os.path.join(IMG_DIR, 'acf.png'), dpi=150, bbox_inches='tight')

signal.stat.test_stationarity()
signal.stat.test_stationarity(method='kpss')
signal.stat.test_seasonality()

#--- Split data between train and test ---
timestamp = '2018-04-01'
signal.validation_split(timestamp=timestamp)

#--- Named decomposition: remove trend, train models on residual ---
signal.decompose("MA_Trend")
signal.decompositions["MA_Trend"] -= MovingAverageTrend(30).fit(signal.data)

fig = signal.plot()
fig.savefig(os.path.join(IMG_DIR, 'op.png'), dpi=150, bbox_inches='tight')

#--- Apply models on the named residual ---
# save_model=True indicates that the fitted estimator and its predictions will be saved
# in a joblib file in signal.path

signal.decompositions["MA_Trend"].apply_model(XGBModel(lags=250), save_model=True)
signal.decompositions["MA_Trend"].apply_model(RandomForestModel(lags=250), save_model=True)

#--- Forecast: composes predictions + forecast back to original space ---
# signal.models["MA_Trend__XGBModel"] is now populated with composed predictions
signal.forecast("MA_Trend__XGBModel", 100, save_model=True)
signal.forecast("MA_Trend__RandomForestModel", 100, save_model=True)

#--- Compute scores (on composed predictions vs original signal) ---
signal.compute_scores()

#--- Visualize predictions ---
fig = signal.plot.predictions()
fig.savefig(os.path.join(IMG_DIR, 'pred.png'), dpi=150, bbox_inches='tight')

#--- Aggregated Model ---
signal.apply_model(AggregatedModel(
    {'MA_Trend__XGBModel': signal.models['MA_Trend__XGBModel']['model'],
     'MA_Trend__RandomForestModel': signal.models['MA_Trend__RandomForestModel']['model']},
    test_data=signal.test_data,
), save_model=True)
signal.compute_scores(axis=1)
signal.compute_conf_intervals()

#--- Additional forecasts ---
signal.forecast("AggregatedModel", 100, save_model=True)

signal.compute_conf_intervals()

fig = signal.plot.forecast()
fig.savefig(os.path.join(IMG_DIR, 'fc.png'), dpi=150, bbox_inches='tight')

fig = signal.plot.forecast(aggregated_only=True)
fig.savefig(os.path.join(IMG_DIR, 'fc_ag.png'), dpi=150, bbox_inches='tight')


#----- Multivariate -----
# Use 4 variables: power loads at 3 levels + oil temperature

df_m = df_daily[['HUFL', 'MUFL', 'LUFL', 'OT']]
signal_m = Signal(df_m, path='examples/readme/ETTh1_multivariate')

#--- Analyze data ---
print(signal_m.properties)
signal_m.stat.test_causality()
signal_m.plot()

#--- Split and decompose ---
timestamp = '2018-04-01'
signal_m.validation_split(timestamp=timestamp)

signal_m.decompose("MA_Trend")
signal_m.decompositions["MA_Trend"] -= MovingAverageTrend(30).fit(signal_m.data)

#--- Apply models on the named residual ---
lags = [-325]
signal_m.decompositions["MA_Trend"].apply_model(XGBModel(lags=lags), save_model=True)
signal_m.decompositions["MA_Trend"].apply_model(RandomForestModel(lags=lags), save_model=True)

#--- Forecast: composes predictions + forecast back to original space ---
signal_m.forecast("MA_Trend__XGBModel", 50, save_model=True)
signal_m.forecast("MA_Trend__RandomForestModel", 50, save_model=True)

#--- Visualize forecasts ---
fig = signal_m.plot.forecast()
ax = fig.gca()
ax.set_xlim(("2018", "2018-08"))
fig.savefig("examples/readme/forecast.png")
