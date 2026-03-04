import copy
import os

import matplotlib.pyplot as plt
import pandas as pd

from darts.datasets import ETTh1Dataset
from darts.models import XGBModel, RandomForestModel, Prophet

from pasts.signal import Signal
from pasts.components.aggregated_model import AggregatedModel
from pasts.components import MovingAverageTrend, HighPassFilterTrend

IMG_DIR = "examples/readme"
trend = HighPassFilterTrend(cutoff=0.05, fs=12, forecast_model=RandomForestModel(lags=50))
lags = 500
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
signal.decompositions["MA_Trend"] -= trend
#--- Apply models on the named residual ---
# save_model=True indicates that the fitted estimator and its predictions will be saved
# in a joblib file in signal.path

signal.apply_model(XGBModel(lags=lags), save_model=True, decomposition="MA_Trend")
signal.apply_model(RandomForestModel(lags=lags), save_model=True, decomposition="MA_Trend")

#--- Plot historical trend + predicted trend ---
trend_fitted = copy.deepcopy(trend)
trend_fitted.fit(signal.data)
historical_trend = trend_fitted._trend
n_forecast = 100
future_trend = trend_fitted.reverse_transform(n_forecast)

fig, ax = plt.subplots()
signal.data.plot(ax=ax, color='gray', alpha=0.5, legend=False, label='_nolegend_')
historical_trend.plot(ax=ax, color='blue', linewidth=2, legend=False)
future_trend.plot(ax=ax, color='red', linestyle='--', linewidth=2, legend=False)
ax.axvline(x=signal.data.index[-1], color='black', linestyle=':', alpha=0.5)
labels = [f'raw: {c}' for c in signal.data.columns]
labels += [f'trend: {c}' for c in historical_trend.columns]
labels += [f'trend forecast: {c}' for c in future_trend.columns]
ax.legend(labels)
ax.set_xlabel('time')
ax.set_ylabel('values')
ax.set_title('Trend: historical + predicted')
fig.savefig(os.path.join(IMG_DIR, 'trend.png'), dpi=150, bbox_inches='tight')

signal.plot()



#--- Compute scores (on composed predictions vs original signal) ---
signal.compute_scores()

#--- Visualize predictions ---
fig = signal.plot.predictions()
fig.savefig(os.path.join(IMG_DIR, 'pred.png'), dpi=300, bbox_inches='tight')


#--- Aggregated Model ---
signal.apply_model(AggregatedModel(
    {'MA_Trend__XGBModel': signal.models['MA_Trend__XGBModel']['model'],
     'MA_Trend__RandomForestModel': signal.models['MA_Trend__RandomForestModel']['model']},
), save_model=True)
signal.compute_scores(axis=1)
signal.compute_conf_intervals()

#--- Forecast from end of train (covers test period + beyond) ---
signal.forecast("MA_Trend__XGBModel", 100, save_model=True)
signal.forecast("MA_Trend__RandomForestModel", 100, save_model=True)
signal.forecast("AggregatedModel", 100, save_model=True)

signal.compute_conf_intervals()

fig = signal.plot.forecast()
fig.savefig(os.path.join(IMG_DIR, 'fc.png'), dpi=300, bbox_inches='tight')

fig = signal.plot.forecast(aggregated_only=True)
fig.savefig(os.path.join(IMG_DIR, 'fc_ag.png'), dpi=300, bbox_inches='tight')


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
signal_m.decompositions["MA_Trend"] -= trend

#--- Apply models on the named residual ---
signal_m.apply_model(XGBModel(lags=lags), save_model=True, decomposition="MA_Trend")
signal_m.apply_model(RandomForestModel(lags=lags), save_model=True, decomposition="MA_Trend")

#--- Refit on full data, then forecast ---
signal_m.refit()
signal_m.forecast("MA_Trend__XGBModel", 50, save_model=True)
signal_m.forecast("MA_Trend__RandomForestModel", 50, save_model=True)

#--- Visualize forecasts ---
fig = signal_m.plot.forecast()
ax = fig.gca()
ax.set_xlim(("2018", "2018-08"))
fig.legend([])
fig.savefig("examples/readme/forecast.png")
