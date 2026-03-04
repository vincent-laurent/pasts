"""
Demo: detrending avec extrapolation robuste sur ETTh1 (Oil Temperature).

Pour chaque methode de trend non-parametrique, on utilise un
forecast_model (AutoARIMA) pour extrapoler la tendance dans le futur
— au lieu des strategies naives (constant/linear).

AutoARIMA selectionne automatiquement l'ordre de differenciation et les
parametres AR/MA : il capture la dynamique de la tendance (niveau,
pente, acceleration) sans hypothese rigide sur sa forme.

Pipeline:
  1. Retire la tendance du signal (detrending non-parametrique)
  2. Entraine un ExponentialSmoothing sur le residu (phase TEST)
  3. Refit sur toutes les donnees puis forecast 50 jours (phase FORECAST)
  4. Affiche predictions et forecast
"""

import matplotlib.pyplot as plt
import pandas as pd
from darts.datasets import ETTh1Dataset
from darts.models import XGBModel, Prophet

from pasts.signal import Signal
from pasts.components.trend import (
    LinearTrend,
    HPFilterTrend,
    STLTrend,
    HighPassFilterTrend,
)

# ----- Chargement des donnees -----
series = ETTh1Dataset().load()
df_all = pd.DataFrame(series.values())
df_all.columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
df_all.index = series.time_index

# Resample journalier pour un calcul raisonnable
df_daily = df_all.resample('D').mean()

# Univarie : Oil Temperature
dt = df_daily[['OT']]

print("Shape:", dt.shape)
print(dt.head())

# ----- Configuration -----
timestamp = '2018-04-01'
horizon = 50

trends = {
    "Linear":    LinearTrend(lags=90),
    "HPFilter":  HPFilterTrend(lamb=14400, forecast_model=Prophet()),
    "STL":       STLTrend(period=30, forecast_model=Prophet()),
    "HighPass":  HighPassFilterTrend(cutoff=0.05, fs=12, forecast_model=Prophet()),
}

# ----- Signal + split -----
signal = Signal(dt)
signal.validation_split(timestamp=timestamp)

print(f"\nTrain: {len(signal.train_data)} points")
print(f"Test:  {len(signal.test_data)} points")

# ----- Appliquer chaque trend + modele -----
for name, trend in trends.items():
    print(f"\n--- {name} ---") 
    signal.decompose(name)
    signal.decompositions[name] -= trend
    signal.apply_model(XGBModel(lags=5), decomposition=name)

# ----- Scores -----
signal.compute_scores()
for model_name, result in signal.models.items():
    scores = result["scores"]["unit_wise"]
    if isinstance(scores, dict) and scores:
        for metric, df in scores.items():
            print(f"  {model_name} | {metric}:\n{df}")

# ----- Refit + Forecast -----
signal.refit()
for model_name in signal.models:
    signal.forecast(model_name, horizon)
    print(f"Forecast {model_name}: {len(signal.models[model_name]['forecast'])} points")

# ----- Visualisation -----
fig_pred = signal.plot.predictions(show_fitted=True)
fig_pred.suptitle("Predictions + fitted (phase test)")
fig_pred.tight_layout()
fig_pred.savefig("examples/trend/predictions.png", dpi=150)
print("\n-> examples/trend/predictions.png")

fig_fc = signal.plot.forecast()
fig_fc.suptitle("Forecast")
fig_fc.tight_layout()
fig_fc.savefig("examples/trend/forecast.png", dpi=150)
print("-> examples/trend/forecast.png")

# ----- Résidus (signal détrendé) -----
fig_res = signal.plot.residuals()
fig_res.suptitle("Residus (signal detrende)")
fig_res.tight_layout()
fig_res.savefig("examples/trend/residuals.png", dpi=150)
print("-> examples/trend/residuals.png")

# ----- Erreurs de prédiction en fonction du temps -----
fig_err, ax_err = plt.subplots(figsize=(12, 5))
test_data = signal.test_data
for model_name, result in signal.models.items():
    if result.predictions is not None:
        error = result.predictions - test_data
        for col in error.columns:
            ax_err.plot(error.index, error[col], label=model_name)
ax_err.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax_err.set_xlabel("Time")
ax_err.set_ylabel("Error (pred - actual)")
ax_err.legend()
fig_err.suptitle("Prediction errors over time")
fig_err.tight_layout()
fig_err.savefig("examples/trend/errors.png", dpi=150)
print("-> examples/trend/errors.png")
