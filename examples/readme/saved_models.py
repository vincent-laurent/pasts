import pandas as pd

from darts.datasets import ETTh1Dataset
# make sure to import all models even if not explicitly called in this code
# from darts.models import AutoARIMA, ExponentialSmoothing, XGBModel, VARIMA

from pasts.signal import Signal


if __name__ == '__main__':

    # ----- Univariate -----

    # ---- Load data ----
    series = ETTh1Dataset().load()
    df_all = pd.DataFrame(series.values())
    df_all.columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    df_all.index = series.time_index
    dt = df_all.resample('D').mean()[['OT']]

    signal = Signal(dt, path='examples/readme/ETTh1_univariate')

    # --- Get saved models ---
    signal.get_saved_models()
    signal.compute_conf_intervals(window_size=7)

    # --- Forecasts with models previously fitted only on train set ---
    signal.forecast("AggregatedModel", 100, save_model=True)
    signal.forecast("AutoARIMA", 100, save_model=True)
    signal.forecast("ExponentialSmoothing", 100, save_model=True)

    # --- Forecasts with models previously fitted only on entire dataset ---
    signal.forecast("AggregatedModel", 100)
    signal.forecast("AutoARIMA", 100)
    signal.forecast("ExponentialSmoothing", 100)
