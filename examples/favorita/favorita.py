"""
Benchmark pasts on the Corporación Favorita Grocery Sales dataset.
===================================================================

Dataset: https://www.kaggle.com/c/favorita-grocery-sales-forecasting
Expected location: data/favorita-grocery-sales-forecasting/

Predicts daily sales for **all items at once** as a single multivariate
Signal (one column per item, aggregated across all stores).

Usage
-----
    python examples/favorita/favorita.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from darts.models import ExponentialSmoothing, RandomForestModel
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from matplotlib import pyplot as plt

from pasts.signal import Signal
from pasts.components import LinearTrend
from pasts.components.aggregated_model import AggregatedModel

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join("data", "favorita-grocery-sales-forecasting")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
ITEMS_PATH = os.path.join(DATA_DIR, "items.csv")
STORES_PATH = os.path.join(DATA_DIR, "stores.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "favorita_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading metadata ...")
items = pd.read_csv(ITEMS_PATH)
stores = pd.read_csv(STORES_PATH)
print(f"  {len(items)} items, {len(stores)} stores")

print("Loading train.csv (this may take a few minutes) ...")
t0 = time.time()
dtypes = {
    "item_nbr": "int32",
    "unit_sales": "float32",
}
train = pd.read_csv(
    TRAIN_PATH,
    dtype=dtypes,
    parse_dates=["date"],
    usecols=["date", "item_nbr", "unit_sales"],
)
train["unit_sales"] = train["unit_sales"].clip(lower=0)
print(f"  Loaded {len(train):,} rows in {time.time() - t0:.0f}s")

# ---------------------------------------------------------------------------
# Aggregate: daily sales per item (summed across all stores)
# ---------------------------------------------------------------------------
df = (
    train.groupby(["date", "item_nbr"])["unit_sales"]
    .sum()
    .unstack("item_nbr")
    .sort_index()
)
df.columns = [f"item_{c}" for c in df.columns]
del train

# Fill gaps with daily index
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
df = df.reindex(full_idx, fill_value=0)
df.index.freq = "D"

# Keep only items with meaningful sales
total_sales = df.sum()
df = df.loc[:, total_sales > 100]

print(f"\nSignal shape: {df.shape}  ({df.shape[1]} items after filtering)")

# ---------------------------------------------------------------------------
# Group items by correlation similarity
# ---------------------------------------------------------------------------
N_GROUPS = 300
print(f"\nClustering {df.shape[1]} items into ~{N_GROUPS} groups by correlation ...")

corr = df.corr().fillna(0)
dist = 1 - corr.abs()
np.fill_diagonal(dist.values, 0)
dist = dist.clip(lower=0)
condensed = squareform(dist.values)
Z = linkage(condensed, method="ward")
labels = fcluster(Z, t=N_GROUPS, criterion="maxclust")
group_map = pd.Series(labels, index=df.columns)

for g in sorted(group_map.unique()):
    cols = group_map[group_map == g].index.tolist()
    print(f"  Group {g}: {len(cols)} items")

# ---------------------------------------------------------------------------
# Validation split timestamp (shared across groups)
# ---------------------------------------------------------------------------
timestamp = str(df.index[-1] - pd.Timedelta(days=60))[:10]
print(f"\nValidation split at {timestamp}")

# ---------------------------------------------------------------------------
# Fit one model per group
# ---------------------------------------------------------------------------
for g in sorted(group_map.unique()):
    cols = group_map[group_map == g].index.tolist()
    print(f"\n--- Group {g} ({len(cols)} items) ---")

    group_df = df[cols]
    signal = Signal(group_df, path=os.path.join(OUTPUT_DIR, f"group_{g}"))
    signal.handle_nan(method="fill")
    signal.validation_split(timestamp=timestamp)

    signal.decompose()
    signal.residual -= LinearTrend(lags=200)

    t1 = time.time()
    signal.apply_model(RandomForestModel(lags=100, n_jobs=-1, n_estimators=20, max_depth=10))
    print(f"  LightGBM fit in {time.time() - t1:.1f}s")

    signal.compute_scores()
    uw = signal.performance_models.get("unit_wise", {})
    if "rmse" in uw:
        rmse_df = uw["rmse"]
        print(f"  RMSE mean: {rmse_df.iloc[:, 0].mean():.2f}")

    # Save per-item plots for first 3 items of each group
    n_train = len(signal.train_data)
    n_test = len(signal.test_data)
    # Trend from the registered LinearTrend (fitted on train)
    trend_train = signal.models["RandomForestModel"].estimator_on_train.reverse_transform(-n_train)
    trend_test = signal.models["RandomForestModel"].estimator_on_train.reverse_transform(n_test)
    trend_test.index = signal.test_data.index
    # Model predictions on test
    model_name = [k for k in signal.models if k != "default"][0]
    pred = signal.models[model_name]["predictions"]
    # Retrofit: model fitted values on train period
    retrofit = signal.models[model_name].estimator_on_train.reverse_transform(-n_train)

    for col in cols[:3]:
        fig, ax = plt.subplots(figsize=(12, 4))
        signal.train_data[col].plot(ax=ax, color="gray", label="train")
        signal.test_data[col].plot(ax=ax, color="black", label="test")
        pd.concat([trend_train[col], trend_test[col]]).plot(
            ax=ax, color="orange", linestyle="--", label="trend")
        retrofit[col].plot(ax=ax, color="tab:blue", alpha=0.5, label=f"{model_name} (retrofit)")
        pred[col].plot(ax=ax, color="tab:blue", label=f"{model_name} (pred)")
        ax.set_title(f"{col} (group {g})")
        ax.legend(fontsize=8)
        ax.set_xlim(("2016", "2018"))
        fig.savefig(os.path.join(OUTPUT_DIR, f"{col}_group{g}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {col}_group{g}.png")

print("\nDone.")
