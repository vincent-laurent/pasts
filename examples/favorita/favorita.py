"""
Benchmark pasts on the Corporación Favorita Grocery Sales dataset.
===================================================================

Dataset: https://www.kaggle.com/c/favorita-grocery-sales-forecasting
Expected location: data/favorita-grocery-sales-forecasting/

The script tests pasts at three aggregation levels that cover very
different time-series profiles:

1. **Store level** (54 series)  – smooth, high-volume daily sales
2. **Family × Store** (~1 800 series) – mid-level hierarchy, mixed volumes
3. **Curated items** – hand-picked profiles:
   - high-rotation grocery item
   - low-volume / intermittent item
   - product launched mid-dataset (cold start)

Usage
-----
    python examples/favorita/favorita.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from darts.models import ExponentialSmoothing, RandomForest

from matplotlib import pyplot as plt

from pasts.signal import Signal
from pasts.components import LinearTrend
from pasts.components.aggregated_model import AggregatedModel

warnings.filterwarnings("ignore")


def _make_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex a DataFrame with a complete daily DatetimeIndex, filling gaps with 0."""
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(full_idx, fill_value=0)


_MSG_FIT_ES = "  Fitting ExponentialSmoothing ..."
_MSG_FIT_RF = "  Fitting RandomForest ..."
_MSG_SCORES = "  Scores (unit-wise):"
_MSG_DONE = "  Done."

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join("data",
                        "favorita-grocery-sales-forecasting")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
ITEMS_PATH = os.path.join(DATA_DIR, "items.csv")
STORES_PATH = os.path.join(DATA_DIR, "stores.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "favorita_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 0. Load metadata
# ---------------------------------------------------------------------------
print("Loading metadata ...")
items = pd.read_csv(ITEMS_PATH)
stores = pd.read_csv(STORES_PATH)
print(f"  {len(items)} items, {len(stores)} stores")


# ---------------------------------------------------------------------------
# 1. Load train data (chunked to limit memory)
# ---------------------------------------------------------------------------
print("Loading train.csv (this may take a few minutes) ...")
t0 = time.time()

dtypes = {
    "id": "int32",
    "store_nbr": "int16",
    "item_nbr": "int32",
    "unit_sales": "float32",
    "onpromotion": "object",
}
train = pd.read_csv(TRAIN_PATH, dtype=dtypes, parse_dates=["date"])
# Clip negative sales to 0 (returns are encoded as negatives)
train["unit_sales"] = train["unit_sales"].clip(lower=0)

print(f"  Loaded {len(train):,} rows in {time.time() - t0:.0f}s")

# Merge metadata
train = train.merge(items[["item_nbr", "family", "perishable"]], on="item_nbr")
train = train.merge(stores[["store_nbr", "city", "state", "type"]], on="store_nbr")


# ===================================================================
# SCENARIO 1 — Store-level aggregation (54 multivariate series)
# ===================================================================
def run_scenario_store_level():
    print("\n" + "=" * 70)
    print("SCENARIO 1 : Store-level daily sales (54 series)")
    print("=" * 70)

    df_store = (
        train.groupby(["date", "store_nbr"])["unit_sales"]
        .sum()
        .unstack("store_nbr")
        .sort_index()
    )
    df_store.columns = [f"store_{c}" for c in df_store.columns]
    df_store = _make_daily_index(df_store)

    # Pick a subset of 6 diverse stores for tractable model fitting
    stores_sample = ["store_3", "store_25", "store_44", "store_47", "store_50", "store_53"]
    stores_sample = [s for s in stores_sample if s in df_store.columns]
    df = df_store[stores_sample].copy()

    signal = Signal(df, path=os.path.join(OUTPUT_DIR, "store_level"))


    # Statistical tests
    signal.stat.test_stationarity()


    # Validation split: last 2 months as test
    timestamp = str(df.index[-1] - pd.Timedelta(days=60))[:10]
    signal.validation_split(timestamp=timestamp)

    # Decomposition: remove trend
    signal.decompose()
    signal.residual -= LinearTrend().fit(signal.data)
    signal.data["store_53"].plot()
    

    # Models
    print(_MSG_FIT_ES)
    signal.apply_model(ExponentialSmoothing())
    print(_MSG_FIT_RF)
    signal.apply_model(RandomForest(lags=30))

    # Scores
    signal.compute_scores()
    print(_MSG_SCORES)
    for metric, df_score in signal.performance_models.get("unit_wise", {}).items():
        print(f"    {metric}:\n{df_score}")

    # Aggregated model
    signal.apply_model(AggregatedModel(
        {'ExponentialSmoothing': ExponentialSmoothing(), 'RandomForest': RandomForest(lags=30)},
        test_data=signal.test_data,
    ))
    signal.compute_scores(axis=1)
    signal.compute_conf_intervals(window_size=7)

    # Forecast
    signal.forecast("AggregatedModel", horizon=30)
    signal.compute_conf_intervals(window_size=7)
    plt.close(signal.plot.predictions())
    plt.close(signal.plot.forecast())
    print(_MSG_DONE)
    return signal


# ===================================================================
# SCENARIO 2 — Family × Store (mid-level hierarchy, ~1800 series)
# ===================================================================
def run_scenario_family_store():
    print("\n" + "=" * 70)
    print("SCENARIO 2 : Family x Store — single store, all families")
    print("=" * 70)

    # Pick one large store
    store_id = 44
    df_family = (
        train[train["store_nbr"] == store_id]
        .groupby(["date", "family"])["unit_sales"]
        .sum()
        .unstack("family")
        .sort_index()
    )
    df_family = _make_daily_index(df_family)

    print(f"  Store {store_id}: {df_family.shape[1]} product families")
    print(f"  Shape: {df_family.shape}")

    # Pick top-6 families by total volume for tractable run
    top_families = df_family.sum().nlargest(6).index.tolist()
    df = df_family[top_families].copy()
    print(f"  Selected families: {top_families}")

    signal = Signal(df, path=os.path.join(OUTPUT_DIR, "family_store"))

    timestamp = str(df.index[-1] - pd.Timedelta(days=60))[:10]
    print(f"  Validation split at {timestamp}")
    signal.validation_split(timestamp=timestamp)

    signal.decompose()
    signal.residual -= LinearTrend().fit(signal.data)

    print(_MSG_FIT_ES)
    signal.apply_model(ExponentialSmoothing())
    print(_MSG_FIT_RF)
    signal.apply_model(RandomForest(lags=14))

    signal.compute_scores()
    print(_MSG_SCORES)
    for metric, df_score in signal.performance_models.get("unit_wise", {}).items():
        print(f"    {metric}:\n{df_score}")

    signal.apply_model(AggregatedModel(
        {'ExponentialSmoothing': ExponentialSmoothing(), 'RandomForest': RandomForest(lags=14)},
        test_data=signal.test_data,
    ))
    signal.compute_scores(axis=1)
    signal.compute_conf_intervals(window_size=7)
    plt.close(signal.plot.predictions())

    signal.forecast("AggregatedModel", horizon=28)
    signal.compute_conf_intervals(window_size=7)
    plt.close(signal.plot.forecast())
    print(_MSG_DONE)
    return signal


# ===================================================================
# SCENARIO 3 — Curated individual items (diverse profiles)
# ===================================================================
def run_scenario_curated_items():
    print("\n" + "=" * 70)
    print("SCENARIO 3 : Curated item-level series (diverse profiles)")
    print("=" * 70)

    store_id = 44

    df_items = train[train["store_nbr"] == store_id].copy()
    df_items_pivot = (
        df_items.groupby(["date", "item_nbr"])["unit_sales"]
        .sum()
        .unstack("item_nbr")
        .sort_index()
    )
    df_items_pivot = _make_daily_index(df_items_pivot)

    total_sales = df_items_pivot.sum()
    n_days = len(df_items_pivot)

    # --- Profile 1: High-rotation item (top seller) ---
    high_vol_item = total_sales.idxmax()

    # --- Profile 2: Low-volume / intermittent item ---
    # Items with > 50% zero days and at least some sales
    zero_pct = (df_items_pivot == 0).sum() / n_days
    low_vol_candidates = total_sales[(zero_pct > 0.5) & (total_sales > 10)]
    low_vol_item = low_vol_candidates.idxmin() if len(low_vol_candidates) > 0 else total_sales.nsmallest(5).index[-1]

    # --- Profile 3: New product launch (appears late in dataset) ---
    first_sale = (df_items_pivot > 0).idxmax()  # first date with a sale per item
    mid_point = df_items_pivot.index[n_days // 2]
    late_starters = first_sale[first_sale > mid_point]
    # Among late starters, pick one with decent total sales
    if len(late_starters) > 0:
        launch_candidates = total_sales[late_starters.index]
        launch_item = launch_candidates.idxmax()
    else:
        # Fallback: item with latest first sale
        launch_item = first_sale.idxmax()

    selected = [high_vol_item, low_vol_item, launch_item]
    labels = {
        high_vol_item: f"high_vol_{high_vol_item}",
        low_vol_item: f"low_vol_{low_vol_item}",
        launch_item: f"launch_{launch_item}",
    }

    print(f"  High-volume item: {high_vol_item} "
          f"(total={total_sales[high_vol_item]:.0f}, "
          f"zero_pct={zero_pct[high_vol_item]:.1%})")
    print(f"  Low-volume item:  {low_vol_item} "
          f"(total={total_sales[low_vol_item]:.0f}, "
          f"zero_pct={zero_pct[low_vol_item]:.1%})")
    print(f"  Launch item:      {launch_item} "
          f"(first sale={first_sale[launch_item].date()}, "
          f"total={total_sales[launch_item]:.0f})")

    df = df_items_pivot[selected].rename(columns=labels).copy()

    # For the launch item, trim to start from its first sale
    launch_col = labels[launch_item]
    first_nonzero = df[launch_col][df[launch_col] > 0].index[0]
    df = df.loc[first_nonzero:]
    print(f"  Trimmed to {first_nonzero.date()} -> {df.index[-1].date()} ({len(df)} days)")

    signal = Signal(df, path=os.path.join(OUTPUT_DIR, "curated_items"))
    print(f"  Properties: {signal.properties['shape']}")
    plt.close(signal.plot())

    timestamp = str(df.index[-1] - pd.Timedelta(days=45))[:10]
    print(f"  Validation split at {timestamp}")
    signal.validation_split(timestamp=timestamp)

    signal.decompose()
    signal.residual -= LinearTrend().fit(signal.data)

    print(_MSG_FIT_ES)
    signal.apply_model(ExponentialSmoothing())
    print(_MSG_FIT_RF)
    signal.apply_model(RandomForest(lags=14))

    signal.compute_scores()
    print(_MSG_SCORES)
    for metric, df_score in signal.performance_models.get("unit_wise", {}).items():
        print(f"    {metric}:\n{df_score}")

    signal.apply_model(AggregatedModel(
        {'ExponentialSmoothing': ExponentialSmoothing(), 'RandomForest': RandomForest(lags=14)},
        test_data=signal.test_data,
    ))
    signal.compute_scores(axis=1)
    signal.compute_conf_intervals(window_size=7)
    plt.close(signal.plot.predictions())

    signal.forecast("AggregatedModel", horizon=28)
    signal.compute_conf_intervals(window_size=7)
    plt.close(signal.plot.forecast())
    print(_MSG_DONE)
    return signal


# ===================================================================
# SCENARIO 4 — Scale test: many items in parallel (wide DataFrame)
# ===================================================================
def run_scenario_scale_test(n_items=50):
    print("\n" + "=" * 70)
    print(f"SCENARIO 4 : Scale test — {n_items} random items, single store")
    print("=" * 70)

    store_id = 44
    df_items = train[train["store_nbr"] == store_id].copy()

    # Pick n_items with meaningful sales
    item_sales = df_items.groupby("item_nbr")["unit_sales"].sum()
    candidates = item_sales[item_sales > 100].index.tolist()
    rng = np.random.default_rng(42)
    chosen = rng.choice(candidates, size=min(n_items, len(candidates)), replace=False)

    df_pivot = (
        df_items[df_items["item_nbr"].isin(chosen)]
        .groupby(["date", "item_nbr"])["unit_sales"]
        .sum()
        .unstack("item_nbr")
        .sort_index()
    )
    df_pivot.columns = [f"item_{c}" for c in df_pivot.columns]
    df_pivot = _make_daily_index(df_pivot)

    print(f"  Shape: {df_pivot.shape}")

    signal = Signal(df_pivot, path=os.path.join(OUTPUT_DIR, "scale_test"))

    timestamp = str(df_pivot.index[-1] - pd.Timedelta(days=30))[:10]
    signal.validation_split(timestamp=timestamp)

    signal.decompose()
    signal.residual -= LinearTrend().fit(signal.data)

    print(_MSG_FIT_ES)
    t0 = time.time()
    signal.apply_model(ExponentialSmoothing())
    print(f"    -> {time.time() - t0:.1f}s")

    print(_MSG_FIT_RF)
    t0 = time.time()
    signal.apply_model(RandomForest(lags=14))
    print(f"    -> {time.time() - t0:.1f}s")

    signal.compute_scores()
    print("  Scores summary (RMSE, unit-wise):")
    uw = signal.performance_models.get("unit_wise", {})
    if "rmse" in uw:
        rmse_df = uw["rmse"]
        print(f"    ExponentialSmoothing: mean={rmse_df['ExponentialSmoothing'].mean():.2f}")
        print(f"    RandomForest:        mean={rmse_df['RandomForest'].mean():.2f}")

    signal.apply_model(AggregatedModel(
        {'ExponentialSmoothing': ExponentialSmoothing(), 'RandomForest': RandomForest(lags=14)},
        test_data=signal.test_data,
    ))
    signal.compute_scores(axis=1)

    signal.forecast("AggregatedModel", horizon=14)
    print(_MSG_DONE)
    return signal


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print(f"\nDataset location: {os.path.abspath(DATA_DIR)}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}\n")

    sig_store = run_scenario_store_level()
    sig_family = run_scenario_family_store()
    sig_items = run_scenario_curated_items()
    sig_scale = run_scenario_scale_test(n_items=50)

    print("\n" + "=" * 70)
    print("ALL SCENARIOS COMPLETED")
    print("=" * 70)
    print(f"Results saved in: {os.path.abspath(OUTPUT_DIR)}")