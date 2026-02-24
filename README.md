# Python AnalySis for Time Series

[![pytest](https://github.com/eurobios-mews-labs/pasts/actions/workflows/pytest.yml/badge.svg?event=push)](https://docs.pytest.org)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/eurobios-mews-labs/pasts/graphs/commit-activity)


This package aims to structure the way time series analysis and forecasting is done.

#### Purpose of the Package
+ Provide a unified interface for time series forecasting and analysis.
+ All models — parametric (e.g. linear trend), ML/DL (via [Darts](https://unit8co.github.io/darts/)), and ensembles — share the same `TimeSeriesModel` interface.

#### Features
+ Collection of analysis methods:
  - Statistical testing (stationarity, seasonality, Granger causality)
  - Decomposition (trend removal, log transforms, custom operations)
  - Visualization via `signal.plot` accessor (matplotlib and Plotly)
+ Pluggable model components:
  - `LinearTrend` — linear parametric model
  - Non-parametric trend methods: `MovingAverageTrend`, `HPFilterTrend`, `STLTrend`, `EMDTrend`, `HighPassFilterTrend`
  - `Differencing` — transform-based detrending via `DataCube.apply()`
  - `DartsModel` — wraps any Darts estimator (ARIMA, ExponentialSmoothing, Prophet, XGBoost, etc.)
  - `AggregatedModel` — RMSE-weighted ensemble of models

#### Installation
The package can be installed by:
```bash
python3 -m pip install git+https://github.com/eurobios-mews-labs/pasts
```

#### Building the documentation

First, make sure you have sphinx and the Readthedocs theme installed.

```shell script
pip install sphinx sphinx_rtd_theme
```

Then build the doc with:
```shell script
cd doc
make html
```

The documentation can then be accessed from `doc/_build/html/index.html`.


## Architecture

The library is orchestrated through the `Signal` class, which composes all subsystems:

```
Signal
├── TimeSeriesModel (ABC)          — unified fit / reverse_transform interface
│   ├── LinearTrend                — linear parametric model
│   ├── NonParametricTrend         — base for non-parametric trend methods
│   │   ├── MovingAverageTrend, HPFilterTrend, STLTrend, EMDTrend, HighPassFilterTrend
│   ├── DartsModel                 — wraps any Darts ML/DL estimator
│   └── AggregatedModel            — RMSE-weighted ensemble
├── Decomposition                  — records and reverses signal transformations
├── Validation                     — train/test splits
├── Metrics                        — R², RMSE, MAPE, SMAPE, MAE
└── PlotAccessor (.plot)           — matplotlib / Plotly plots
```

## Usage and example

You can find full examples in `examples/ex_model.py`.

### Start project
Import your data as a pandas DataFrame with a temporal index and use the `Signal` class.
```python
import pandas as pd

from darts.datasets import AirPassengersDataset
from darts.models import AutoARIMA, ExponentialSmoothing

from pasts.signal import Signal

series = AirPassengersDataset().load()
dt = pd.DataFrame(series.values())
dt.rename(columns={0: 'passengers'}, inplace=True)
dt.index = series.time_index
signal = Signal(dt)
```

### Visualize and analyze data
The `properties` attribute contains some information about the data.
Use the `.plot` accessor to generate various types of plots.
```python
print(signal.properties)
```
Output:
```python
>>> {'shape': (144, 1), 'types': passengers    float64
dtype: object,
'is_univariate': True,
'nanSum': passengers   0
dtype: int64,
'quantiles':   0.00   0.05   0.50    0.95    0.99   1.00
passengers     104.0  121.6  265.5  488.15  585.79  622.0}
```
```python
signal.plot()
signal.plot.acf()
```
Yield:

<img src="examples/ex_plot1.png" alt="drawing" width="700"/>
<img src="examples/ex_acf.png" alt="drawing" width="700"/>

You can also perform statistical tests specific to time series.
```python
signal.stat.test_stationarity()
signal.stat.test_stationarity(method='kpss')
signal.stat.test_seasonality()
print(signal.tests_stat)
```

### Machine Learning

Choose a date to split the series between train and test.
```python
timestamp = '1958-12-01'
signal.validation_split(timestamp=timestamp)
```

#### Decomposition

The library provides components to decompose the signal before fitting forecasting models.
Use `decompose()` to initialize the residual, then subtract components (e.g. trend).
Forecasting models are then fitted on this residual, and the decomposition is automatically
reversed when generating predictions.

```python
from pasts.components import LinearTrend

signal.decompose()
signal.residual -= LinearTrend().fit(signal.data)
signal.plot()
```
<img src="examples/ex_op.png" alt="drawing" width="700"/>

#### Fitting a model on the residual

Use `learn()` on the residual to train a forecasting model on the de-trended series.
Then call `forecast(horizon=N)` on the parent signal — the decomposition is automatically
reversed (trend added back).

```python
signal.residual.learn(ExponentialSmoothing())
signal.forecast(horizon=100)
signal.plot.forecast()
```

#### Applying models (with train/test evaluation)

Alternatively, use `apply_model` for a train/test evaluation workflow.
It accepts any `TimeSeriesModel` (e.g. `DartsModel`, `LinearTrend`) or a raw Darts model
(auto-wrapped in `DartsModel`). Predictions are automatically recomposed with the
decomposition (trend added back).

```python
signal.apply_model(ExponentialSmoothing())
signal.apply_model(AutoARIMA())
```

Gridsearch is also supported:
```python
from darts.utils.utils import ModelMode, SeasonalityMode
param_grid = {
    'trend': [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
    'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE],
}
signal.apply_model(ExponentialSmoothing(), gridsearch=True, parameters=param_grid)
```

You can also use the explicit `DartsModel` wrapper:
```python
from pasts.components import DartsModel

signal.apply_model(DartsModel(ExponentialSmoothing()))
```

#### Scores

Pass a list of metrics to `compute_scores`. By default, it computes R², MSE, RMSE, MAPE, SMAPE and MAE.
Choose `axis=1` (default) for unit-wise scores or `axis=0` for time-wise scores.
```python
signal.compute_scores(list_metrics=['rmse', 'r2'])
signal.compute_scores()
print(signal.performance_models['unit_wise']['rmse'])
```
Output:
```python
>>>            ExponentialSmoothing  AutoARIMA
passengers            40.306771   26.718103
```

Visualize predictions:
```python
signal.plot.predictions()
```
<img src="examples/ex_pred.png" alt="drawing" width="700"/>

#### Forecast

Generate predictions for future dates using `forecast`:
```python
signal.forecast("AutoARIMA", 100)
signal.forecast("ExponentialSmoothing", 100)
signal.plot.forecast()
```
<img src="examples/ex_fc.png" alt="drawing" width="700"/>

#### Aggregation of models

`AggregatedModel` combines models weighted by their RMSE on test data.
The more a model performs compared to others, the greater its weight.
Pass it to `apply_model` like any other model.

```python
from pasts.components.aggregated_model import AggregatedModel

signal.apply_model(AggregatedModel(
    {'AutoARIMA': AutoARIMA(), 'ExponentialSmoothing': ExponentialSmoothing()},
    test_data=signal.test_data,
))
signal.compute_scores(axis=1)
signal.plot.predictions()
signal.forecast("AggregatedModel", 100)
```
<img src="examples/ex_fc_ag.png" alt="drawing" width="700"/>

### Author
<img src="logoEurobiosMewsLabs.png" alt="drawing" width="400"/>
