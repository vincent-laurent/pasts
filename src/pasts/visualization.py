# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING, Union

import matplotlib
import matplotlib.figure
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
import plotly.graph_objects as go

if TYPE_CHECKING:
    from pasts.signal import Signal

_NO_PREDICTIONS_MSG = 'No predictions have been computed.'

colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def _pick_color(j: int) -> str:
    if j < len(colors):
        return colors[j]
    return random.choice(list(matplotlib.colors.cnames.values()))


def _confidence_bounds(itv_series) -> pd.DataFrame:
    bounds = pd.DataFrame(index=itv_series.index)
    bounds['lower'] = [v[0] for v in itv_series.values]
    bounds['upper'] = [v[1] for v in itv_series.values]
    return bounds


def _ts_to_df(model_entry: dict, key: str) -> pd.DataFrame:
    return model_entry[key]


def _get_fitted(model_result, train_data: pd.DataFrame) -> pd.DataFrame:
    """Return fitted values (in-sample predictions), computing and caching if needed.

    For models with a decomposition, returns the trend component
    (``train_data - residual``).  For plain models, returns the full
    in-sample reconstruction via ``compute_historical_residuals``.
    """
    if model_result.fitted_values is not None:
        return model_result.fitted_values
    decomp = getattr(model_result.estimator_on_train, '_decomposition', None)
    if decomp is not None and getattr(decomp, '_residual', None) is not None:
        residual = decomp._residual
        if len(residual) == len(train_data):
            residual = residual.set_axis(train_data.index)
        fitted = train_data - residual
    else:
        residuals = model_result.estimator_on_train.compute_historical_residuals(train_data)
        fitted = train_data.loc[residuals.index] - residuals
    model_result.fitted_values = fitted
    return fitted


class PlotAccessor:
    """Accessor for Signal.plot that provides a fluent plotting API.

    All methods return the figure object (matplotlib or plotly).

    Usage
    -----
        signal.plot()                              # raw signal
        signal.plot.acf()                          # autocorrelation
        signal.plot.predictions()                  # predictions (matplotlib)
        signal.plot.predictions(backend="plotly")  # predictions (plotly)
        signal.plot.forecast()                     # forecast (matplotlib)
        signal.plot.forecast(backend="plotly")     # forecast (plotly)
    """

    def __init__(self, signal: Signal):
        self._signal = signal

    def __call__(self, legend: bool = True, **kwargs) -> matplotlib.figure.Figure:
        """Plot raw signal data.

        Parameters
        ----------
        legend : bool
            Whether to show the legend (default ``True``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots()
        legend_labels = []
        self._signal.data.plot(ax=ax, **kwargs)
        for col in self._signal.data.columns:
            legend_labels.append(f'raw data: {col}')
        if legend:
            ax.legend(legend_labels)
        return fig

    def acf(self) -> matplotlib.figure.Figure:
        """Plot autocorrelation (only for univariate series).

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self._signal.properties['is_univariate']:
            raise ValueError('Can only plot acf for univariate series')
        fig, ax = plt.subplots()
        autocorrelation_plot(self._signal.data, ax=ax)
        return fig

    def predictions(self, aggregated_only: bool = False,
                    backend: str = "matplotlib",
                    show_fitted: bool = False,
                    legend: bool = True,
                    ) -> Union[matplotlib.figure.Figure, go.Figure]:
        """Plot raw data and predicted values on same graph.

        Parameters
        ----------
        aggregated_only : bool
            If True, only plot the aggregated model predictions.
        backend : str
            "matplotlib" (default) or "plotly".
        show_fitted : bool
            If True, also plot in-sample fitted values on the training period.
        legend : bool
            Whether to show the legend (default ``True``).

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
        """
        if backend == "plotly":
            return self._show_plotly(
                'predictions', 'test_confidence_interval', aggregated_only,
                show_fitted=show_fitted, legend=legend)
        return self._show(
            'predictions', 'test_confidence_interval', aggregated_only,
            show_fitted=show_fitted, legend=legend)

    def forecast(self, aggregated_only: bool = False,
                 backend: str = "matplotlib",
                 legend: bool = True,
                 ) -> Union[matplotlib.figure.Figure, go.Figure]:
        """Plot raw data and forecasted values (for future dates) on same graph.

        Parameters
        ----------
        aggregated_only : bool
            If True, only plot the aggregated model forecasts.
        backend : str
            "matplotlib" (default) or "plotly".
        legend : bool
            Whether to show the legend (default ``True``).

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
        """
        if backend == "plotly":
            return self._show_plotly(
                'forecast', 'forecast_confidence_interval',
                aggregated_only, prepend_last_obs=True, legend=legend)
        return self._show(
            'forecast', 'forecast_confidence_interval',
            aggregated_only, prepend_last_obs=True, legend=legend)

    def residuals(self, backend: str = "matplotlib",
                  legend: bool = True,
                  ) -> Union[matplotlib.figure.Figure, go.Figure]:
        """Plot decomposition residuals (detrended signal) for each model.

        Only models with an attached decomposition are shown.

        Parameters
        ----------
        backend : str
            "matplotlib" (default) or "plotly".
        legend : bool
            Whether to show the legend (default ``True``).

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
        """
        if not self._signal.models:
            raise ValueError(_NO_PREDICTIONS_MSG)

        items = []
        for j, (name, result) in enumerate(self._signal.models.items()):
            decomp = getattr(result.estimator_on_train, '_decomposition', None)
            if decomp is None or getattr(decomp, '_residual', None) is None:
                continue
            residual = decomp._residual
            if len(residual) == len(self._signal.train_data):
                residual = residual.set_axis(self._signal.train_data.index)
            items.append((j, name, residual))

        if not items:
            raise ValueError('No models with decomposition residuals found.')

        if backend == "plotly":
            fig = go.Figure()
            for j, name, residual in items:
                color = _pick_color(j)
                for i, unit in enumerate(residual.columns):
                    fig.add_trace(go.Scatter(
                        x=residual.index, y=residual[unit], mode='lines',
                        name=f'{name}_s{i + 1}', line={'color': color},
                    ))
            fig.update_layout(title='Residuals (detrended signal)',
                              xaxis_title='Time', yaxis_title='Residual',
                              showlegend=legend)
            return fig

        fig, ax = plt.subplots()
        labels = []
        n_signals = self._signal.train_data.shape[1]
        for j, name, residual in items:
            residual.plot(ax=ax, color=_pick_color(j), legend=False)
            labels += [f'{name}_s{i}' for i in range(1, n_signals + 1)]
        if legend:
            ax.legend(labels)
        ax.set_xlabel('time')
        ax.set_ylabel('residual')
        return fig

    # --- Internal shared rendering ---

    def _show(self, data_key: str, ci_key: str, aggregated_only: bool,
              prepend_last_obs: bool = False,
              show_fitted: bool = False,
              legend: bool = True) -> matplotlib.figure.Figure:
        if not self._signal.models:
            raise ValueError(_NO_PREDICTIONS_MSG)

        fig, ax = plt.subplots()
        n_signals = self._signal.test_data.shape[1]
        labels = [f'Actuals_s{i}' for i in range(1, n_signals + 1)]
        self._signal.data.plot(ax=ax, color='gray', legend=False)

        if aggregated_only:
            if 'AggregatedModel' not in self._signal.models:
                raise ValueError('No predictions have been computed with aggregated model')
            to_plot = ['AggregatedModel']
            for i, unit in enumerate(self._signal.data.columns):
                itv = self._signal.models['AggregatedModel'][ci_key]
                bounds = _confidence_bounds(itv[unit])
                bounds.plot(ax=ax, color='green', linestyle='--', legend=False)
                ax.fill_between(bounds.index, bounds['lower'], bounds['upper'],
                                color='green', alpha=0.3)
                labels += [f'lower_s{i + 1}', f'upper_s{i + 1}', f'interval_s{i + 1}']
        else:
            to_plot = list(self._signal.models.keys())

        for j, model in enumerate(to_plot):
            if data_key not in self._signal.models[model]:
                warnings.warn(f'No {data_key} have been computed with {model}')
                continue
            model_result = self._signal.models[model]
            pred = _ts_to_df(model_result, data_key)
            if prepend_last_obs:
                if model_result.estimator_on_all is not None:
                    anchor = self._signal.data.iloc[-1:]
                else:
                    anchor = self._signal.train_data.iloc[-1:]
                pred = pd.concat([anchor, pred])
            color = _pick_color(j)
            pred.plot(ax=ax, color=color, legend=False)
            labels += [f'{model}_s{i}' for i in range(1, n_signals + 1)]

            if show_fitted:
                fitted = _get_fitted(self._signal.models[model], self._signal.train_data)
                fitted.plot(ax=ax, color=color, linestyle='--', alpha=0.7, legend=False)
                labels += [f'{model}_fitted_s{i}' for i in range(1, n_signals + 1)]

        if legend:
            ax.legend(labels)
        ax.set_xlabel('time')
        ax.set_ylabel('values')
        return fig

    def _show_plotly(self, data_key: str, ci_key: str,
                     aggregated_only: bool = False,
                     prepend_last_obs: bool = False,
                     show_fitted: bool = False,
                     legend: bool = True) -> go.Figure:
        if not self._signal.models:
            raise ValueError(_NO_PREDICTIONS_MSG)

        fig = go.Figure()
        for i, unit in enumerate(self._signal.data.columns):
            fig.add_trace(go.Scatter(
                x=self._signal.data.index, y=self._signal.data[unit],
                mode='lines', name=f'Actual_s{i + 1}', line={'color': '#7f7f7f'}
            ))

        if aggregated_only:
            if 'AggregatedModel' not in self._signal.models:
                raise ValueError('No predictions have been computed with aggregated model')
            models_to_plot = {'AggregatedModel': self._signal.models['AggregatedModel']}
        else:
            models_to_plot = self._signal.models

        j = 0
        for model_name, model_data in models_to_plot.items():
            if data_key not in model_data:
                warnings.warn(f'No {data_key} have been computed with {model_name}')
                continue

            pred = _ts_to_df(model_data, data_key)
            if prepend_last_obs:
                if model_data.estimator_on_all is not None:
                    anchor = self._signal.data.iloc[-1:]
                else:
                    anchor = self._signal.train_data.iloc[-1:]
                pred = pd.concat([anchor, pred])

            for i, unit in enumerate(pred.columns):
                trace_color = _pick_color(j)
                fig.add_trace(go.Scatter(
                    x=pred.index, y=pred[unit], mode='lines',
                    name=f'{model_name}_s{i + 1}', line={'color': trace_color}
                ))

                if ci_key in model_data and unit in model_data[ci_key].columns:
                    itv = model_data[ci_key][unit]
                    bounds = _confidence_bounds(itv)
                    ci_index = itv.index

                    fig.add_trace(go.Scatter(
                        x=ci_index, y=bounds['lower'], mode='lines',
                        line={'color': trace_color, 'dash': 'dash'}, showlegend=False,
                        legendgroup=f'CI_{model_name}_s{i + 1}',
                        name=f'CI_{model_name}_s{i + 1}'
                    ))
                    rgb = tuple(int(trace_color.lstrip('#')[k:k + 2], 16) for k in (0, 2, 4))
                    fig.add_trace(go.Scatter(
                        x=ci_index, y=bounds['upper'], mode='lines',
                        line={'color': trace_color, 'dash': 'dash'}, fill='tonexty',
                        fillcolor=f'rgba{rgb + (0.3,)}',
                        legendgroup=f'CI_{model_name}_s{i + 1}',
                        name=f'CI_{model_name}_s{i + 1}'
                    ))

                if show_fitted:
                    fitted = _get_fitted(model_data, self._signal.train_data)
                    if unit in fitted.columns:
                        fig.add_trace(go.Scatter(
                            x=fitted.index, y=fitted[unit], mode='lines',
                            name=f'{model_name}_fitted_s{i + 1}',
                            line={'color': trace_color, 'dash': 'dash'},
                            opacity=0.7,
                        ))
                j += 1

        title = 'Forecasts with Confidence Intervals' if prepend_last_obs else 'Predictions with Confidence Intervals'
        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Values', showlegend=legend)
        return fig
