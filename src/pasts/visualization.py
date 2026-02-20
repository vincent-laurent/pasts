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
from typing import TYPE_CHECKING

import matplotlib
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
    ts = model_entry[key]
    return pd.DataFrame(ts.values(), columns=ts.columns, index=ts.time_index)


class Visualization:
    """
    A class to visualize signals.

    Attributes
    ----------
    __signal : Signal

    Methods
    -------
    plot_signal():
        Plots raw data and transformed data if operations have been applied.
    plot_smoothing(resample_size, window_size):
        Plots resampled data.
    acf_plot():
        Plots autocorrelation (only for univariate series).
    show_predictions():
        Plots raw data and predicted values on same graph.
    show_forecast():
        Plots raw data and forecasted values (for future dates) on same graph.

    See also
    --------
    pasts.signal for details on the Signal object.
    pasts.components for details on time series components.
    pasts.model for details on predictions and forecast methods.
    """

    def __init__(self, signal: Signal):
        self.__signal = signal

    def plot_signal(self, display=True, **kwargs) -> None:
        """Plots raw data and transformed data if operations have been applied."""
        _, ax = plt.subplots()
        legend = []
        self.__signal.data.plot(ax=ax, **kwargs)
        for col in self.__signal.data.columns:
            legend.append(f'raw data: {col}')
        if self.__signal._residual is not None:
            self.__signal.rest_data.plot(ax=ax, **kwargs)
            for col in self.__signal.rest_data.columns:
                legend.append(f'residual: {col}')
            plt.title(f'Decomposition: {self.__signal.decomposition}',
                      fontdict={'fontsize': 10})
        plt.legend(legend)
        if display:
            plt.show()
        else:
            plt.close()

    def acf_plot(self) -> None:
        """Plots autocorrelation (only for univariate series)."""
        if not self.__signal.properties['is_univariate']:
            raise ValueError('Can only plot acf for univariate series')
        autocorrelation_plot(self.__signal.data)

    # --- Internal shared rendering ---

    def _show(self, data_key: str, ci_key: str, aggregated_only: bool,
              display: bool, prepend_last_obs: bool = False) -> None:
        """Shared matplotlib renderer for predictions and forecasts."""
        if not self.__signal.models:
            raise ValueError(_NO_PREDICTIONS_MSG)

        _, ax = plt.subplots()
        n_signals = self.__signal.test_data.shape[1]
        labels = [f'Actuals_s{i}' for i in range(1, n_signals + 1)]
        ax.plot(self.__signal.data, c='gray')

        if aggregated_only:
            if 'AggregatedModel' not in self.__signal.models:
                raise ValueError('No predictions have been computed with aggregated model')
            to_plot = ['AggregatedModel']
            for i, unit in enumerate(self.__signal.data.columns):
                itv = self.__signal.models['AggregatedModel'][ci_key]
                bounds = _confidence_bounds(itv[unit])
                ax.plot(bounds, color='green', linestyle='--')
                ax.fill_between(bounds.index, bounds['lower'], bounds['upper'],
                                color='green', alpha=0.3)
                labels += [f'lower_s{i + 1}', f'upper_s{i + 1}', f'interval_s{i + 1}']
        else:
            to_plot = list(self.__signal.models.keys())

        for model in to_plot:
            if data_key not in self.__signal.models[model]:
                warnings.warn(f'No {data_key} have been computed with {model}')
                continue
            pred = _ts_to_df(self.__signal.models[model], data_key)
            if prepend_last_obs:
                pred = pd.concat([self.__signal.data.iloc[-1:], pred])
            ax.plot(pred)
            labels += [f'{model}_s{i}' for i in range(1, n_signals + 1)]

        ax.legend(labels)
        plt.xlabel('time')
        plt.ylabel('values')
        if display:
            plt.show()
        else:
            plt.close()

    def _show_plotly(self, data_key: str, ci_key: str,
                     prepend_last_obs: bool = False) -> None:
        """Shared Plotly renderer for predictions and forecasts."""
        if not self.__signal.models:
            raise ValueError(_NO_PREDICTIONS_MSG)

        fig = go.Figure()
        for i, unit in enumerate(self.__signal.data.columns):
            fig.add_trace(go.Scatter(
                x=self.__signal.data.index, y=self.__signal.data[unit],
                mode='lines', name=f'Actual_s{i + 1}', line={'color': '#7f7f7f'}
            ))

        j = 0
        last_obs = self.__signal.data.iloc[-1:]
        for model_name, model_data in self.__signal.models.items():
            if data_key not in model_data:
                warnings.warn(f'No {data_key} have been computed with {model_name}')
                continue

            pred = _ts_to_df(model_data, data_key)
            if prepend_last_obs:
                pred = pd.concat([last_obs, pred])

            for i, unit in enumerate(pred.columns):
                trace_color = _pick_color(j)
                fig.add_trace(go.Scatter(
                    x=pred.index, y=pred[unit], mode='lines',
                    name=f'{model_name}_s{i + 1}', line={'color': trace_color}
                ))

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
                j += 1

        title = 'Forecasts with Confidence Intervals' if prepend_last_obs else 'Predictions with Confidence Intervals'
        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Values', showlegend=True)
        fig.show()

    # --- Public methods ---

    def show_predictions(self, aggregated_only=False, display=True) -> None:
        """Plots raw data and predicted values on same graph."""
        self._show('predictions', 'test_confidence_interval', aggregated_only, display)

    def show_predictions_plotly(self) -> None:
        """Plots raw data and predicted values on the same Plotly graph with confidence intervals."""
        self._show_plotly('predictions', 'test_confidence_interval')

    def show_forecast(self, aggregated_only=False, display=True) -> None:
        """Plots raw data and forecasted values (for future dates) on same graph."""
        self._show('forecast', 'forecast_confidence_interval', aggregated_only, display,
                   prepend_last_obs=True)

    def show_forecast_plotly(self) -> None:
        """Plots raw data and predicted future values on the same Plotly graph with confidence intervals."""
        self._show_plotly('forecast', 'forecast_confidence_interval', prepend_last_obs=True)
