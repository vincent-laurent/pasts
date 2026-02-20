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

import pandas as pd
from pandas import MultiIndex
from sklearn.metrics import root_mean_squared_error

from pasts.core.base_model import TimeSeriesModel


def _weighted_aggregate(
    model_predictions: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-unit RMSE-weighted average of predictions across models."""
    first_key = next(iter(model_predictions))
    first_df = model_predictions[first_key]
    df_ag = pd.DataFrame(0.0, index=first_df.index, columns=first_df.columns)
    for ref in df_ag.columns:
        df_ag[ref] = sum(
            model_predictions[m][ref].values * weights.loc[ref, m]
            for m in model_predictions
        )
    return df_ag


class AggregatedModel(TimeSeriesModel):
    """Weighted average of multiple TimeSeriesModel sub-models.

    Parameters
    ----------
    models : dict[str, TimeSeriesModel]
        Named sub-models to aggregate.
    weights : pd.DataFrame, optional
        Pre-computed weights (index=units, columns=model names).
        If None, must be set via :meth:`compute_weights` before
        calling :meth:`reverse_transform`.
    """

    def __init__(self, models: dict[str, TimeSeriesModel],
                 weights: pd.DataFrame = None):
        self._models = models
        self._weights = weights

    @property
    def weights(self) -> pd.DataFrame:
        return self._weights

    @weights.setter
    def weights(self, value: pd.DataFrame):
        self._weights = value

    def fit(self, X) -> "AggregatedModel":
        """Fit all sub-models on *X*.

        Parameters
        ----------
        X : pd.DataFrame or DataCube
            Training data. Passed to each sub-model's ``fit()``.
        """
        for model in self._models.values():
            model.fit(X)
        return self

    def reverse_transform(self, i: int) -> pd.DataFrame:
        """Return weighted average of sub-models' ``reverse_transform(i)``."""
        predictions = {
            name: model.reverse_transform(i)
            for name, model in self._models.items()
        }
        return _weighted_aggregate(predictions, self._weights)

    @staticmethod
    def compute_weights(
        predictions: dict[str, pd.DataFrame],
        test_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute per-unit RMSE-based weights from predictions vs test data.

        Parameters
        ----------
        predictions : dict[str, pd.DataFrame]
            Predicted values per model name.
        test_data : pd.DataFrame
            Actual test values.

        Returns
        -------
        pd.DataFrame
            Weights with units as index and model names as columns.
        """
        model_names = list(predictions.keys())
        weights = pd.DataFrame(
            index=MultiIndex.from_product(
                [test_data.index, test_data.columns],
                names=['Date', 'Unité']
            ),
            columns=model_names
        )
        weights.drop(test_data.index[0], level=0, inplace=True)

        for model_name in model_names:
            df_pred = predictions[model_name].copy()
            for date in weights.index.get_level_values(0).unique():
                df_pred_temp = df_pred[df_pred.index < date]
                df_test_temp = test_data[test_data.index < date]
                for ref in weights.index.get_level_values(1).unique():
                    weights.loc[(date, ref), model_name] = (
                        1 / root_mean_squared_error(
                            df_test_temp[ref], df_pred_temp[ref]
                        )
                    )

        for i in weights.index:
            weights.loc[i] = weights.loc[i] / weights.loc[i].sum()
        weights = weights.groupby('Unité')[model_names].mean()
        return weights
