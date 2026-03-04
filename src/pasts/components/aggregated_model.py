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

import copy

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

    Works like any other :class:`TimeSeriesModel`: pass it to
    ``Signal.apply_model()`` and the standard ``fit`` / ``reverse_transform``
    cycle handles everything.

    Weights are computed automatically during :meth:`fit` via an internal
    hold-out split on the training data (no test data is ever seen by the
    model).

    Parameters
    ----------
    models : dict[str, TimeSeriesModel]
        Named sub-models to aggregate.
    weights : pd.DataFrame, optional
        Pre-computed weights (index=units, columns=model names).
        If *None*, weights are computed automatically during :meth:`fit`
        using an internal validation split.
    val_ratio : float, optional
        Fraction of training data to hold out for internal weight
        computation (default ``0.2``).  Ignored when *weights* is provided.
    min_train_size : int, optional
        Minimum number of rows required for the internal training split
        (default ``10``).  Raises ``ValueError`` if the dataset is too
        small after the hold-out.
    decomposition : :class:`~pasts.core.decomposition.Decomposition`, optional
        When provided, :meth:`fit` replays the decomposition on the
        training data first and fits the sub-models on the residual.
        :meth:`forecast` then recomposes predictions back to the
        original signal space.
    """

    def __init__(self, models: dict[str, TimeSeriesModel],
                 weights: pd.DataFrame = None,
                 val_ratio: float = 0.2,
                 min_train_size: int = 10,
                 decomposition=None):
        from pasts.components.darts_model import DartsModel
        self._models = {
            name: model if isinstance(model, TimeSeriesModel) else DartsModel(model)
            for name, model in models.items()
        }
        self._weights = weights
        self._val_ratio = val_ratio
        self._min_train_size = min_train_size
        self._decomposition = decomposition

    @property
    def nan_safe(self) -> bool:
        """An aggregated model is nan_safe only if all sub-models are."""
        return all(m.nan_safe for m in self._models.values())

    @property
    def models(self) -> dict:
        """Sub-models dictionary."""
        return self._models

    @property
    def weights(self) -> pd.DataFrame:
        return self._weights

    @weights.setter
    def weights(self, value: pd.DataFrame):
        self._weights = value

    def fit(self, X, covariates=None) -> "AggregatedModel":
        """Fit all sub-models on *X*.

        When weights are not pre-computed, an internal hold-out split is
        used to estimate RMSE-based weights.  Sub-models are then refitted
        on the full training data *X*.

        Parameters
        ----------
        X : pd.DataFrame or DataCube
            Training data.  Passed to each sub-model's ``fit()``.
        covariates : :class:`~pasts.covariates.Covariates`, optional
            Covariates forwarded to each sub-model's ``fit()``.
        """
        from pasts.core.datacube import DataCube
        if isinstance(X, DataCube):
            X = X.data

        if self._decomposition is not None:
            self._decomposition.fit(X, covariates=covariates)
            X = self._decomposition.residual

        if self._weights is not None:
            for model in self._models.values():
                model.fit(X, covariates=covariates)
            return self

        if len(self._models) == 1:
            for model in self._models.values():
                model.fit(X, covariates=covariates)
            name = next(iter(self._models))
            self._weights = pd.DataFrame({name: 1.0}, index=X.columns)
            return self

        # Internal hold-out for weight computation
        n_val = max(2, int(len(X) * self._val_ratio))
        n_train = len(X) - n_val
        if n_train < self._min_train_size:
            raise ValueError(
                f"Training data too small for internal validation: "
                f"{len(X)} rows with val_ratio={self._val_ratio} "
                f"leaves only {n_train} training rows "
                f"(minimum: {self._min_train_size})."
            )
        train_int = X.iloc[:n_train]
        val_int = X.iloc[n_train:]

        models_tmp = {
            name: copy.deepcopy(m) for name, m in self._models.items()
        }
        # Darts accepts covariates longer than the training series
        for m in models_tmp.values():
            m.fit(train_int, covariates=covariates)
        predictions = {
            name: m.forecast(len(val_int))
            for name, m in models_tmp.items()
        }
        for name in predictions:
            if len(predictions[name]) == len(val_int):
                predictions[name].index = val_int.index
        self._weights = self.compute_weights(predictions, val_int)

        # Refit all sub-models on the full training data
        for model in self._models.values():
            model.fit(X, covariates=covariates)
        return self

    def reverse_transform(self, i: int) -> pd.DataFrame:
        """Return weighted average of sub-models' ``reverse_transform(i)``."""
        predictions = {
            name: model.reverse_transform(i)
            for name, model in self._models.items()
        }
        return _weighted_aggregate(predictions, self._weights)

    def forecast(self, horizon: int) -> pd.DataFrame:
        """Forecast the next *horizon* steps.

        Always delegates to each sub-model's ``forecast()``, so that each
        sub-model handles its own decomposition recomposition.  If this
        aggregated model also has its own decomposition, it is applied on
        top of the aggregated result.

        Parameters
        ----------
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        pd.DataFrame
        """
        predictions = {
            name: model.forecast(horizon)
            for name, model in self._models.items()
        }
        result = _weighted_aggregate(predictions, self._weights)
        if self._decomposition is not None:
            return self._decomposition.recompose(result, horizon)
        return result

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
                    rmse = root_mean_squared_error(
                        df_test_temp[ref], df_pred_temp[ref]
                    )
                    weights.loc[(date, ref), model_name] = 1 / (rmse + 1e-10)

        for i in weights.index:
            weights.loc[i] = weights.loc[i] / weights.loc[i].sum()
        weights = weights.groupby('Unité')[model_names].mean()
        return weights
