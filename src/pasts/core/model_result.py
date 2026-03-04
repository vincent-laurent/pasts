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

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ModelResult:
    """Container for all artifacts produced by fitting/predicting a model.

    Replaces the raw dict previously used in ``Signal.models``.
    Supports dict-like access (``result['predictions']``) for backward
    compatibility with Metrics, Visualization, persistence, and tests.
    For model-specific attributes (e.g. ``weights``, ``models`` on
    :class:`AggregatedModel`), ``__getitem__`` delegates to ``self.model``.

    Attributes
    ----------
    estimator_on_train : TimeSeriesModel
        The model fitted on **training data only**.
    predictions : pd.DataFrame
        Predictions covering the test period (model fitted on train).
    estimator_on_all : TimeSeriesModel or None
        The model refitted on **full data** (train + test).
        Populated by :meth:`Signal.refit`.
    forecast_data : pd.DataFrame or None
        Forecasts produced after :meth:`Signal.refit` + :meth:`Signal.forecast`.
    """

    estimator_on_train: object = None
    predictions: Optional[pd.DataFrame] = None
    best_parameters: object = "default"
    scores: dict = field(default_factory=lambda: {'unit_wise': {}, 'time_wise': {}})
    estimator_on_all: object = None
    forecast_data: Optional[pd.DataFrame] = None
    test_residuals: Optional[pd.DataFrame] = None
    test_confidence_interval: Optional[pd.DataFrame] = None
    forecast_confidence_interval: Optional[pd.DataFrame] = None
    historical_residuals: Optional[pd.DataFrame] = None
    fitted_values: Optional[pd.DataFrame] = None

    # Data references (set by Signal)
    _data: object = field(default=None, repr=False)
    _covariates: object = field(default=None, repr=False)

    # -- dict-like compat ------------------------------------------------

    _FIELD_ALIASES = {
        "forecast": "forecast_data",
        "model": "estimator_on_train",
        "final_estimator": "estimator_on_all",
    }

    def __getitem__(self, key):
        key = self._FIELD_ALIASES.get(key, key)
        val = getattr(self, key, None)
        if val is not None:
            return val
        # Delegate to the model (e.g. AggregatedModel.weights, .models)
        return getattr(self.estimator_on_train, key, None)

    def __setitem__(self, key, value):
        key = self._FIELD_ALIASES.get(key, key)
        setattr(self, key, value)

    def __contains__(self, key):
        key = self._FIELD_ALIASES.get(key, key)
        val = getattr(self, key, None)
        if val is not None:
            return True
        return getattr(self.estimator_on_train, key, None) is not None

    def __len__(self):
        """Count of non-None essential fields (backward compat with tests)."""
        return sum(
            1 for f in [
                'estimator_on_train', 'predictions', 'best_parameters', 'scores',
                'estimator_on_all', 'forecast_data',
            ]
            if getattr(self, f, None) is not None
        )

    def keys(self):
        """Iterate over non-None field names (dict compat)."""
        for f in self.__dataclass_fields__:
            if f.startswith('_'):
                continue
            if getattr(self, f, None) is not None:
                yield f
