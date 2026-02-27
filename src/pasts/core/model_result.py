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
import warnings
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

    Call :meth:`forecast` to lazily refit the model on the full dataset
    and produce a forecast::

        result.forecast(horizon)           # refit + forecast
        result.forecast_data               # stored DataFrame
    """

    model: object
    predictions: Optional[pd.DataFrame] = None
    best_parameters: object = "default"
    scores: dict = field(default_factory=lambda: {'unit_wise': {}, 'time_wise': {}})
    final_estimator: object = None
    forecast_data: Optional[pd.DataFrame] = None
    test_residuals: Optional[pd.DataFrame] = None
    test_confidence_interval: Optional[pd.DataFrame] = None
    forecast_confidence_interval: Optional[pd.DataFrame] = None
    historical_residuals: Optional[pd.DataFrame] = None

    # Data references for lazy refit (set by Signal)
    _data: object = field(default=None, repr=False)
    _covariates: object = field(default=None, repr=False)

    # -- forecast ----------------------------------------------------------

    def _ensure_final_estimator(self) -> None:
        """Lazily refit the model on the full dataset."""
        if self.final_estimator is None:
            warnings.warn(f"Fitting model {self.model.name} on whole dataset...")
            self.final_estimator = copy.deepcopy(self.model)
            self.final_estimator.fit(self._data, covariates=self._covariates)

    def forecast(self, horizon: int) -> pd.DataFrame:
        """Lazily refit the model on full data and forecast.

        On the first call, deep-copies the model, refits it on
        ``_data`` (the full signal), and stores it as
        ``final_estimator``.  Subsequent calls with a different
        horizon reuse the already-fitted estimator.

        Parameters
        ----------
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        pd.DataFrame
        """
        self._ensure_final_estimator()
        self.forecast_data = self.final_estimator.forecast(horizon)
        return self.forecast_data

    # -- dict-like compat ------------------------------------------------

    _FIELD_ALIASES = {"forecast": "forecast_data"}

    def __getitem__(self, key):
        key = self._FIELD_ALIASES.get(key, key)
        val = getattr(self, key, None)
        if val is not None:
            return val
        # Delegate to the model (e.g. AggregatedModel.weights, .models)
        return getattr(self.model, key, None)

    def __setitem__(self, key, value):
        key = self._FIELD_ALIASES.get(key, key)
        setattr(self, key, value)

    def __contains__(self, key):
        key = self._FIELD_ALIASES.get(key, key)
        val = getattr(self, key, None)
        if val is not None:
            return True
        return getattr(self.model, key, None) is not None

    def __len__(self):
        """Count of non-None essential fields (backward compat with tests)."""
        return sum(
            1 for f in [
                'model', 'predictions', 'best_parameters', 'scores',
                'final_estimator', 'forecast_data',
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
