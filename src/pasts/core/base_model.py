# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from abc import ABC, abstractmethod

import pandas as pd


class TimeSeriesModel(ABC):
    """Abstract base class for time series decomposition models.

    A model extracts a signal component (trend, seasonality, …) from a time
    series and provides the inverse operation for recomposition.

    Two families exist:

    - Trend components (e.g. :class:`LinearTrend`, :class:`NonParametricTrend`)
      — extract and restore signal components (trend, etc.).
    - :class:`DartsModel` — component backed by a Darts ML/DL estimator;
      applicable to any :class:`~pasts.core.datacube.DataCube` or DataFrame,
      whether it is the raw signal or a residual.

    Subclasses must implement :meth:`fit` and :meth:`reverse_transform`.
    :meth:`transform` is provided as a default (negation of ``reverse_transform``).

    Attributes
    ----------
    nan_safe : bool
        Whether ``fit()`` can handle NaN values in the input data.
        Defaults to ``False``.  Subclasses that handle NaN internally
        (e.g. via interpolation) should set this to ``True``.
    """

    nan_safe: bool = False

    @property
    def name(self) -> str:
        """Model name used as key in ``Signal.models``."""
        return self.__class__.__name__

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "TimeSeriesModel":
        """Fit the model to the time series *X*."""

    @abstractmethod
    def reverse_transform(self, i: int) -> pd.DataFrame:
        """Return values to add back the component for recomposition.

        Parameters
        ----------
        i : int
            Length of output (negative for past, positive for future).
        """

    def transform(self, i: int) -> pd.DataFrame:
        """Return values to remove the component (negation of ``reverse_transform``).

        Parameters
        ----------
        i : int
            Length of output (negative for past, positive for future).
        """
        return -self.reverse_transform(i)
