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

import logging
from typing import TYPE_CHECKING, Union

import pandas as pd

if TYPE_CHECKING:
    from pasts.signal import Signal

logger = logging.getLogger(__name__)


class ValidationAccessor:
    """Accessor for train/test split operations on a :class:`~pasts.signal.Signal`.

    Accessed via ``signal.validation``.  Follows the same accessor pattern as
    :class:`~pasts.statistical_tests.StatAccessor` and
    :class:`~pasts.visualization.PlotAccessor`.

    Stores only the split timestamp (one scalar) — no DataFrame copies.
    ``Signal.train_data`` and ``Signal.test_data`` compute ``.loc`` slices
    on demand from the signal's own data and this timestamp.

    Decompositions share the parent signal's ``ValidationAccessor`` instance
    directly, so any split change is immediately visible to all decompositions.

    Examples
    --------
    >>> signal.validation.split("2020-01-01")
    >>> signal.train_data          # .loc slice computed on demand
    >>> signal.test_data           # .loc slice computed on demand
    >>> signal.validation.split("2020-01-01", n_splits_cv=5)
    >>> signal.validation.cv_tseries
    """

    def __init__(self, signal: "Signal"):
        self._signal     = signal
        self._timestamp  = None
        self._cv_tseries = None

    def split(self, timestamp: Union[int, str, pd.Timestamp],
              n_splits_cv: int = None) -> None:
        """Split the series between train (<= timestamp) and test (> timestamp).

        Parameters
        ----------
        timestamp :
            Index value that marks the train/test boundary.
        n_splits_cv : int, optional
            Number of folds for cross-validation on the training set.
        """
        train = self._signal.data.loc[self._signal.data.index <= timestamp]
        if len(train) < 2:
            raise ValueError("Train set is empty or too small.")
        self._timestamp = timestamp
        logger.info("Split applied on: %s", timestamp)
        if n_splits_cv is not None:
            from sklearn.model_selection import TimeSeriesSplit
            splits = list(TimeSeriesSplit(n_splits=n_splits_cv).split(train))
            self._cv_tseries = iter(splits)
        else:
            self._cv_tseries = None

    def reset(self) -> None:
        """Clear the split (called by :meth:`Signal.handle_nan`)."""
        self._timestamp  = None
        self._cv_tseries = None

    @property
    def cv_tseries(self):
        """Iterator over (train_indices, test_indices) folds, or ``None``."""
        return self._cv_tseries
