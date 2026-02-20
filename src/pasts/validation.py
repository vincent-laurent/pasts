# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import logging
from typing import Union

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class Validation:
    """
    A class to split data between train and test sets (possibly cross-validation).

    Attributes
    ----------
    __data : pd.Dataframe
        Dataset to split
    train_data : pd.Dataframe (default None)
        Train set as a dataframe
    test_data : pd.Dataframe(default None)
        Test set as dataframe

    Methods
    -------
    split_cv(timestamp, n_splits_cv) :
        Splits data between train set (<= timestamp) and test set (> timestamp).
    """

    def __init__(self, data: pd.DataFrame):
        self.__cv_tseries = None
        self.__data = data
        self.train_data = None
        self.test_data = None

    @property
    def cv_tseries(self):
        """Cross-validation indexes if requested (default None)"""
        return self.__cv_tseries

    def split_cv(self, timestamp: Union[int, str, pd.Timestamp], n_splits_cv=None) -> None:
        """
        Splits data between train set (<= timestamp) and test set (> timestamp).

        Parameters
        ----------
        timestamp : Union[int, pd.Timestamp, str]
            Split index
        n_splits_cv : int (default None)
            Number of folds for cross-validation
        """
        self.train_data = self.__data.loc[self.__data.index <= timestamp]
        self.test_data = self.__data.loc[self.__data.index > timestamp]

        logger.info("Split applied on: %s", timestamp)

        if n_splits_cv is not None:
            cv = TimeSeriesSplit(n_splits=n_splits_cv)
            splits = list(cv.split(self.train_data))
            for fold, (train_index, test_index) in enumerate(splits):
                logger.info("Fold %d: train %d→%d, test %d→%d",
                            fold, train_index[0], train_index[-1],
                            test_index[0], test_index[-1])
            self.__cv_tseries = iter(splits)
