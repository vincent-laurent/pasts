# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class Trend:
    """
    A class to remove and put back the trend in time series.

    Attributes
    ----------
    origin_index : TimeIndex
        TimeIndex of passed series.
    time_index: numpy array
        TimeIndex converted to float index.
    features_: list[str]
        Columns of passed series.
    coef_: pd.Dataframe
        Trend coefficient
    intercept_: pd.Dataframe
        Trend intercept.
    dt_: float
        Average difference between two consecutive indexes.
    t0_: float
        Last index.

    Methods
    -------
    fit(X, y):
        Extracts trend from passed X series.
    _from_i_to_vector(i):
        Computes past or future float index.
    _from_i_to_time_index(i):
        Computes past or future time index.
    _t(i):
        Creates a matrix to be used when transforming a series.
    _intercept_to_frame(shape):
        Converts the trend intercept into a matrix, used for transforming.
    transform(i, y=None):
        Computes Dataframe to remove trend from a series.
    reverse_transform(i, y=None):
        Computes Dataframe to add trend to a series.
    """
    def __init__(self):
        ...

    def fit(self, X: pd.DataFrame) -> "Trend":
        """
        Finds linear trend in passed series.

        Parameters
        ----------
        X : pd.Dataframe
            Time series to detrend.
            Index must be TimeIndex.

        Returns
        -------
        Modified Trend object

        """
        self.origin_index = X.index
        self.time_index = X.index.to_numpy(dtype="float64")
        self.time_index = self.time_index.astype(float)*1e-17
        self.features_ = X.columns
        frame = self.time_index.reshape(-1, 1)
        estimator = LinearRegression()
        estimator.fit(frame, X.values)
        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_
        self.dt_ = np.mean(np.diff(self.time_index))
        self.t0_ = self.time_index[-1]
        return self

    def _from_i_to_vector(self, i: int) -> np.array:
        """
        Computes past or future float index of length |i|.

        Parameters
        ----------
        i : int
            Size of output index.
            If positive, extension of attribute time_index.
            If negative, last values of time_index.

        Returns
        -------
        Array of size |i|
        """
        if i > 0:
            return self.t0_ + np.array(range(i)) * self.dt_
        else:
            return self.time_index[i:]

    def _from_i_to_time_index(self, i: int) -> pd.DatetimeIndex:
        """
        Computes past or future time index of length |i|.

        Parameters
        ----------
        i : int
            Size of output index.
            If positive, extension of attribute time_index.
            If negative, last values of time_index.

        Returns
        -------
        TimeIndex of size |i|
        """
        if i > 0:
            time_index = pd.date_range(start=self.origin_index[-1], periods=i+1, freq=self.origin_index.freq)[1:]
        else:
            time_index = self.origin_index[i:]
        return time_index

    def _t(self, i: int) -> np.array:
        """
        Creates a matrix to be used when transforming a series.

        Parameters
        ----------
        i : int
            Size of output.

        Returns
        -------
        Computed matrix.
        """
        t = self._from_i_to_vector(i)
        t = t.reshape(1, -1) * np.ones((
            self.coef_.shape[0],
            np.abs(i)))
        return t

    def _intercept_to_frame(self, shape: int) -> np.array:
        """
        Converts the trend intercept into a matrix, used for transforming.

        Parameters
        ----------
        shape : int
            Length of output

        Returns
        -------
        Computed matrix.
        """
        return self.intercept_.reshape(-1, 1) * np.ones((
            self.coef_.shape[0],
            shape))

    def transform(self, i: int) -> pd.DataFrame:
        """
        Computes Dataframe to remove trend from a series.
        Removes trend when added to a dataframe of same shape.

        Parameters
        ----------
        i : int
            Length of ouput.

        Returns
        -------
        Computed dataframe.
        """
        t = self._t(i)
        b = self._intercept_to_frame(t.shape[1])
        ret = - self.coef_ * t - b
        return pd.DataFrame(ret,
                            columns=self._from_i_to_time_index(i),
                            index=self.features_.to_list()).T

    def reverse_transform(self, i: int) -> pd.DataFrame:
        """
        Computes Dataframe to add trend to a series.
        Adds trend when added to a dataframe of same shape.

        Parameters
        ----------
        i : int
            Length of output.

        Returns
        -------
        Computed dataframe.
        """
        t = self._t(i)
        b = self._intercept_to_frame(t.shape[1])
        ret = self.coef_ * t + b
        return pd.DataFrame(ret,
                            columns=self._from_i_to_time_index(i),
                            index=self.features_.to_list()).T


class Seasonality:
    """
    A class to remove and put back the seasonality in times series.

    Attributes
    ----------
    seasonality: int
        Seasonality period.
    time_index: TimeIndex
        Index of passed series.
    seasonal_component: pd.Dataframe
        Extracted seasonal component of the series.
    estimator_future_season: XGBModel instance
        Fitted estimator used to predict future seasonal component.

    Methods
    -------
    fit(X, y):
        Extracts seasonality from passed X series.
    transform(i, y=None):
        Computes Dataframe to remove seasonality from a series.
    reverse_transform(i, y=None):
        Computes Dataframe to add seasonality to a series.
    """
    def __init__(self, seasonality: int):
        """
        Constructs all the necessary attributes for the seasonality object.

        Parameters
        ----------
        seasonality: int
            Seasonality period.
        """
        self.period = int(seasonality)

    def fit(self, X: pd.DataFrame) -> None:
        """
        Finds seasonal component in passed series.
        Fills attributes seasonal_component and estimator_future_season.

        Parameters
        ----------
        X : pd.Dataframe
            Time series.
            Index must be of type DatetimeIndex.

        Returns
        -------
        None
        """
        self.time_index = X.index
        df_diff = pd.DataFrame(index=self.time_index, columns=X.columns)
        for col in X.columns:
            values = X[col].values
            diff = [0 for _ in range(self.period)]
            for i in range(self.period, len(values)):
                res = values[i - self.period]
                diff.append(res)
            df_diff[col] = diff
        self.seasonal_component = df_diff
        from darts import TimeSeries
        from darts.models import XGBModel
        self.estimator_future_season = XGBModel(lags=self.period)
        self.estimator_future_season.fit(TimeSeries.from_dataframe(self.seasonal_component))

    def transform(self, i: int) -> pd.DataFrame:
        """
        Computes Dataframe to remove seasonal component from a series.
        Removes seasonality when added to a dataframe of same shape.

        Parameters
        ----------
        i : int
            Length of ouput.

        Returns
        -------
        Computed dataframe.
        """
        if i > 0:
            output = self.estimator_future_season.predict(i).to_dataframe()
        else:
            output = self.seasonal_component.iloc[i:]
        return - output

    def reverse_transform(self, i: int) -> pd.DataFrame:
        """
        Computes Dataframe to add seasonal component to a series.
        Adds seasonality when added to a dataframe of same shape.

        Parameters
        ----------
        i : int
           Length of output.

        Returns
        -------
        Computed dataframe.
        """
        if i > 0:
            output = self.estimator_future_season.predict(i).to_dataframe()
        else:
            output = self.seasonal_component.iloc[i:]
        return output


