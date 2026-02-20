# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from pasts.core.base_model import TimeSeriesModel


class DartsModel(TimeSeriesModel):
    """A time series model backed by a Darts ML/DL estimator.

    Unlike :class:`~pasts.components.parametric_model.ParametricModel`,
    a ``DartsModel`` learns patterns from data rather than fitting a
    closed-form function of time.  It can be applied to any
    :class:`~pasts.core.datacube.DataCube` or ``pd.DataFrame`` —
    the raw signal, a residual after detrending, or any intermediate series.

    Parameters
    ----------
    model : darts.models base class instance
        Any fitted or unfitted Darts model
        (e.g. ``NaiveSeasonal()``, ``ExponentialSmoothing()``).

    Notes
    -----
    ``reverse_transform(i > 0)`` calls ``model.predict(i)`` for future values.
    ``reverse_transform(i < 0)`` calls ``model.historical_forecasts`` to
    retrieve the |i| last in-sample one-step-ahead predictions.
    """

    def __init__(self, model):
        self._model = model

    def fit(self, X) -> "DartsModel":
        """Fit the Darts model on *X*.

        Parameters
        ----------
        X : pd.DataFrame or :class:`~pasts.core.datacube.DataCube`
            Training data. Both the original signal and any residual are valid.

        Returns
        -------
        self
        """
        from pasts.core.datacube import DataCube
        if isinstance(X, DataCube):
            X = X.data
        from darts import TimeSeries
        self._train_series = TimeSeries.from_dataframe(X)
        self._model.fit(self._train_series)
        self._n_train = len(self._train_series)
        return self

    def reverse_transform(self, i: int):
        """Return component values for recomposition.

        Parameters
        ----------
        i : int
            i > 0 : forecast the next *i* steps.
            i < 0 : return the last *|i|* one-step-ahead in-sample predictions.
        """
        if i > 0:
            return self._model.predict(i).to_dataframe()
        hf = self._model.historical_forecasts(
            self._train_series,
            start=self._n_train + i,
            forecast_horizon=1,
            retrain=False,
        )
        return hf.to_dataframe()
