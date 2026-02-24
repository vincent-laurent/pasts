# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import glob
import os
import re
import warnings

import joblib
import pandas as pd


def save_model(path: str, model_name: str, model_data, suffix: str = "train") -> None:
    """Save a model entry to a joblib file.

    Parameters
    ----------
    path : str
        Directory where the file will be written.
    model_name : str
        Name used in the filename (e.g. ``"ExponentialSmoothing"``).
    model_data : ModelResult or dict
        The model result (predictions, model, scores, …).
    suffix : str
        ``"train"`` or ``"final"``.
    """
    joblib.dump(model_data, os.path.join(path, f'{model_name}_{suffix}_jlib'))


def save_common_data(path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save train and test DataFrames to joblib files."""
    joblib.dump(test_data, os.path.join(path, 'test_data_jlib'))
    joblib.dump(train_data, os.path.join(path, 'train_data_jlib'))


def _load_saved_file(file: str, filename: str, models: dict,
                     set_train: callable, set_test: callable) -> None:
    """Load a single joblib file into the appropriate slot.

    Parameters
    ----------
    file : str
        Full path to the joblib file.
    filename : str
        Basename of the file.
    models : dict
        Reference to ``Signal.models``.
    set_train : callable
        Callback to set train data on the Signal.
    set_test : callable
        Callback to set test data on the Signal.
    """
    match_data = re.search(r'(.+)_data_jlib', filename)
    if match_data:
        name = match_data.group(1)
        if name == 'test':
            set_test(joblib.load(file))
        elif name == 'train':
            set_train(joblib.load(file))
        return

    match_final = re.search(r'(.+)_final_jlib', filename)
    if match_final:
        models[match_final.group(1)] = joblib.load(file)
        return

    match_train = re.search(r'(.+)_train_jlib', filename)
    if match_train:
        name = match_train.group(1)
        if name not in models or 'forecast' not in models[name]:
            models[name] = joblib.load(file)
        return

    warnings.warn(f"File {filename} does not correspond to a saved model.")


def load_saved_models(path: str, models: dict,
                      set_train: callable, set_test: callable) -> None:
    """Load all joblib files from *path* into the models dict.

    Parameters
    ----------
    path : str
        Directory containing joblib files.
    models : dict
        Reference to ``Signal.models``.
    set_train : callable
        Callback to set train data on the Signal.
    set_test : callable
        Callback to set test data on the Signal.
    """
    files = glob.glob(os.path.join(path, "*jlib"))
    if not files:
        warnings.warn("No saved models were found.")
        return
    for file in files:
        _load_saved_file(file, os.path.basename(file), models, set_train, set_test)
