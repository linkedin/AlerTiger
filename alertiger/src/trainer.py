import datetime
import logging
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras import Model
from keras.callbacks import History
from pandas import DataFrame

from .config import MlpLayerConfig
from .features import AlerTigerFeatures
from .model import AlerTigerForecastModelBuilder, AlerTigerClassificationModelBuilder
from .utils import normalize_timeseries


def train_alertiger_model(
        list_univariate_timeseries_data: List[DataFrame],
        epoch: int = 50,
        start_date: datetime.date = datetime.date(2023, 1, 1),
        end_date: datetime.date = datetime.date(2023, 7, 1),
        random_seed: int = 1,
        mlp_config_for_historical_values: List[MlpLayerConfig] = None,
        mlp_config_for_concatenation: List[MlpLayerConfig] = None,
        mlp_config_for_classification: List[MlpLayerConfig] = None) -> Tuple[Model, History, History]:
    """
    End-end training a AlerTiger anomaly detection model using the time series. This function do the following steps for training:
        - STEP-1: feature engineering using features.py
        - STEP-2: train the AlerTiger forecast model `alertiger_forecast_model` using model.py
        - STEP-3: train the AlerTiger classification model `alertiger_classification_model` using model.py
        - STEP-4: return the `alertiger_classification_model`, together with the loss for both training.

    :param list_univariate_timeseries_data: the list of timeseries thtat we will use for fitting the AlerTiger model. Here is the example schema of the timeseries
        dataframe that we expect to receive.
            ----------------------------------------------------------------------------------
            `DATE`                      `VALUE`             `ANOMALY`
            ----------------------------------------------------------------------------------
            2022-01-01                  1.0                 False
            2022-01-02                  1.1                 False
            2022-01-03                  1.2                 False
            2022-01-04                  1.3                 False
            2022-01-05                  1.2                 False
            2022-01-06                  1.1                 False
            2022-01-07                  1.0                 False
            2022-01-08                  1.0                 False
            2022-01-09                  1.1                 False
            2022-01-10                  1.2                 False
            2022-01-11                  1.3                 False
            2022-01-12                  1.2                 False
            2022-01-13                  1.1                 False
            2022-01-14                  1.0                 False
            2022-01-15                  1.0                 False
            2022-01-16                  10.1                True
            2022-01-17                  10.2                True
            2022-01-18                  1.3                 False
            ----------------------------------------------------------------------------------
    :param epoch: the number of epoch for training the AlerTiger forecast model and classification model.
    :param start_date: the starting date for which we will use for training the model
    :param end_date: the ending data of the duration for which we will use for training the model.
    :param random_seed: the random seed for the training.
    :param mlp_config_for_historical_values: the MLP layer structure, this MLP model will be applied on the historical time series. By default we use the
        following 4-layer MLP structure.
        ```
            mlp_config_for_concatenation = [
                MlpLayerConfig(units=16, activation="relu"),
                MlpLayerConfig(units=16, activation="relu"),
                MlpLayerConfig(units=8, activation="relu"),
                MlpLayerConfig(units=8, activation="relu")
            ]
        ```

    :param mlp_config_for_concatenation: the MLP layer structure, this MLP model will be applied on the concatenated tensor between historical timeseries
        values and the seasonality tensor.By default we use the
        following 3-layer MLP structure.
        ```
            mlp_config_for_historical_values = [
                MlpLayerConfig(units=14, activation="relu"),
                MlpLayerConfig(units=9, activation="relu"),
                MlpLayerConfig(units=9, activation="relu")
            ]
        ```

    :param mlp_config_for_classification: the MLP layer structure, this MLP model will be used in the classification model.By default we use the
        following 3-layer MLP structure.
        ```
            mlp_config_for_classification = [
                MlpLayerConfig(units=14, activation="relu"),
                MlpLayerConfig(units=9, activation="relu"),
                MlpLayerConfig(units=9, activation="relu")
            ]
        ```
    :return: the end-end trained AlerTiger anomaly detection model, together with the histories for training forecast and classification models.
    """
    # update the default values.
    if mlp_config_for_concatenation is None:
        mlp_config_for_concatenation = [
            MlpLayerConfig(units=16, activation="relu"),
            MlpLayerConfig(units=16, activation="relu"),
            MlpLayerConfig(units=8, activation="relu"),
            MlpLayerConfig(units=8, activation="relu")
        ]
    if mlp_config_for_historical_values is None:
        mlp_config_for_historical_values = [
            MlpLayerConfig(units=14, activation="relu"),
            MlpLayerConfig(units=9, activation="relu"),
            MlpLayerConfig(units=9, activation="relu")
        ]
    if mlp_config_for_classification is None:
        mlp_config_for_classification = [
            MlpLayerConfig(units=14, activation="relu"),
            MlpLayerConfig(units=9, activation="relu"),
            MlpLayerConfig(units=9, activation="relu")
        ]

    # set random seed.
    tf.keras.utils.set_random_seed(random_seed)

    # Normalization each time series to be 0 mean and 1 standard deviation.
    # Note that this will not introduce the information leakage (i.e. future value input past feature) because:
    # 1. we only do the global normalization in the training phase, not in inferencer.py
    # 2. the normalization will ensure the loss function is calculating in the normalized space, making various time series comparable.
    normalize_timeseries(list_univariate_timeseries_data)

    # dataset construction
    features: DataFrame = AlerTigerFeatures.feature_engineering(list_univariate_timeseries_data,
                                                                start_date=start_date,
                                                                end_date=end_date)

    logging.debug(f"# data points for training = {features.shape[0]}")

    alertiger_forecast_model_builder: AlerTigerForecastModelBuilder = AlerTigerForecastModelBuilder(
        mlp_config_for_historical_values=mlp_config_for_historical_values,
        mlp_config_for_concatenation=mlp_config_for_concatenation
    )

    alertiger_forecast_model: Model = alertiger_forecast_model_builder.build(historical_value_dim=28, seasonality_input_dim=7)
    alertiger_forecast_model.compile(optimizer="adam", loss=alertiger_forecast_model_builder.loss)
    alertiger_forecast_model.summary()

    history_forecast: History = alertiger_forecast_model.fit(
        x={
            "INPUT_NAME_HISTORICAL_VALUE": np.array(features["historical_value_tensor"].to_numpy().tolist()),
            "INPUT_NAME_SEASONALITY": np.array(features["day_of_week_tensor"].to_numpy().tolist())
        },
        y=features[["current_value", "anomaly_label"]],
        epochs=epoch
    )

    # classification model
    alertiger_classification_model_builder: AlerTigerClassificationModelBuilder = AlerTigerClassificationModelBuilder(mlp_layers=mlp_config_for_classification)
    alertiger_classification_model: Model = alertiger_classification_model_builder.build(alertiger_forecast_model=alertiger_forecast_model)
    alertiger_classification_model.compile(optimizer="adam", loss=alertiger_classification_model_builder.loss)
    alertiger_classification_model.summary()
    history_classification: History = alertiger_classification_model.fit(
        x={
            "INPUT_NAME_HISTORICAL_VALUE": np.array(features["historical_value_tensor"].to_numpy().tolist()),
            "INPUT_NAME_SEASONALITY": np.array(features["day_of_week_tensor"].to_numpy().tolist()),
            "CURRENT": np.array(features["current_value"].to_numpy().tolist()),
        },
        y=features[["current_value", "anomaly_label"]],
        epochs=epoch
    )
    return alertiger_classification_model, history_forecast, history_classification
