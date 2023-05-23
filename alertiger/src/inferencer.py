import datetime
from typing import List

import numpy as np
import pandas as pd
from keras import Model
from pandas import DataFrame

from .constants import COLUMN_NAME_DATE, \
    COLUMN_NAME_PREDICTED_BASELINE, COLUMN_NAME_PREDICTED_LOWER_BOUND, COLUMN_NAME_PREDICTED_UPPER_BOUND, \
    COLUMN_NAME_PREDICTED_ANOMALY_SCORE, COLUMN_NAME_PREDICTED_ANOMALY, ANOMALY_SCORE_THRESHOLD, COLUMN_NAME_VALUE
from .features import AlerTigerFeatures
from .utils import duration_severity_filter


def inference_alertiger_model(
        univariate_timeseries: List[DataFrame],
        alertiger_model: Model,
        start_date: datetime.date = datetime.date(2023, 1, 1),
        end_date: datetime.date = datetime.date(2023, 7, 1),
        duration_threshold: int = 2,
        duration_window_size: int = 3,
        severity_threshold: float = 1.0) -> List[DataFrame]:
    """
    Run anomaly detection on the univariate_timeseries using the training alertiger_model.

    :param univariate_timeseries: the list of univariate timeseries that we will run anomaly detection on. We should have `DATE` and `VALUE` as the input column
        below is such an example
            ----------------------------------------------------------------------------------
            `DATE`                      `VALUE`
            ----------------------------------------------------------------------------------
            2022-01-01                  1.0
            2022-01-02                  1.1
            2022-01-03                  1.2
            2022-01-04                  1.3
            2022-01-05                  1.2
            2022-01-06                  1.1
            2022-01-07                  1.0
            2022-01-08                  1.0
            2022-01-09                  1.1
            2022-01-10                  1.2
            2022-01-11                  1.3
            2022-01-12                  1.2
            2022-01-13                  1.1
            2022-01-14                  1.0
            2022-01-15                  1.0
            2022-01-16                  10.1
            2022-01-17                  10.2
            2022-01-18                  1.3
            ----------------------------------------------------------------------------------
    :param alertiger_model: the trained Keras AlerTiger model (see trainer.py for details)
    :param start_date: the starting time of anomaly detection
    :param end_date: the end time of anomaly detection
    :param duration_threshold: the duration filter of anomaly detection.
        the duration is defined as the number of anomaly data points within a sliding window.
    :param duration_window_size: the duration filter's window size, we will keep an anomaly if within a window of size `duration_window_size` there are at least
        duration_threshold number of data points.
    :param severity_threshold: the severity filter of anomaly detection.
        the severity is calculated using |predicted_baseline-actual_value|/|upper_bound-lower_bound|
    :return: the list of dataframe, each corresponds to the input dataframe. Each dataframe has the following schema:
        - DATE: the date for which we make prediction (same as the input column)
        - VALUE: the time series value (same as the input column)
        - BASELINE: the predicted baseline
        - LOWER_BOUND: the predicted confidence interval's lower bound
        - UPPER_BOUND: the predicted confidence interval's upper bound
        - ANOMALY_SCORE: the anomaly score, which is the stochastic anomaly detection result that the anomaly classifier believe the `VALUE` is abnormal.
        - DETECTED_ANOMALY: a boolean value, which is the deterministic anomaly detection result that the anomaly classifier believe the `VALUE` is abnormal.
    """

    result: List[DataFrame] = []
    for timeseries in univariate_timeseries:
        # dataset construction
        features: DataFrame = AlerTigerFeatures.feature_engineering([timeseries],
                                                                    start_date=start_date,
                                                                    end_date=end_date)

        prediction: np.array = alertiger_model.predict(
            {
                "INPUT_NAME_HISTORICAL_VALUE": np.array(features["historical_value_tensor"].to_numpy().tolist()),
                "INPUT_NAME_SEASONALITY": np.array(features["day_of_week_tensor"].to_numpy().tolist()),
                "CURRENT": np.array(features["current_value"].to_numpy().tolist()),
            }
        )
        df_prediction: pd.DataFrame = pd.DataFrame({
            COLUMN_NAME_DATE: features[COLUMN_NAME_DATE],
            COLUMN_NAME_PREDICTED_BASELINE: prediction[:, 0],
            COLUMN_NAME_PREDICTED_LOWER_BOUND: prediction[:, 1],
            COLUMN_NAME_PREDICTED_UPPER_BOUND: prediction[:, 2],
            COLUMN_NAME_PREDICTED_ANOMALY_SCORE: prediction[:, 3]
        })

        timeseries_with_prediction = timeseries.merge(df_prediction, on=COLUMN_NAME_DATE, how="left")

        # anomaly detection result
        timeseries_with_prediction[COLUMN_NAME_PREDICTED_ANOMALY] = \
            (timeseries_with_prediction[COLUMN_NAME_PREDICTED_ANOMALY_SCORE] > ANOMALY_SCORE_THRESHOLD) & \
            ~((timeseries_with_prediction[COLUMN_NAME_PREDICTED_LOWER_BOUND] <= timeseries_with_prediction[COLUMN_NAME_VALUE]) &
              (timeseries_with_prediction[COLUMN_NAME_VALUE] <= timeseries_with_prediction[COLUMN_NAME_PREDICTED_UPPER_BOUND]))

        # duration and severity anomaly filtering.
        timeseries_with_prediction = duration_severity_filter(timeseries_with_prediction,
                                                              duration_threshold=duration_threshold,
                                                              duration_window_size=duration_window_size,
                                                              severity_threshold=severity_threshold)

        result.append(timeseries_with_prediction)
    return result
