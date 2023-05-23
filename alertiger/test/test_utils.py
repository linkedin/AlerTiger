import datetime
from typing import List
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame

from alertiger.src.constants import COLUMN_NAME_DATE, COLUMN_NAME_VALUE, COLUMN_NAME_PREDICTED_LOWER_BOUND, COLUMN_NAME_PREDICTED_UPPER_BOUND, \
    COLUMN_NAME_PREDICTED_BASELINE, COLUMN_NAME_PREDICTED_ANOMALY
from alertiger.src.utils import mock_univariate_time_series_with_anomaly, visualize_time_series_with_anomaly, normalize_timeseries, duration_severity_filter


class TestUtils(TestCase):

    def test_mock_univariate_time_series_with_anomaly(self):
        timeseries: List[DataFrame] = mock_univariate_time_series_with_anomaly(
            start_date=datetime.date(2022, 1, 1),
            end_date=datetime.date(2023, 1, 1),
            seasonality_number_of_square_timeseries=10,
            seasonality_number_of_triangle_timeseries=10,
            seasonality_number_of_sine_timeseries=10,
            seasonality_number_of_constant_timeseries=10)

        self.assertEqual(len(timeseries), 40)

    def test_visualize_time_series_with_anomaly(self):
        mock_timeseries = mock_univariate_time_series_with_anomaly()
        visualize_time_series_with_anomaly(mock_timeseries[0])

    def test_normalize_timeseries(self):
        df = pd.DataFrame({
            COLUMN_NAME_DATE: pd.date_range(
                start=pd.to_datetime("2023-1-1"),
                end=pd.to_datetime("2023-1-15")),
            COLUMN_NAME_VALUE: np.arange(1, 16) * 1.0
        })
        normalize_timeseries([df])
        self.assertListEqual(
            [-1.5652440842576791, -1.341637786506582, -1.118031488755485, -0.894425191004388, -0.670818893253291, -0.447212595502194, -0.223606297751097, 0.0,
             0.223606297751097, 0.447212595502194, 0.670818893253291, 0.894425191004388, 1.118031488755485, 1.341637786506582, 1.5652440842576791],
            df[COLUMN_NAME_VALUE].tolist()
        )

    def test_duration_severity_filter(self):
        normal_value: float = 0.0
        less_severe_abnormal_value: float = 0.5
        more_severe_abnormal_value: float = 10.0
        df = pd.DataFrame({
            COLUMN_NAME_DATE: pd.date_range(start=pd.to_datetime("2023-1-1"),
                                            end=pd.to_datetime("2023-1-15")),
            COLUMN_NAME_VALUE: [normal_value] * 5 +
                               [less_severe_abnormal_value] * 2 +  # this should not be alerted because of less severity
                               [normal_value] * 2 +
                               [more_severe_abnormal_value] * 1 +  # this should not be alerted because of short duration
                               [normal_value] * 2 +
                               [more_severe_abnormal_value] * 2 +  # this should be alerted because both duration and severity.
                               [normal_value] * 1,
            COLUMN_NAME_PREDICTED_LOWER_BOUND: [-0.5] * 15,
            COLUMN_NAME_PREDICTED_UPPER_BOUND: [0.5] * 15,
            COLUMN_NAME_PREDICTED_BASELINE: [0.0] * 15,
            COLUMN_NAME_PREDICTED_ANOMALY: [False] * 5 + [True] * 2 + [False] * 2 + [True] * 1 + [False] * 2 + [True] * 2 + [False] * 1
        })

        df_filtered: DataFrame = duration_severity_filter(df)

        # only the last anomaly is kept, others are all removed (i.e. "COLUMN_NAME_PREDICTED_ANOMALY" columns changed from False to True.
        self.assertListEqual(
            [False] * 12 + [True] * 2 + [False] * 1,
            df_filtered[COLUMN_NAME_PREDICTED_ANOMALY].tolist()
        )
