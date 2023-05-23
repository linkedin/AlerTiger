import datetime
import math
from unittest import TestCase

import numpy as np
import pandas as pd

from alertiger.src.constants import COLUMN_NAME_DATE, COLUMN_NAME_VALUE, COLUMN_NAME_ANOMALY, COLUMN_NAME_PREDICTED_ANOMALY, COLUMN_NAME_PREDICTED_ANOMALY_SCORE, \
    COLUMN_NAME_PREDICTED_UPPER_BOUND, COLUMN_NAME_PREDICTED_LOWER_BOUND, COLUMN_NAME_PREDICTED_BASELINE
from alertiger.src.inferencer import inference_alertiger_model
from alertiger.src.trainer import train_alertiger_model


class TestInferencer(TestCase):

    def test_inference_alertiger_model(self):
        # let's first train a alertiger model
        univariate_timeseries = [
            pd.DataFrame({
                COLUMN_NAME_DATE: pd.to_datetime([datetime.date(2023, 1, 1) + datetime.timedelta(days=x) for x in range(100)]),
                COLUMN_NAME_VALUE: np.array([math.sin(2 * math.pi * i / 7) for i in range(100)]) + np.array([0] * 95 + [1] * 5),
                COLUMN_NAME_ANOMALY: [False] * 95 + [True] * 5
            })
        ]
        alertiger_classification_model, history_forecast, history_classification = train_alertiger_model(
            univariate_timeseries,
            epoch=10,
            random_seed=37)
        inf_res = inference_alertiger_model(univariate_timeseries, alertiger_classification_model)
        self.assertSetEqual(
            {
                COLUMN_NAME_DATE,
                COLUMN_NAME_VALUE,
                COLUMN_NAME_ANOMALY,
                COLUMN_NAME_PREDICTED_BASELINE,
                COLUMN_NAME_PREDICTED_LOWER_BOUND,
                COLUMN_NAME_PREDICTED_UPPER_BOUND,
                COLUMN_NAME_PREDICTED_ANOMALY_SCORE,
                COLUMN_NAME_PREDICTED_ANOMALY
            },
            set(inf_res[0].columns)
        )
