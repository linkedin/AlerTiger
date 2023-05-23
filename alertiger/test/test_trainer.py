import datetime
import math
from unittest import TestCase

import numpy as np
import pandas as pd

from alertiger.src.constants import COLUMN_NAME_VALUE, COLUMN_NAME_DATE, COLUMN_NAME_ANOMALY
from alertiger.src.trainer import train_alertiger_model


class TestTrainer(TestCase):

    def test_train_alertiger_model(self):
        univariate_timeseries = [
            pd.DataFrame({
                COLUMN_NAME_DATE: pd.to_datetime([datetime.date(2023, 1, 1) + datetime.timedelta(days=x) for x in range(100)]),
                COLUMN_NAME_VALUE: np.array([math.sin(2 * math.pi * i / 7) for i in range(100)]) + np.array([0] * 95 + [1] * 5),
                COLUMN_NAME_ANOMALY: [False] * 95 + [True] * 5
            })
        ]
        _, history_forecast, history_classification = train_alertiger_model(
            univariate_timeseries,
            epoch=3,
            random_seed=37)

        self.assertEquals(3, len(history_forecast.history['loss']))
        self.assertEquals(3, len(history_classification.history['loss']))
