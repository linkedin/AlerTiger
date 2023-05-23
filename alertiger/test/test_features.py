import copy
from unittest import TestCase

import numpy as np
import pandas as pd

from alertiger.src.constants import COLUMN_NAME_DATE, COLUMN_NAME_VALUE, FEATURE_NAME_DAY_OF_WEEK_TENSOR, COLUMN_NAME_ANOMALY, LABEL_NAME_ANOMALY_LABEL, \
    FEATURE_NAME_HISTORICAL_VALUE_TENSOR, COLUMN_NAME_IS_VALID_FEATURE
from alertiger.src.features import AlerTigerFeatures


class TestFeatures(TestCase):

    def setUp(self) -> None:
        self.timeseries = pd.DataFrame(
            {
                COLUMN_NAME_DATE: pd.date_range(start=pd.to_datetime("2023-1-1"), end=pd.to_datetime("2023-2-28")),
                COLUMN_NAME_VALUE: np.sin([i / 7.0 * 2 * np.pi for i in range(59)]),
                COLUMN_NAME_ANOMALY: [False] * 59
            }
        )
        # inject anomaly
        self.timeseries[self.timeseries[COLUMN_NAME_DATE] > pd.to_datetime("2023-2-25")][COLUMN_NAME_VALUE] += 10
        self.timeseries[self.timeseries[COLUMN_NAME_DATE] > pd.to_datetime("2023-2-25")][COLUMN_NAME_ANOMALY] = True

    def test_feature_generation_day_of_week(self):
        df_feature_day_of_week: pd.DataFrame = AlerTigerFeatures._feature_generation_day_of_week(df_data=copy.deepcopy(self.timeseries),
                                                                                                 feature_name=FEATURE_NAME_DAY_OF_WEEK_TENSOR)
        self.assertEqual((59, 4), df_feature_day_of_week.shape)

        # 2023-1-1 is Sunday, therefore the 1.0 is at the last index.
        self.assertListEqual(
            df_feature_day_of_week.at[0, FEATURE_NAME_DAY_OF_WEEK_TENSOR],
            [0.0] * 6 + [1.0],
        )

    def test_feature_generation_label(self):
        df_feature_label: pd.DataFrame = AlerTigerFeatures._feature_generation_label(df_data=copy.deepcopy(self.timeseries),
                                                                                     feature_name=LABEL_NAME_ANOMALY_LABEL)
        self.assertEqual((59, 4), df_feature_label.shape)
        self.assertEqual(
            0,
            df_feature_label.at[58, LABEL_NAME_ANOMALY_LABEL]
        )

    def test_feature_generation_historical_values(self):
        timeseries = copy.deepcopy(self.timeseries)
        timeseries[COLUMN_NAME_IS_VALID_FEATURE] = True
        df_feature_historical_values: pd.DataFrame = AlerTigerFeatures._feature_generation_historical_values(df_data=timeseries,
                                                                                                             feature_name=FEATURE_NAME_HISTORICAL_VALUE_TENSOR)

        self.assertEqual((59, 5), df_feature_historical_values.shape)
        self.assertListEqual(
            [0.9749279121818238, 0.4338837391175591, -0.4338837391175571, -0.9749279121818234, -0.7818314824680327, -1.2246467991473533e-15, 0.781831482468029,
             0.9749279121818238, 0.4338837391175593, -0.4338837391175569, -0.9749279121818233, -0.7818314824680307, -1.4695761589768238e-15, 0.7818314824680288,
             0.974927912181824, 0.43388373911755956, -0.4338837391175567, -0.9749279121818233, -0.7818314824680308, -1.7145055188062944e-15, 0.7818314824680287,
             0.974927912181824, 0.43388373911755973, -0.43388373911755646, -0.9749279121818232, -0.781831482468031, -1.959434878635765e-15, 0.7818314824680286],
            df_feature_historical_values.at[58, FEATURE_NAME_HISTORICAL_VALUE_TENSOR]
        )
        self.assertTrue(np.isnan(df_feature_historical_values.at[0, FEATURE_NAME_HISTORICAL_VALUE_TENSOR]))
