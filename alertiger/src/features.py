import copy
from datetime import datetime
from functools import reduce
from typing import List

import numpy as np
import pandas as pd
import tqdm as tqdm
from pandas import DataFrame, Timedelta

from .constants import COLUMN_NAME_DATE, COLUMN_NAME_VALUE, COLUMN_NAME_ANOMALY, COLUMN_NAME_IS_VALID_FEATURE, FEATURE_NAME_DAY_OF_WEEK_TENSOR, \
    FEATURE_NAME_CURRENT_VALUE, FEATURE_NAME_HISTORICAL_VALUE_TENSOR, LABEL_NAME_ANOMALY_LABEL


class AlerTigerFeatures:
    """
    Take time series as input, do the feature engineering for the time series. The main function of this class is `feature_engineering` which has the following
    interface:

    Input:
        df_datas: a dataframe with 3 columns: "DATE", "VALUE", and "ANOMALY"

    Output:
        dataset: a tensorflow Dataset object with the following columns as the feature.
            - day_of_week_tensor: a one-hot-encoded seasonality vector of size 7
            - historical_value_tensor: a float vector of size `historical_length` that include the previous H = `historical_length` days' value
            - current_value: a float value for the current value (i.e. the regression target)
            - anomaly_label: a boolean value that's the anomaly detection classification label.
    """

    @staticmethod
    def feature_engineering(
            df_datas: List[DataFrame],
            start_date: datetime.date,
            end_date: datetime.date,
            historical_length: int = 28) -> DataFrame:
        """
        Generate the features for AlerTiger model, this function can handle a bunch of time series in the form of List[DataFrame]

        :param df_datas: the list of time series dataframe, each dataframe should have the following schema with examples:
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
        :param start_date: the starting date of the duration for which we will generate features for.
        :param end_date: the ending date of the duration for which we will generate features for.
        :param historical_length: the historical length for the historical features. The unit is the number of data points. For instance, for the daily
            granularity data, a historical_length of 28 means we will use the past 4 weeks' data as the feature for making the prediction.
        :return: the dataframe with the feature generated on new columns. Now we support the following the features:
            - day_of_week_tensor: a one-hot-encoded seasonality vector of size 7
            - historical_value_tensor: a float vector of size `historical_length` that include the previous H = `historical_length` days' value
            - current_value: a float value for the current value (i.e. the regression target)
            - anomaly_label: a boolean value that's the anomaly detection classification label.
        """

        features: List[DataFrame] = []
        for df_data in tqdm.tqdm(df_datas):
            features.append(AlerTigerFeatures.feature_engineering_univariate_timeseries(df_data, start_date=start_date, end_date=end_date,
                                                                                        historical_length=historical_length))

        features: DataFrame = reduce(lambda d1, d2: pd.concat([d1, d2], axis=0), features)
        return features

    @staticmethod
    def feature_engineering_univariate_timeseries(
            df_data: DataFrame,
            start_date: datetime.date,
            end_date: datetime.date,
            historical_length: int) -> DataFrame:
        """
        Generate the feature for a single time series.

        :param df_data: the list of time series dataframe, each dataframe should have the following schema with examples:
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
        :param start_date: the starting date of the duration for which we will generate features for.
        :param end_date: the ending date of the duration for which we will generate features for.
        :param historical_length: the historical length for the historical features. The unit is the number of data points. For instance, for the daily
            granularity data, a historical_length of 28 means we will use the past 4 weeks' data as the feature for making the prediction.
        :return: the feature dataframe with the input columns together with the following feature columns:
            - day_of_week_tensor: a one-hot-encoded seasonality vector of size 7
            - historical_value_tensor: a float vector of size `historical_length` that include the previous H = `historical_length` days' value
            - current_value: a float value for the current value (i.e. the regression target)
            - anomaly_label: a boolean value that's the anomaly detection classification label.
        """

        # load the data.
        df_data = df_data[[COLUMN_NAME_DATE, COLUMN_NAME_VALUE, COLUMN_NAME_ANOMALY]]

        # filter on the date time
        df_data = df_data[(df_data[COLUMN_NAME_DATE] >= pd.to_datetime(start_date.strftime("%Y-%m-%d %H:%M:%S"))) & (
                df_data[COLUMN_NAME_DATE] <= pd.to_datetime(end_date.strftime("%Y-%m-%d %H:%M:%S")))]

        df_data[COLUMN_NAME_IS_VALID_FEATURE] = True

        # feature engineering
        df_data = AlerTigerFeatures._feature_generation_day_of_week(df_data=df_data, feature_name=FEATURE_NAME_DAY_OF_WEEK_TENSOR)
        df_data = AlerTigerFeatures._feature_generation_current_value(df_data=df_data, feature_name=FEATURE_NAME_CURRENT_VALUE)
        df_data = AlerTigerFeatures._feature_generation_historical_values(df_data=df_data, feature_name=FEATURE_NAME_HISTORICAL_VALUE_TENSOR,
                                                                          historical_length=historical_length)
        df_data = AlerTigerFeatures._feature_generation_label(df_data=df_data, feature_name=LABEL_NAME_ANOMALY_LABEL)

        # remove invalid features
        df_data = df_data[df_data[COLUMN_NAME_IS_VALID_FEATURE]]
        features: DataFrame = df_data[[
            COLUMN_NAME_DATE,
            FEATURE_NAME_HISTORICAL_VALUE_TENSOR,
            FEATURE_NAME_DAY_OF_WEEK_TENSOR,
            FEATURE_NAME_CURRENT_VALUE,
            LABEL_NAME_ANOMALY_LABEL]]
        return features

    @staticmethod
    def _feature_generation_day_of_week(
            df_data: DataFrame,
            feature_name: str = FEATURE_NAME_DAY_OF_WEEK_TENSOR) -> DataFrame:
        """
        Generate the feature "day of week" one-hot-encoding, we will generate a new column named COLUMN_NAME_FEATURES and the
        day of week is generated using the column COLUMN_NAME_DATETIME

        :param df_data: input dataframe with the following schema with examples:
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
        :return: the updated dataframe
            ----------------------------------------------------------------------------------
            `DATE`                      `VALUE`             `ANOMALY`       `day_of_week_tensor`
            ----------------------------------------------------------------------------------
            2022-01-01                  1.0                 False           [0,0,0,0,0,0,1]
            2022-01-02                  1.1                 False           [1,0,0,0,0,0,0]
            2022-01-03                  1.2                 False           [0,1,0,0,0,0,0]
            2022-01-04                  1.3                 False           [0,0,1,0,0,0,0]
            2022-01-05                  1.2                 False           [0,0,0,1,0,0,0]
            2022-01-06                  1.1                 False           [0,0,0,0,1,0,0]
            2022-01-07                  1.0                 False           [0,0,0,0,0,1,0]
            2022-01-08                  1.0                 False           [0,0,0,0,0,0,1]
            2022-01-09                  1.1                 False           [1,0,0,0,0,0,0]
            2022-01-10                  1.2                 False           [0,1,0,0,0,0,0]
            2022-01-11                  1.3                 False           [0,0,1,0,0,0,0]
            2022-01-12                  1.2                 False           [0,0,0,1,0,0,0]
            2022-01-13                  1.1                 False           [0,0,0,0,1,0,0]
            2022-01-14                  1.0                 False           [0,0,0,0,0,1,0]
            2022-01-15                  1.0                 False           [0,0,0,0,0,0,1]
            2022-01-16                  10.1                True            [1,0,0,0,0,0,0]
            2022-01-17                  10.2                True            [0,1,0,0,0,0,0]
            2022-01-18                  1.3                 False           [0,0,1,0,0,0,0]
            ----------------------------------------------------------------------------------
        """
        df_data[feature_name] = df_data[COLUMN_NAME_DATE].apply(
            lambda x: AlerTigerFeatures._day_of_week_one_hot_encoding(x.dayofweek))
        return df_data

    @staticmethod
    def _feature_generation_current_value(
            df_data: DataFrame,
            feature_name: str = FEATURE_NAME_CURRENT_VALUE) -> DataFrame:
        """
        Generate the feature "current_value", we will generate a new column besides the input dataframe named COLUMN_NAME_FEATURES.

        :param df_data: input dataframe with the following schema with examples:
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
        :return: the updated dataframe
            ----------------------------------------------------------------------------------
            `DATE`                      `VALUE`             `ANOMALY`       `current_value`
            ----------------------------------------------------------------------------------
            2022-01-01                  1.0                 False           1.0
            2022-01-02                  1.1                 False           1.1
            2022-01-03                  1.2                 False           1.2
            2022-01-04                  1.3                 False           1.3
            2022-01-05                  1.2                 False           1.2
            2022-01-06                  1.1                 False           1.1
            2022-01-07                  1.0                 False           1.0
            2022-01-08                  1.0                 False           1.0
            2022-01-09                  1.1                 False           1.1
            2022-01-10                  1.2                 False           1.2
            2022-01-11                  1.3                 False           1.3
            2022-01-12                  1.2                 False           1.2
            2022-01-13                  1.1                 False           1.1
            2022-01-14                  1.0                 False           1.0
            2022-01-15                  1.0                 False           1.0
            2022-01-16                  10.1                True            10.1
            2022-01-17                  10.2                True            10.2
            2022-01-18                  1.3                 False           1.3
            ----------------------------------------------------------------------------------
        """
        df_data[feature_name] = df_data[COLUMN_NAME_VALUE]
        return df_data

    @staticmethod
    def _feature_generation_label(
            df_data: DataFrame,
            feature_name: str = LABEL_NAME_ANOMALY_LABEL) -> DataFrame:
        """
        Generate the feature "current_value", we will generate a new column besides the input dataframe named COLUMN_NAME_FEATURES.

        :param df_data: input dataframe with the following schema with examples:
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
        :return: the updated dataframe
            ----------------------------------------------------------------------------------
            `DATE`                      `VALUE`             `ANOMALY`       `anomaly_label`
            ----------------------------------------------------------------------------------
            2022-01-01                  1.0                 False           False
            2022-01-02                  1.1                 False           False
            2022-01-03                  1.2                 False           False
            2022-01-04                  1.3                 False           False
            2022-01-05                  1.2                 False           False
            2022-01-06                  1.1                 False           False
            2022-01-07                  1.0                 False           False
            2022-01-08                  1.0                 False           False
            2022-01-09                  1.1                 False           False
            2022-01-10                  1.2                 False           False
            2022-01-11                  1.3                 False           False
            2022-01-12                  1.2                 False           False
            2022-01-13                  1.1                 False           False
            2022-01-14                  1.0                 False           False
            2022-01-15                  1.0                 False           False
            2022-01-16                  10.1                True            True
            2022-01-17                  10.2                True            True
            2022-01-18                  1.3                 False           False
            ----------------------------------------------------------------------------------
        """
        df_data[feature_name] = df_data[COLUMN_NAME_ANOMALY].replace({True: 1, False: 0})

        # sanity checking
        if not set(df_data[feature_name].to_numpy().tolist()).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"The anomaly_label feature is not a subset of {0, 1}, which is problematic. The actual value is {df_data[feature_name]}")

        return df_data

    @staticmethod
    def _feature_generation_historical_values(
            df_data: DataFrame,
            feature_name: str = FEATURE_NAME_HISTORICAL_VALUE_TENSOR,
            historical_length: int = 28) -> DataFrame:
        """
        Generate the feature "historical_values", we will generate a two column besides the input dataframe named COLUMN_NAME_FEATURES and
        COLUMN_NAME_FEATURE_VALUES.

        :param historical_length: the historical length for training.
        :param df_data: input dataframe with the following schema with examples:
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
        :return: the updated dataframe
            ----------------------------------------------------------------------------------
            `DATE`                      `VALUE`             `ANOMALY`       `historical_value_tensor`
            ----------------------------------------------------------------------------------
            2022-01-01                  1.0                 False           np.nan
            2022-01-02                  1.1                 False           np.nan
            2022-01-03                  1.2                 False           np.nan
            2022-01-04                  1.3                 False           np.nan
            2022-01-05                  1.2                 False           np.nan
            2022-01-06                  1.1                 False           np.nan
            2022-01-07                  1.0                 False           np.nan
            2022-01-08                  1.0                 False           np.nan
            2022-01-09                  1.1                 False           np.nan
            2022-01-10                  1.2                 False           np.nan
            2022-01-11                  1.3                 False           np.nan
            2022-01-12                  1.2                 False           np.nan
            2022-01-13                  1.1                 False           np.nan
            2022-01-14                  1.0                 False           np.nan
            2022-01-15                  1.0                 False           [1.0, 1.1, 1.2, ..., 1.0]
            2022-01-16                  10.1                True            [1.1, 1.2, 1.3, ..., 1.0]
            2022-01-17                  10.2                True            [1.2, 1.3, 1.2, ..., 10.1]
            2022-01-18                  1.3                 False           [1.3, 1.2, 1.1, ..., 10.2]
            ----------------------------------------------------------------------------------
        """

        # for temporal related feature, sort the dataframe first.
        df_data.sort_values(by=[COLUMN_NAME_DATE], inplace=True)

        # new feature value column `COLUMN_NAME_FEATURE_VALUES`
        df_data[feature_name] = [[] for _ in range(df_data.shape[0])]
        for i in range(df_data.shape[0]):
            date = df_data[COLUMN_NAME_DATE].iloc[i]
            historical_df = copy.deepcopy(
                df_data[(df_data[COLUMN_NAME_DATE].between(date - Timedelta(days=historical_length), date - Timedelta(days=1)))])
            # sanity checking, make sure the historical time series length equals to the expected `historical_length`.
            if historical_df.shape[0] != historical_length:
                df_data.at[i, COLUMN_NAME_IS_VALID_FEATURE] = False
                continue
            if historical_df[[COLUMN_NAME_DATE]].drop_duplicates().shape[0] != historical_length:
                df_data.at[i, COLUMN_NAME_IS_VALID_FEATURE] = False
                continue
            # generate historical value sliding window.
            historical_df.sort_values(by=[COLUMN_NAME_DATE], inplace=True)
            df_data.at[i, feature_name] = historical_df[COLUMN_NAME_VALUE].tolist()

        # if the feature generation is not successful, then we will use np.nan as the feature value.
        df_data[feature_name] = df_data[feature_name].apply(lambda x: np.nan if len(x) == 0 else x)

        return df_data

    @staticmethod
    def _day_of_week_one_hot_encoding(x: int) -> List[float]:
        """
        Generate the one hot encoding for the day of week feature.

        :param x: an integer between 0 and 6 (inclusive)
        :return: a one-hot-encoded 7-dimension day-of-week vector.
        """
        if not 0 <= x <= 6:
            raise ValueError(f"The x should be between 0 and 6 (inclusive), but actually get {x}")
        one_hot_encoding_vector: List[float] = [0.0] * 7
        one_hot_encoding_vector[x] = 1.0
        return one_hot_encoding_vector
