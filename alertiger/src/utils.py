import copy
import datetime
import math
import random
from typing import List, Set, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Timestamp, date_range

from .constants import COLUMN_NAME_ANOMALY, COLUMN_NAME_PREDICTED_ANOMALY_SCORE, COLUMN_NAME_PREDICTED_LOWER_BOUND, COLUMN_NAME_DATE, \
    COLUMN_NAME_PREDICTED_BASELINE, COLUMN_NAME_PREDICTED_UPPER_BOUND, COLUMN_NAME_VALUE, COLUMN_NAME_PREDICTED_ANOMALY, NORMALIZATION_EPSILON

EPSILON: float = 1e-2


def mock_univariate_time_series_with_anomaly(
        start_date: datetime.date = datetime.date(2022, 1, 1),
        end_date: datetime.date = datetime.date(2023, 1, 1),

        # trend component configuration
        trend_slope_range_min: float = 0.0,
        trend_slope_range_max: float = 0.001,
        trend_bias_range_min: float = -5.0,
        trend_bias_range_max: float = 5.0,

        # seasonality configurations
        seasonality_frequency: int = 7,
        seasonality_number_of_square_timeseries: int = 50,
        seasonality_number_of_triangle_timeseries: int = 50,
        seasonality_number_of_sine_timeseries: int = 50,
        seasonality_number_of_constant_timeseries: int = 50,

        # noisiness configuration
        noisiness_scale_in_std: float = 0.05,

        # anomaly injection configuration (each time series will only inject at most one anomaly duration for simplicity)
        anomaly_type_spike_ratio: float = 0.4,
        anomaly_type_level_shift_ratio: float = 0.4,
        anomaly_type_no_anomaly_ratio: float = 0.2,

        # anomaly duration ratio.
        anomaly_duration_min: int = 2,
        anomaly_duration_max: int = 10,
        anomaly_start_min_index: int = 28,

        # anomaly level requirement
        anomaly_level_min: float = 0.5,
        anomaly_level_max: float = 1.0,
        min_anomaly_change: float = 0.1,

        # random seed for all random components.
        random_seed: int = 1,
        random_phase: bool = False
):
    """
    Random time series generator with various types of time series types (triangle, square, sine, constant) and noise types (level-shift, spike).
    We model the time series as addition of three terms: trend, seasonality, and noise. i.e., y(t) = trend(t) + seasonality(t) + noise(t).
    where:
        trend(t) := slope * t + b
        seasonality(t) := triangle_shape(t) OR square_shape(t) OR sine_shape(t) OR constant(t)
        noise(t) = Gaussian_noise(t)

    :param start_date: the starting date of time series
    :param end_date: the ending date of time series

    :param trend_slope_range_min: the trand component's slop range's min
    :param trend_slope_range_max: the trend component's slop range's max

    :param seasonality_frequency: the seasonality term's frequency. (for the case of non-seasonality time series, the constant seasonality type will handle it).
    :param seasonality_number_of_square_timeseries: number of seasonal square timeseries
    :param seasonality_number_of_triangle_timeseries: number of seasonal triangle timeseries
    :param seasonality_number_of_sine_timeseries: number of seasonal sine timeseries
    :param seasonality_number_of_constant_timeseries: number of seasonal constant timeseries

    :param noisiness_scale_in_std: the randomness scale in the time series, since the time series is within the random of 0 and 1 (before trend added), there
    this std can be considered as a relative randomness scale compared to the time series range.

    :param anomaly_type_spike_ratio: the ratio of spike anomaly, should be within range [0, 1]
    :param anomaly_type_level_shift_ratio: the ratio of level-shift anomaly, should be within range [0, 1]
    :param anomaly_type_no_anomaly_ratio: the ratio of no anomaly injected, should be within range [0, 1], the summation between all ratios should be 1.0

    :param anomaly_duration_min: the minimum duration of mock anomaly injected
    :param anomaly_duration_max: the maximum duration of mock anomaly injected
    :param anomaly_start_min_index: the minimum index within a time series that we will inject anomaly into.

    :param anomaly_level_min: the min anomaly severity that we will inject into time series, this paramater is valid for both level and spike anomaly, where for
     spike anomaly this is the highlest point's deviation from the normal range, for the level-shift anomaly this is the new level's deviation from the normal
     range.
    :param anomaly_level_max:  similar to the anomaly_level_min, but defining the upper bound of the anomaly severity.
    :param min_anomaly_change: this is a final check on the anomaly's deviation before returning the result. we will retry the inject if the anomaly injected's
      average deviation from the original time series didn't satisfy this minimum threshold.

    :param random_seed: this is the seed for all the randomness in this mock dataset generation.
    :param random_phase: a boolean switch that control whether we want the time series has random phase (e.g. the peak can occur on any day of week) vs. fixed


    :return: a list of dataframe, where each dataframe represent a single univariate time series with anomaly injected. the schema of the dataframe with example
        is shown here:
        +----------------+----------------+-------------------+
        | date           |  value         |     anomaly       |
        +----------------+----------------+-------------------+
        | 2023-01-01     |  0.1           |     False         |
        | 2023-01-02     |  0.1           |     False         |
        | 2023-01-03     |  3.0           |     True          |
        +----------------+----------------+-------------------+
    """

    random.seed(random_seed)
    np.random.seed(random_seed)

    # the length of the time series
    len_timeseries: int = (end_date - start_date).days
    time_index = np.linspace(0, len_timeseries - 1, num=len_timeseries)

    seasonality_types = ["constant"] * seasonality_number_of_constant_timeseries + ["triangle"] * seasonality_number_of_triangle_timeseries + \
                        ["square"] * seasonality_number_of_square_timeseries + ["sine"] * seasonality_number_of_sine_timeseries

    if not (anomaly_type_spike_ratio + anomaly_type_level_shift_ratio + anomaly_type_no_anomaly_ratio == 1.0 and 0 <= anomaly_type_spike_ratio <= 1.0 and
            0 <= anomaly_type_level_shift_ratio <= 1.0 and 0 <= anomaly_type_no_anomaly_ratio <= 1.0):
        raise ValueError("Please make sure to input valid value for anomaly_type_spike_ratio and anomaly_type_level_shift_ratio")
    total_number_of_timeseries = len(seasonality_types)

    anomaly_types: List[str] = random.choices(["spike", "level_shift", "no_anomaly"],
                                              weights=[int(100 * anomaly_type_spike_ratio),
                                                       int(100 * anomaly_type_level_shift_ratio),
                                                       int(100 * anomaly_type_no_anomaly_ratio)],
                                              k=total_number_of_timeseries)
    # time series result.
    result_timeseries: List[pd.DataFrame] = []
    for idx in range(total_number_of_timeseries):

        # Component 1: trend component of the time series.
        ts_trend: np.arary = time_index * random.uniform(trend_slope_range_min, trend_slope_range_max) + random.uniform(trend_bias_range_min,
                                                                                                                        trend_bias_range_max)

        # Component 2: seasonality component
        ts_seasonality = None
        seasonality_type = seasonality_types[idx]
        if seasonality_type == "sine":
            # Define amplitude, frequency, and phase for sine wave
            amp_sin = 1
            freq_sin = 1.0 / seasonality_frequency
            phase_sin = 0 if not random_phase else random.uniform(0, 2 * np.pi)
            # Generate sine wave
            sin_wave = amp_sin * np.sin(2 * np.pi * freq_sin * time_index + phase_sin)
            ts_seasonality = sin_wave

        elif seasonality_type == "square":
            # Define amplitude, frequency, and phase for square wave
            amp_sq = 1
            freq_sq = 1.0 / seasonality_frequency
            phase_sq = 0 if not random_phase else random.uniform(0, 2 * np.pi)

            # Generate square wave
            sq_wave = amp_sq * np.where(np.sin(2 * np.pi * freq_sq * time_index + phase_sq) > EPSILON, 1, -1)
            ts_seasonality = sq_wave

        elif seasonality_type == "triangle":
            # Define amplitude, frequency, and phase for triangle wave
            amp_tri = 1
            freq_tri = 1.0 / seasonality_frequency
            phase_tri = 0 if not random_phase else random.uniform(0, 2 * np.pi)

            # create the triangle wave
            ts_seasonality = amp_tri * np.abs((2 * time_index * freq_tri + phase_tri) % 2 - 1)

        elif seasonality_type == "constant":
            # no seasonality
            ts_seasonality = time_index * 0.0

        else:
            raise ValueError("The seasonality type is not accepted.")

        # Component 3: The noise term.
        ts_noise = np.random.normal(0, noisiness_scale_in_std, len(time_index))

        # combine the three components to the final time series
        timeseries_value_normal = ts_noise + ts_seasonality + ts_trend

        # convert time to datetime.datetime
        date = pd.date_range(start=start_date, periods=len_timeseries)

        univariate_timeseries: pd.DataFrame = pd.DataFrame({
            COLUMN_NAME_DATE: date,
            COLUMN_NAME_VALUE: timeseries_value_normal,
            COLUMN_NAME_ANOMALY: [False] * len_timeseries
        })

        anomaly_type = anomaly_types[idx]

        # calculate the anomaly start and end index
        if anomaly_duration_max + anomaly_start_min_index > len_timeseries:
            raise ValueError(
                f"The requirement on the anomaly_duration_max = {anomaly_duration_max} + anomaly_start_min_index = {anomaly_start_min_index} can NOT be "
                f"satisfied given the length of time series = {len_timeseries}")
        anomaly_duration: int = random.randint(anomaly_duration_min, anomaly_duration_max)
        anomaly_start_index: int = random.randint(anomaly_start_min_index, len_timeseries - anomaly_duration)
        anomaly_end_index: int = anomaly_start_index + anomaly_duration - 1  # this is inclusive.

        if anomaly_type == "spike":
            anomaly_injection_function = _inject_spike_anomaly
        elif anomaly_type == "level_shift":
            anomaly_injection_function = _inject_level_shift_anomaly
        elif anomaly_type == "no_anomaly":
            result_timeseries.append(univariate_timeseries)
            continue
        else:
            raise ValueError(f"Get unexpected anomaly type {anomaly_type}")

        anomaly_injection_success = False
        while not anomaly_injection_success:
            anomaly_injection_success = anomaly_injection_function(df=univariate_timeseries,
                                                                   start_index=anomaly_start_index,
                                                                   end_index=anomaly_end_index,
                                                                   anomaly_level_min=anomaly_level_min,
                                                                   anomaly_level_max=anomaly_level_max,
                                                                   min_anomaly_change=min_anomaly_change)

        result_timeseries.append(univariate_timeseries)
    return result_timeseries


def _inject_spike_anomaly(
        df: pd.DataFrame,
        start_index: int,
        end_index: int,
        anomaly_level_min: float = 0.1,
        anomaly_level_max: float = 1.0,
        min_anomaly_change: float = 0.02) -> bool:
    """
    Inject anomaly into data frame df, starting at time step start_index, and return the end index of anomaly
    Injection is done in place (input data frame is modified).

    :param df: data frame containing the time series where an anomaly is to be injected
    :param start_index: starting row index where for the injected anomaly
    :param end_index: ending row index where for the injected anomaly
    :param anomaly_level_min: minimum allowable change in anomaly level shift
    :param anomaly_level_max: maximum allowable change in anomaly level shift
    :param min_anomaly_change: minimum change for the anomalous duration compared to baseline timeseries
    :return: the success of this injection. We will inject the anomaly per the configuration and do a final check with min_anomaly_change. If the
     "min_anomaly_change" checking failed, we will return False.
    """
    duration = end_index - start_index + 1

    spike_level = np.random.uniform(anomaly_level_min, anomaly_level_max)
    elevations_first_half = list(np.linspace(0, spike_level, (duration + 1) // 2 + 1)[1:])
    elevations_second_half = elevations_first_half[::-1]
    if duration % 2 == 1:
        elevations_second_half = elevations_second_half[1:]
    elevations = np.array(elevations_first_half + elevations_second_half)
    time_series_col_index = list(df).index(COLUMN_NAME_VALUE)
    if COLUMN_NAME_ANOMALY not in df.columns:
        df.columns["anomaly"] = False
    anomaly_col_index = list(df).index(COLUMN_NAME_ANOMALY)
    anomaly_values = df.iloc[start_index:end_index + 1, time_series_col_index] + elevations
    base_average = np.mean(df[COLUMN_NAME_VALUE][start_index:end_index + 1])
    anomaly_average = np.mean(anomaly_values)

    if not np.isclose(base_average, 0):
        if abs(anomaly_average - base_average) / abs(base_average) <= min_anomaly_change:
            return False
    elif np.isclose(anomaly_average, 0):
        return False
    # update the value in place.
    df.iloc[start_index:end_index + 1, time_series_col_index] = anomaly_values
    # update the anomaly label column
    df.iloc[start_index:end_index + 1, anomaly_col_index] = True
    # it's a success injection.
    return True


def _inject_level_shift_anomaly(
        df: pd.DataFrame,
        start_index: int,
        end_index: int,
        anomaly_level_min: float = 0.1,
        anomaly_level_max: float = 1.0,
        min_anomaly_change: float = 0.02) -> bool:
    """
    Inject anomaly into data frame df, starting at time step start_index, and return the end index of anomaly
    Injection is done in place (input data frame is modified).

    :param df: data frame containing the time series where an anomaly is to be injected
    :param start_index: starting row index where for the injected anomaly
    :param end_index: ending row index where for the injected anomaly
    :param anomaly_level_min: minimum allowable change in anomaly level shift
    :param anomaly_level_max: maximum allowable change in anomaly level shift
    :param min_anomaly_change: minimum change for the anomalous duration compared to baseline timeseries
    :return: the success of this injection. We will inject the anomaly per the configuration and do a final check with min_anomaly_change. If the
     "min_anomaly_change" checking failed, we will return False.
    """
    shift_level = np.random.uniform(anomaly_level_min, anomaly_level_max)
    time_series_col_index = list(df).index(COLUMN_NAME_VALUE)
    if COLUMN_NAME_ANOMALY not in df.columns:
        df.columns[COLUMN_NAME_ANOMALY] = False
    anomaly_col_index = list(df).index(COLUMN_NAME_ANOMALY)
    anomaly_values = df.iloc[start_index:end_index + 1, time_series_col_index] + shift_level
    base_average = np.mean(df[COLUMN_NAME_VALUE][start_index:end_index + 1])
    # success checking.
    anomaly_average = np.mean(anomaly_values)
    if not np.isclose(base_average, 0):
        if abs(anomaly_average - base_average) / abs(base_average) <= min_anomaly_change:
            return False
    elif np.isclose(anomaly_average, 0):
        return False
    # update the value in place.
    df.iloc[start_index:end_index + 1, time_series_col_index] = anomaly_values
    # update the anomaly label column
    df.iloc[start_index:end_index + 1, anomaly_col_index] = True
    # it's a success injection.
    return True


def visualize_time_series_with_anomaly(df: pd.DataFrame):
    """
    Visualize the time series dataframe, we expect the following schema of the dataframe.
    The abnormal time series will be highlighted with red color.

    :param df: the time series for visualization.
    +----------------+----------------+-------------------+
    | date           |  value         |     anomaly       |
    +----------------+----------------+-------------------+
    | 2023-01-01     |  0.1           |     False         |
    | 2023-01-02     |  0.1           |     False         |
    | 2023-01-03     |  3.0           |     True          |
    +----------------+----------------+-------------------+
    :return: the time series plot with the abnormal region highlighted with red color.
    """

    fig, ax = plt.subplots(1, figsize=(10, 5))

    # plot the baseline times eries
    ax.plot(df[COLUMN_NAME_DATE], df[COLUMN_NAME_VALUE], "-", color="k", linewidth=2.0, label="observed value")

    # plot the prediction baseline.
    if COLUMN_NAME_PREDICTED_ANOMALY in df.columns:
        df_anomaly = df[df[COLUMN_NAME_PREDICTED_ANOMALY]]
        ax.plot(df_anomaly[COLUMN_NAME_DATE], df_anomaly[COLUMN_NAME_VALUE], ls="", marker="o", color="tab:red", label="predicted anomaly")
    elif COLUMN_NAME_ANOMALY in df.columns:
        df_anomaly = df[df[COLUMN_NAME_ANOMALY]]
        ax.plot(df_anomaly[COLUMN_NAME_DATE], df_anomaly[COLUMN_NAME_VALUE], ls="-", color="tab:red", label="label anomaly")

    # plot the prediction baseline.
    if COLUMN_NAME_PREDICTED_BASELINE in df.columns:
        ax.plot(df[COLUMN_NAME_DATE], df[COLUMN_NAME_PREDICTED_BASELINE], ls="--", color="tab:orange", label="predicted value")

    # plot the prediction region
    if COLUMN_NAME_PREDICTED_LOWER_BOUND in df.columns and COLUMN_NAME_PREDICTED_UPPER_BOUND in df.columns:
        ax.fill_between(df[COLUMN_NAME_DATE], df[COLUMN_NAME_PREDICTED_LOWER_BOUND], df[COLUMN_NAME_PREDICTED_UPPER_BOUND],
                        color="tab:blue", alpha=0.5, label="predicted boundary")

    # plot the prediction region
    if COLUMN_NAME_PREDICTED_ANOMALY_SCORE in df.columns:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(df[COLUMN_NAME_DATE], df[COLUMN_NAME_PREDICTED_ANOMALY_SCORE], color="tab:red", ls="-.",
                 label="anomaly score (right y-axis)")
        y_lower = ax2.get_ylim()[0]
        y_upper = ax2.get_ylim()[1]
        ax2.set_ylim(y_lower, y_upper + (y_upper - y_lower) * 4)
        ax2.legend(loc="lower left", bbox_to_anchor=(1.1, 1))
        ax2.set_ylabel("anomaly score")

    ax.legend(loc="lower left", bbox_to_anchor=(1.1, 0))

    # decorate the plot
    num_ticks = 15
    dts = df[COLUMN_NAME_DATE]
    min_dt, max_dt = min(dts), max(dts)
    major_dts = pd.date_range(
        start=min_dt, end=max_dt, periods=num_ticks
    ).map(lambda x: x.round("D"))
    minor_dts = pd.date_range(
        start=min_dt, end=max_dt, periods=(max_dt - min_dt).days + 1
    ).map(lambda x: x.round("D"))
    from matplotlib import dates as mdates
    dt_format = mdates.DateFormatter("%Y-%m-%d")

    # Set these x-axis configs for all charts
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_xticks(minor_dts, minor=True)
    ax.set_xticks(major_dts, minor=False)
    ax.set_xlim([min_dt, max_dt])
    ax.xaxis.set_major_formatter(dt_format)
    ax.grid(True, which="minor")
    ax.set_ylabel("time series (and prediction) values")


def normalize_timeseries(list_univariate_timeseries_data: List[DataFrame]) -> None:
    """
    Utility function that can normalize a set of univariate time series to 0 mean and 1 standard deviation.
    Note that the normalization is within each univariate time series.

    :param list_univariate_timeseries_data: list of univariate time series dataframes
    :return: None
    """
    # global normalization (note that this normalization will NOT introduce information leakage because within the tensorflow model we will have another
    # normalization that will override this normalization. This normalization is purely benefiting the training process by ensuring the loss function is
    # calculated at the same scale.
    for df_data in list_univariate_timeseries_data:
        mean: float = df_data[COLUMN_NAME_VALUE].mean()
        std: float = df_data[COLUMN_NAME_VALUE].std()
        df_data[COLUMN_NAME_VALUE] = (df_data[COLUMN_NAME_VALUE] - mean) / (std + NORMALIZATION_EPSILON)


def check_np_nan(x: Any) -> bool:
    """
    This function checks if the float or int type variable is nan

    :param x: any input value to be checked if is np.nan
    :return: a boolean result showing whether the input value is np.nan.
    """
    return type(x) in {float, int} and np.isnan(x)


def _calculate_relative_severity(
        current: float,
        baseline: float,
        upper_bound: float,
        lower_bound: float) -> float:
    """
    This function calculates severity based on the formula : |current - baseline|/(upper bound - lower bound)

    :param current: the current observed value
    :param baseline: the expected value
    :param upper_bound: the upper bound value
    :param lower_bound: the lower bound value
    :return: the relative severity score
    """

    FALLBACK_SEVERITY_VALUE: float = 0.0
    MAX_SEVERITY_VALUE: float = math.exp(5)
    MIN_SEVERITY_VALUE: float = 0.0

    if baseline == current:
        return MIN_SEVERITY_VALUE
    elif upper_bound == lower_bound:
        return MAX_SEVERITY_VALUE
    elif not check_np_nan(baseline) and baseline is not None and not check_np_nan(current) and current is not None \
            and not check_np_nan(upper_bound) and upper_bound is not None and not check_np_nan(lower_bound) and lower_bound is not None:
        return abs(baseline - current) / abs(upper_bound - lower_bound)
    return FALLBACK_SEVERITY_VALUE


SEVERITY_SCORE: str = "DSC_SEVERITY_SCORE"


def duration_severity_filter(
        df_data: DataFrame,
        duration_window_size: int = 3,
        duration_threshold: int = 2,
        severity_threshold: float = 1.3,
        combine_logic: str = "AND") -> DataFrame:
    """
    performance the duration and severity filtering on the detection result `df_data`. we only require two columns in df_data: `date` and `predicted_anomaly`.
    this function will remove those anomalies (i.e. rows with `predicted_anomaly == True`) that has low anomaly severities or last for too short (i.e. anomaly
    duration shorter than duration_threshold within the window of size `duration_window_size`).

    :param df_data a uni-variate time series with predicted anomaly result `predicted_anomaly`. The dataframe should have the following schema.
        - DATE: the date of the time series
        - VALUE: the observed time series value
        - BASELINE: the predicted baseline
        - LOWER_BOUND: the lower bound of the predicted confidence interval.
        - UPPER_BOUND: the upper bound of the predicted confidence interval.
        - DETECTED_ANOMALY: the point-wise detected anomaly for every single datapoint in the timeseries.
    :param duration_window_size the size of window that we will performance anomaly duration filtering. more specifically, an anomaly point will be kept only if
        there exists a window of size `duration_window_size` that has at least `duration_threshold` anomaly points within the window including this one.
    :param severity_threshold: the severity score filtering threshold where the severity is the absolute deviation normalized by the prediction boundary
        distance.
    :param duration_threshold: the duration threshold where we will only keep an anomaly point if there is a window of size `duration_window_size` that contains
        at least `duration_threshold` number of abnormal points.
    :param combine_logic: logical operator for duration and severity filter. This param can have "AND" or "OR" values.
    :return: The updated DataFrame with the column `DETECTED_ANOMALY` updated using the duration and severity filter.
    """
    # keep a copy to avoid changing the input dataframe.
    df_data = copy.deepcopy(df_data)

    input_columns: List[str] = list(df_data.columns)

    # temporary columns
    PASS_SEVERITY_FILTER: str = "PASS_SEVERITY_FILTER"
    PASS_DURATION_FILTER: str = "PASS_DURATION_FILTER"
    ANOMALY_INT_TYPE: str = "ANOMALY_INT_TYPE"

    # Severity Filter (S-Filter): For every single detection result, we calculate the severity score with
    # severity := abs(baseline - current) / abs(upper_bound - lower_bound), then compare this severity score with
    # threshold config.severity_threshold. If above threshold then keep this anomaly, otherwise change the anomaly to False.
    df_data[SEVERITY_SCORE] = df_data.apply(
        lambda x: _calculate_relative_severity(current=x[COLUMN_NAME_VALUE],
                                               baseline=x[COLUMN_NAME_PREDICTED_BASELINE],
                                               upper_bound=x[COLUMN_NAME_PREDICTED_UPPER_BOUND],
                                               lower_bound=x[COLUMN_NAME_PREDICTED_LOWER_BOUND]
                                               ), axis=1)
    df_data[PASS_SEVERITY_FILTER] = df_data.apply(
        lambda x: x[SEVERITY_SCORE] >= severity_threshold and x[COLUMN_NAME_PREDICTED_ANOMALY] if not check_np_nan(x[COLUMN_NAME_PREDICTED_ANOMALY]) and x[
            COLUMN_NAME_PREDICTED_ANOMALY] is not None else None, axis=1)

    # D Filter: We create a sliding window of size config.duration_window_size (which is 3 by default), and check the number of
    # abnormal points inside each window. If the number of anomaly points is above or equal to the threshold config.duration_threshold
    # then we will keep the anomaly in the result, otherwise will convert the anomaly to False.
    df_data = df_data.sort_values(by=[COLUMN_NAME_DATE])
    dates: List[Timestamp] = list(df_data.apply(lambda x: x[COLUMN_NAME_DATE], axis=1))
    df_data[ANOMALY_INT_TYPE] = df_data.apply(lambda x: 1 if x[COLUMN_NAME_PREDICTED_ANOMALY] is True else 0, axis=1)
    set_abnormal_date_after_D_filter: Set[Timestamp] = set()
    for sliding_window_start in date_range(min(dates), max(dates), freq=datetime.timedelta(days=1)):
        sliding_window_end = sliding_window_start + datetime.timedelta(days=1) * duration_window_size
        num_anomaly: int = sum(df_data[(df_data[COLUMN_NAME_DATE] >= sliding_window_start) &
                                       (df_data[COLUMN_NAME_DATE] < sliding_window_end)][ANOMALY_INT_TYPE])
        if num_anomaly >= duration_threshold:
            set_abnormal_date_after_D_filter.update(
                set(df_data[(df_data[COLUMN_NAME_DATE] >= sliding_window_start) &
                            (df_data[COLUMN_NAME_DATE] < sliding_window_end) &
                            (df_data[COLUMN_NAME_PREDICTED_ANOMALY])][COLUMN_NAME_DATE])

            )
    df_data[PASS_DURATION_FILTER] = df_data[COLUMN_NAME_PREDICTED_ANOMALY] & df_data[COLUMN_NAME_DATE].isin(set_abnormal_date_after_D_filter)

    # Combine the duration and severity filtering result
    if combine_logic == "AND":
        df_data[COLUMN_NAME_PREDICTED_ANOMALY] = df_data[PASS_SEVERITY_FILTER] & df_data[PASS_DURATION_FILTER]
    elif combine_logic == "OR":
        df_data[COLUMN_NAME_PREDICTED_ANOMALY] = df_data[PASS_SEVERITY_FILTER] | df_data[PASS_DURATION_FILTER]
    else:
        raise ValueError("the combine logic is not expected")
    return df_data[input_columns]
