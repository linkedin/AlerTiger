import datetime
from typing import List

import pandas as pd

from alertiger.src.inferencer import inference_alertiger_model
from alertiger.src.trainer import train_alertiger_model
from alertiger.src.utils import mock_univariate_time_series_with_anomaly, visualize_time_series_with_anomaly


def demo():
    """
    This demo function is exactly the same fucntion as alertiger_demo.ipynb, but wrapped in py file instead of ipynb notebook file.
    """
    training_univariate_timeseries: List[pd.DataFrame] = mock_univariate_time_series_with_anomaly(start_date=datetime.date(2022, 1, 1),
                                                                                                  end_date=datetime.date(2023, 1, 1),
                                                                                                  seasonality_number_of_square_timeseries=10,
                                                                                                  seasonality_number_of_triangle_timeseries=10,
                                                                                                  seasonality_number_of_sine_timeseries=10,
                                                                                                  seasonality_number_of_constant_timeseries=10,
                                                                                                  )

    testing_univariate_timeseries: List[pd.DataFrame] = mock_univariate_time_series_with_anomaly(start_date=datetime.date(2023, 1, 1),
                                                                                                 end_date=datetime.date(2023, 4, 1),
                                                                                                 seasonality_number_of_square_timeseries=1,
                                                                                                 seasonality_number_of_triangle_timeseries=1,
                                                                                                 seasonality_number_of_sine_timeseries=1,
                                                                                                 seasonality_number_of_constant_timeseries=1)

    alertiger_model, history_forecast, history_classification = train_alertiger_model(training_univariate_timeseries,
                                                                                      epoch=50,
                                                                                      start_date=datetime.date(2022, 1, 1),
                                                                                      end_date=datetime.date(2023, 1, 1))

    prediction_result_dataframes: List[pd.DataFrame] = inference_alertiger_model(testing_univariate_timeseries, alertiger_model)

    for result_df in prediction_result_dataframes:
        visualize_time_series_with_anomaly(result_df)


if __name__ == '__main__':
    demo()
