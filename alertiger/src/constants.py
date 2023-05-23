# Dataframe Column Names
# part 1: column names for the input time series dataframe.
COLUMN_NAME_DATE: str = "DATE"
COLUMN_NAME_VALUE: str = "VALUE"
COLUMN_NAME_ANOMALY: str = "ANOMALY"
# part 2: column names for the prediction output
COLUMN_NAME_PREDICTED_BASELINE: str = "BASELINE"
COLUMN_NAME_PREDICTED_LOWER_BOUND: str = "LOWER_BOUND"
COLUMN_NAME_PREDICTED_UPPER_BOUND: str = "UPPER_BOUND"
COLUMN_NAME_PREDICTED_ANOMALY_SCORE: str = "ANOMALY_SCORE"
COLUMN_NAME_PREDICTED_ANOMALY: str = "DETECTED_ANOMALY"
# part 3: intermediate name
COLUMN_NAME_IS_VALID_FEATURE: str = "IS_VALID_FEATURE"

# feature names (we will use it as the column name as well).
FEATURE_NAME_DAY_OF_WEEK_TENSOR: str = "day_of_week_tensor"
FEATURE_NAME_HISTORICAL_VALUE_TENSOR: str = "historical_value_tensor"
FEATURE_NAME_CURRENT_VALUE: str = "current_value"
FEATURE_NAME_MODEL_TRAFFIC_RATIO: str = "model_traffic_ratio"
LABEL_NAME_ANOMALY_LABEL: str = "anomaly_label"

# global parameters.
NORMALIZATION_EPSILON: float = 1e-5
ANOMALY_SCORE_THRESHOLD: float = 0.2
