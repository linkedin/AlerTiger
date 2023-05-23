import logging
import math
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras.layers import Dropout, Subtract
from tensorflow import Tensor
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.python.ops import math_ops

from .config import MlpLayerConfig


class AlerTigerForecastModelBuilder:
    """
    This is the AlerTiger's forecasting model component.

    ** The high-level model architecture is shown below: **
                            compute metric regularity score --> MLP layers ---------------------------------------+
                            |                                                                                     |
                            +----------------------------------------------------------------+                    |
                            |                                                                |                    |
        HISTORICAL_VALUE -> normalization -> MLP layers    -+               +-> UPPER_BOUND -> inv normalization -+                   +--> UPPER_BOUND
                                                            +-> MLP layers -+-> BASELINE    -> inv normalization -+-> amplification --+--> BASELINE
                                     SEASONALITY_INPUT -----+               +-> LOWER_BOUND -> inv normalization -+                   +--> LOWER_BOUND

    **Loss Function**
    The customized loss function include the following parts:
    (1) mean loss (RMSE), (2) quantile loss (pin-ball) (3) weight regularization

    This model has the following major improvements based on a simple `MlpModel`
        1. Supervised learning (not in this file)
        2. Datapoint Normalization
        3. Skipping Recent Historical Value
        4. Regularity Score
    """

    def __init__(
            self,
            mlp_config_for_historical_values: List[MlpLayerConfig] = [MlpLayerConfig(units=8, activation="linear")],
            mlp_config_for_seasonality_score: List[MlpLayerConfig] = [],
            mlp_config_for_concatenation: List[MlpLayerConfig] = [MlpLayerConfig(units=8, activation="linear")],
            mlp_config_for_baseline: List[MlpLayerConfig] = [MlpLayerConfig(units=4, activation="linear")],
            mlp_config_for_lower_bound: List[MlpLayerConfig] = [MlpLayerConfig(units=4, activation="linear")],
            mlp_config_for_upper_bound: List[MlpLayerConfig] = [MlpLayerConfig(units=4, activation="linear")],
            prediction_range: Tuple[float, float] = (0.005, 0.995),
            num_recent_skip_days: int = 3) -> None:

        """
        Builds a Multilayer Perceptron (MLP) model (a class of feedforward artificial neural network).
        Users could define the number of layers and dimensions in each layer. This model assumes all training data comes from one time series, so no metric id
        feature is needed for scaling.

        :param mlp_config_for_historical_values: the MLP layer configurations for the input feature `HISTORICAL_VALUE`
        :param mlp_config_for_concatenation: the MLP layer configuration for the concatenated vector between HISTORICAL_VALUE and SEASONALITY_INPUT
        :param mlp_config_for_baseline: the MLP layer configuration for the output header BASELINE (i.e. the predicted time series value)
        :param mlp_config_for_lower_bound: the MLP layer configuration for the output header LOWER_BOUND
        :param mlp_config_for_upper_bound: the MLP layer configuration for the output header UPPER_BOUND
        :param mlp_config_for_seasonality_score: the MLP layer configuration for the seasonality score (which compute how regular a time series is).
        :param prediction_range: the lower and upper percentile that we want the forecast model predict.
        :param num_recent_skip_days: the number of most recent days that we will skip from the `HISTORICAL_VALUE` input, the purpose is to prevent the model
          from overfitting to the most recent days thus oversensitive. By default, we use 3 days.
        """
        self.mlp_config_for_historical_values: List[MlpLayerConfig] = mlp_config_for_historical_values
        self.mlp_config_for_concatenation_layers: List[MlpLayerConfig] = mlp_config_for_concatenation
        self.mlp_config_for_seasonality_score: List[MlpLayerConfig] = mlp_config_for_seasonality_score
        self.mlp_config_for_baseline: List[MlpLayerConfig] = mlp_config_for_baseline
        self.mlp_config_for_lower_bound: List[MlpLayerConfig] = mlp_config_for_lower_bound
        self.mlp_config_for_upper_bound: List[MlpLayerConfig] = mlp_config_for_upper_bound
        self.prediction_range: Tuple[float, float] = prediction_range
        self.num_recent_skip_days: int = num_recent_skip_days
        self.loss = {"OUTPUT": lambda y, f: self._loss_function(y, f)}

    def build(
            self,
            historical_value_dim: int = 28,
            seasonality_input_dim: int = 7) -> Model:
        """
        Build the model graph and return the Keras Model object.

        :param seasonality_input_dim: the vector dimension for `HISTORICAL_VALUE` input tensor
        :param historical_value_dim: the vector dimension for `SEASONALITY_INPUT` input tensor
        :return: Keras model graph
        """

        # Model input that represent the historical value of the time series.
        historical_value_input: Tensor = Input(shape=(historical_value_dim,), name="INPUT_NAME_HISTORICAL_VALUE")

        # Model input that represent the seasonality of the time series.
        seasonality_inputs: Tensor = Input(shape=(seasonality_input_dim,), name="INPUT_NAME_SEASONALITY")

        # Improvement-1: Skip recent K=3 days for robustness and reduce sensitivity.
        historical_value_skip_dim: int = historical_value_dim - self.num_recent_skip_days
        historical_value_skip: Tensor = Lambda(lambda x: x[:, 0:historical_value_skip_dim], name="VALUE_INPUT_SKIP")(historical_value_input)

        # Improvement-2: Datapoint-wise normalization for generalizability.
        tsr_norm_std_mean: Tensor = Lambda(AlerTigerForecastModelBuilder._norm, name="VALUE_NORM_STD_MEAN")(historical_value_skip)
        historical_value_skip_normalized: Tensor = Lambda(lambda x: x[:, 0: historical_value_skip_dim], output_shape=(historical_value_skip_dim,),
                                                          name="HISTORICAL_VALUE_SKIP_NORM")(tsr_norm_std_mean)
        historical_value_skip_std: Tensor = Lambda(lambda x: x[:, historical_value_skip_dim:historical_value_skip_dim + 1], output_shape=(1,),
                                                   name="HISTORICAL_VALUE_SKIP_STD")(
            tsr_norm_std_mean)
        historical_value_skip_mean: Tensor = Lambda(lambda x: x[:, historical_value_skip_dim + 1:historical_value_skip_dim + 2], output_shape=(1,),
                                                    name="HISTORICAL_VALUE_SKIP_MEAN")(tsr_norm_std_mean)

        # Step-1: MLP on VALUE_INPUT
        historical_value_skip_normalized = AlerTigerForecastModelBuilder._apply_mlp_layers(historical_value_skip_normalized,
                                                                                           self.mlp_config_for_historical_values)

        # Step 2. concat with seasonality inputs, build the hidden layers by updating "concatenated"
        concatenated = Concatenate(name="CONCATENATION_HISTORICAL_VALUE_AND_SEASONALITY")([historical_value_skip_normalized, seasonality_inputs])
        concatenated = AlerTigerForecastModelBuilder._apply_mlp_layers(concatenated, self.mlp_config_for_concatenation_layers)

        # STEP 3. build the hidden layers for 3 outputs
        # predict mean
        dense_output = AlerTigerForecastModelBuilder._apply_mlp_layers(concatenated, self.mlp_config_for_baseline)
        y_mean_scaled = Dense(1, name="Y_MEAN_SCALED")(dense_output)
        # predict upper bound
        dense_output_upper = AlerTigerForecastModelBuilder._apply_mlp_layers(concatenated, self.mlp_config_for_upper_bound)
        y_upper_scaled = Dense(1, name="Y_UPPER_SCALED")(dense_output_upper)
        # predict lower bound
        dense_output_lower = AlerTigerForecastModelBuilder._apply_mlp_layers(concatenated, self.mlp_config_for_lower_bound)
        y_lower_scaled = Dense(1, name="Y_LOWER_SCALED")(dense_output_lower)

        # Improvement-3: Compute regularity score and use it to amplify the boundary
        regularity_score: Tensor = Lambda(AlerTigerForecastModelBuilder._compute_regularity_score, name="REGULARITY_SCORE")(historical_value_skip)
        regularity_score = AlerTigerForecastModelBuilder._apply_mlp_layers(regularity_score, self.mlp_config_for_seasonality_score)
        regularity_score = Dense(1, activation="sigmoid", name="TRANSFORMED_REGULARITY_SCORE")(regularity_score)
        y_upper_scaled = Lambda(lambda x: tf.reshape(x[:, 0] + (x[:, 1] - x[:, 0]) * x[:, 2], [-1, 1]), output_shape=(1,), name="Y_UPPER_SCALED_AMPLIFIED")(
            Concatenate(axis=1)([y_mean_scaled, y_upper_scaled, regularity_score]))
        y_lower_scaled = Lambda(lambda x: tf.reshape(x[:, 0] + (x[:, 1] - x[:, 0]) * x[:, 2], [-1, 1]), output_shape=(1,), name="Y_LOWER_SCALED_AMPLIFIED")(
            Concatenate(axis=1)([y_mean_scaled, y_lower_scaled, regularity_score]))

        # STEP 4. Inverse datapoint-wise normalization.
        y_mean = Lambda(AlerTigerForecastModelBuilder._inv_norm, output_shape=(1,), name="Y_MEAN")(
            Concatenate(axis=1)([y_mean_scaled, historical_value_skip_std, historical_value_skip_mean]))
        y_upper = Lambda(AlerTigerForecastModelBuilder._inv_norm, output_shape=(1,), name="Y_UPPER")(
            Concatenate(axis=1)([y_upper_scaled, historical_value_skip_std, historical_value_skip_mean]))
        y_lower = Lambda(AlerTigerForecastModelBuilder._inv_norm, output_shape=(1,), name="Y_LOWER")(
            Concatenate(axis=1)([y_lower_scaled, historical_value_skip_std, historical_value_skip_mean]))

        # STEP 5. Build keras model with inputs and outputs
        y_output = Concatenate(axis=1, name="OUTPUT")([y_mean, y_lower, y_upper, historical_value_skip_mean, historical_value_skip_std])
        return Model([historical_value_input, seasonality_inputs], y_output)

    def _loss_function(
            self,
            y_target: np.array,
            y_predict: Tensor) -> Tensor:
        """
        A loss function that combines mean and upper/lower quantiles.

        :param y_target: Target
        :param y_predict: Predicted targets
        :return: the loss Tensor.
        """

        y_value = math_ops.cast(y_target[:, 0], y_predict.dtype)
        y_anomaly = math_ops.cast(y_target[:, 1], y_predict.dtype)  # anomaly label

        y_target_scaled = y_value
        f_mean_scaled = y_predict[:, 0]
        f_lower_scaled = y_predict[:, 1]
        f_upper_scaled = y_predict[:, 2]

        q_lower = self.prediction_range[0]
        q_upper = self.prediction_range[1]
        e_lower = (y_target_scaled - f_lower_scaled)
        e_upper = (y_target_scaled - f_upper_scaled)

        # put weight 0 to the anomalous data
        loss_mean = tf.keras.backend.mean(math_ops.multiply(math_ops.squared_difference(f_mean_scaled, y_target_scaled), (1 - y_anomaly)), axis=-1)
        loss_lower = tf.keras.backend.mean(math_ops.multiply(tf.keras.backend.maximum(q_lower * e_lower, (q_lower - 1) * e_lower), (1 - y_anomaly)), axis=-1)
        loss_upper = tf.keras.backend.mean(math_ops.multiply(tf.keras.backend.maximum(q_upper * e_upper, (q_upper - 1) * e_upper), (1 - y_anomaly)), axis=-1)

        loss = loss_mean + loss_lower + loss_upper
        return loss

    @staticmethod
    def _norm(tsr):
        """
        Utility function for datapoint-wise normalization.

        :param tsr: 2D tensor object
        :return:    concatenation of
                    - normalized tensor (2D of shape n_sample x dim)
                    - standard deviation tensor (2D of shape n_sample x 1)
                    - mean tensor (2D of shape n_sample x 1)
        """
        EPS: float = 0.01
        tsr_std: Tensor = tf.math.reduce_std(tsr, axis=1, keepdims=True)
        tsr_std = tf.where(tf.abs(tsr_std) < EPS, tf.ones_like(tsr_std) * EPS, tsr_std)
        tsr_mean: Tensor = tf.math.reduce_mean(tsr, axis=1, keepdims=True)
        tsr_normalized = (tsr - tsr_mean) / tsr_std
        return tf.concat([tsr_normalized, tsr_std, tsr_mean], axis=1)

    @staticmethod
    def _inv_norm(tsr_norm_std_mean: Tensor):
        """
        Utility function for inverse datapoint-wise normalization.

        :param: concatenation of
                - normalized tensor (2D of shape n_sample x dim)
                - standard deviation tensor (2D of shape n_sample x 1)
                - mean tensor (2D of shape n_sample x 1)
        :return: inverse normalized tensor
        """

        if tsr_norm_std_mean.shape[1] <= 2:
            raise ValueError(f"tsr_norm_std_mean.shape[1] == {tsr_norm_std_mean.shape[1]}, which we expect be greater than 2.")

        tsr_norm = Lambda(lambda x: x[:, :-2], output_shape=(1,))(tsr_norm_std_mean)
        tsr_std = Lambda(lambda x: x[:, -2:-1], output_shape=(1,))(tsr_norm_std_mean)
        tsr_mean = Lambda(lambda x: x[:, -1:], output_shape=(1,))(tsr_norm_std_mean)
        tsr = tsr_norm * tsr_std + tsr_mean
        return tsr

    @staticmethod
    def _init_regularizer(
            regularization_method: str,
            l1_coeff: float,
            l2_coeff: float) -> tf.keras.regularizers.Regularizer:
        """
        Helper function to initialize model weight regularizer

        :param regularization_method: the regularization method that used. Now we support the following regularization methods:
            - l1: tf.keras.regularizers.l1
            - l2: tf.keras.regularizers.l2
            - l1_l2: tf.keras.regularizers.l1_l2
        :param l1_coeff: the l1 regularization coefficient for the regularizer
        :param l2_coeff: the l2 regularization coefficient for the regularizer.
        """
        if regularization_method == "l1_l2":
            return tf.keras.regularizers.l1_l2(l1=l1_coeff, l2=l2_coeff)
        elif regularization_method == "l1":
            return tf.keras.regularizers.l1(l1_coeff)
        elif regularization_method == "l2":
            return tf.keras.regularizers.l2(l2_coeff)
        else:
            raise Exception(f"The regularizer {regularization_method} is not supported! only support l1, l2, l1_l2")

    @staticmethod
    def _apply_mlp_layers(
            input_tensor: Tensor,
            mlp_layer_configs: List[MlpLayerConfig]) -> Tensor:
        """
        Apply mlp layers to the input tensor

        :param input_tensor: input tensor for mlp layers
        :param mlp_layer_configs: configurations for mlp layers, each layer corresponds to one `MlpLayerConfig` object
        :return: resulting tensor
        """
        for layer_config in mlp_layer_configs:
            input_tensor = Dense(layer_config.units,
                                 kernel_regularizer=AlerTigerForecastModelBuilder._init_regularizer(layer_config.kernel_regularizer_method,
                                                                                                    layer_config.kernel_regularizer_l1_coeff,
                                                                                                    layer_config.kernel_regularizer_l2_coeff),
                                 bias_regularizer=AlerTigerForecastModelBuilder._init_regularizer(layer_config.bias_regularizer_method,
                                                                                                  layer_config.bias_regularizer_l1_coeff,
                                                                                                  layer_config.bias_regularizer_l2_coeff),
                                 activation=layer_config.activation)(input_tensor)
            if layer_config.dropout_rate > 0.0:
                input_tensor = Dropout(layer_config.dropout_rate)(input_tensor)
        return input_tensor

    @staticmethod
    def _compute_regularity_score(value_input: Tensor) -> Tensor:
        """
        Utility function for computing regularity score for time series segment.

        :param value_input: historical value input matrix of shape (m, n)
        :return: regularity score vector of shape (m, 1)
        """
        periodicity: int = 7
        quantile: Tuple[float, float] = (0.4, 0.6)

        # normalize
        hist_tsr_norm: Tensor = Lambda(AlerTigerForecastModelBuilder._norm)(value_input)[:, :value_input.shape[1]]

        # compute week-over-one-week difference
        diff_tsr: Tensor = tf.concat(
            [hist_tsr_norm[:, i + periodicity:i + periodicity + 1] - hist_tsr_norm[:, i:i + 1] for i in range(value_input.shape[1] - periodicity)], 1)

        # compute quantile of week-over-one-week difference
        diff_tsr_sorted: Tensor = tf.sort(diff_tsr, axis=1, direction="ASCENDING", name=None)
        diff_tsr_sorted = tf.reshape(diff_tsr_sorted, [-1, diff_tsr.shape[1]])
        low_quantile_idx: float = 1.0 * (int(value_input.shape[1]) - periodicity - 1) * quantile[0]
        up_quantile_idx: float = 1.0 * (int(value_input.shape[1]) - periodicity - 1) * quantile[1]
        if low_quantile_idx == float(int(low_quantile_idx)):
            low_quantile: Tensor = diff_tsr_sorted[:, int(low_quantile_idx)]
        else:
            low_quantile: Tensor = diff_tsr_sorted[:, math.floor(low_quantile_idx)] * (math.ceil(low_quantile_idx) - low_quantile_idx) + \
                                   diff_tsr_sorted[:, math.ceil(low_quantile_idx)] * (low_quantile_idx - math.floor(low_quantile_idx))
        if up_quantile_idx == float(int(up_quantile_idx)):
            up_quantile: Tensor = diff_tsr_sorted[:, int(up_quantile_idx)]
        else:
            up_quantile: Tensor = diff_tsr_sorted[:, math.floor(up_quantile_idx)] * (math.ceil(up_quantile_idx) - up_quantile_idx) + \
                                  diff_tsr_sorted[:, math.ceil(up_quantile_idx)] * (up_quantile_idx - math.floor(up_quantile_idx))

        # compute regularity score using quantile difference.
        regularity_score: Tensor = tf.reshape(up_quantile - low_quantile, [-1, 1])

        logging.debug("value_input.shape:", value_input.shape)
        logging.debug("low_quantile_idx:", low_quantile_idx)
        logging.debug("high_quantile_idx:", up_quantile_idx)

        return regularity_score


class AlerTigerClassificationModelBuilder:
    """
    A classification model of 2 stages. In the first stage, a prediction model for mean/upper/lower bounds/normalization mean/normalization std are built.
    In the second stage, classification model is built.

    This model is generally applicable to input any base models whose output contains BASELINE, LOWER_BOUND, UPPER_BOUND, NORMALIZATION_MEAN, NORMALIZATION_STD.
    The difference between `TwoStageGenericClassificationModel` and `TwoStageClassificationModel` is that the former one doesn't has metric_id tensor, and
    requires the baseline model output mean and std, therefore is applicable to unseen time series.

    **Model Structure**
                                     |-UPPER_BOUND  |
    Base_Model_inputs => BaseModel ->|-BASELINE     |
                                     |-LOWER_BOUND  | --> Classification Model.
                                     |-MEAN         |
                                     |-STD          |

    **Loss Function**
    if base_model.trainable = True:
    (1) mean loss (RMSE), (2) quantile loss (pin-ball), (3) classification loss
    otherwise:
    classification loss

    Attributes:
        required_output_names: names of the required 5 outputs (baseline, lower, upper, mean, std) are the output from the base model.
                               anomaly_score is the output from classification model
        required_target_names: required target current, anomaly
        base_model_module: The MlpSeasonalityAdaptiveGenericDailyModel that we use to learn baseline, upper and lower bounds， and input mean and std.
        base_model: the model learned from the base model module
    """

    def __init__(
            self,
            mlp_layers: List[MlpLayerConfig] = [MlpLayerConfig(units=14, activation="sigmoid"),
                                                MlpLayerConfig(units=9, activation="sigmoid"),
                                                MlpLayerConfig(units=9, activation="sigmoid")]) -> None:

        self.mlp_layers: List[MlpLayerConfig] = mlp_layers

        self.loss = {"FINAL_OUTPUT": lambda y, f: self._loss_function(y, f)}

    def _loss_function(
            self,
            y_target: np.array,
            y_predict: Tensor) -> Tensor:
        """
        Add to the base loss function, a classification loss.

        :param y_target: Target
        :param y_predict: Predicted targets
        :return: the loss Tensor.
        """

        anomaly_score = y_predict[:, 3]
        y_anomaly = y_target[:, 1]
        classification_loss = tf.keras.backend.binary_crossentropy(y_anomaly, anomaly_score, from_logits=False)
        return classification_loss

    def build(
            self,
            alertiger_forecast_model: Model) -> Model:
        """
        Build the AlerTiger Classification Model on top of the trained AlerTiger forecast model.

        :param alertiger_forecast_model: the AlerTiger forecast model that's already trained.
        :return: Keras model graph
        """

        # freeze the weights of the base model
        alertiger_forecast_model.trainable = False
        base_outputs = alertiger_forecast_model(alertiger_forecast_model.inputs, training=False)

        if base_outputs.shape[1] != 5:
            raise Exception("Shape not satisfied")

        # parse the base output
        y_mean: Tensor = Lambda(lambda x: x[:, 0:1], output_shape=(1,))(base_outputs)
        y_lower: Tensor = Lambda(lambda x: x[:, 1:2], output_shape=(1,))(base_outputs)
        y_upper: Tensor = Lambda(lambda x: x[:, 2:3], output_shape=(1,))(base_outputs)
        scale_mean: Tensor = Lambda(lambda x: x[:, 3:4], output_shape=(1,))(base_outputs)
        scale_std: Tensor = Lambda(lambda x: x[:, 4:5], output_shape=(1,))(base_outputs)

        # Model input that represent observed value for the date that we will make prediction on.
        # later we will calculate the difference between this observed value and the predicted time series value for anomaly detection.
        current_value = Input(shape=(1,), name="CURRENT")
        current_value_scaled = Lambda(AlerTigerClassificationModelBuilder._norm)((current_value, scale_std, scale_mean))

        # calculate diff
        y_mean_scaled = Lambda(lambda x: x[:, 0: 1], output_shape=(1,), name="Y_MEAN_NORM")(
            Lambda(AlerTigerClassificationModelBuilder._norm)((y_mean, scale_std, scale_mean)))
        y_lower_scaled = Lambda(lambda x: x[:, 0: 1], output_shape=(1,), name="Y_LOWER_NORM")(
            Lambda(AlerTigerClassificationModelBuilder._norm)((y_lower, scale_std, scale_mean)))
        y_upper_scaled = Lambda(lambda x: x[:, 0: 1], output_shape=(1,), name="Y_UPPER_NORM")(
            Lambda(AlerTigerClassificationModelBuilder._norm)((y_upper, scale_std, scale_mean)))

        y_mean_diff = Subtract()([current_value_scaled, y_mean_scaled])
        y_lower_diff = Lambda(lambda x: tf.keras.backend.maximum(0.0, x[1] - x[0]))([current_value_scaled, y_lower_scaled])
        y_upper_diff = Lambda(lambda x: tf.keras.backend.maximum(0.0, x[0] - x[1]))([current_value_scaled, y_upper_scaled])

        # concatenate info
        dense_anomaly = Concatenate(axis=1, name="CONCAT_FOR_CLASSIFICATION")([y_mean_diff, y_upper_diff, y_lower_diff, alertiger_forecast_model.inputs[1]])

        for layer in self.mlp_layers:
            dense_anomaly = Dense(layer.units, activation=layer.activation)(dense_anomaly)

        anomaly_score = Dense(1, activation="sigmoid", name="ANOMALY_SCORE")(dense_anomaly)

        # concatenate with value outputs
        y_output = Concatenate(axis=1, name="FINAL_OUTPUT")([y_mean, y_lower, y_upper, anomaly_score])
        model: Model = Model(alertiger_forecast_model.inputs + [current_value], y_output)

        # model visualization.
        model.summary()

        return model

    @staticmethod
    def _norm(tsr_orig_std_mean: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Use the provided mean and standard deviation (calculated from the forecaster) to normalize the current observed value.

        :param tsr_orig_std_mean: a tuple of the original tensor, standard deviation value tensor, and the mean value tensor.
        :return: the normalized tensor using the provided standard deviation and the mean value.
        """
        tsr_original: tf.Tensor = tsr_orig_std_mean[0]
        tsr_std: tf.Tensor = tsr_orig_std_mean[1]
        tsr_mean: tf.Tensor = tsr_orig_std_mean[2]
        tsr: tf.Tensor = (tsr_original - tsr_mean) / tsr_std
        return tsr
