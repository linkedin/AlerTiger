from unittest import TestCase

import numpy as np
import tensorflow as tf
from keras import Model

from alertiger.src.config import MlpLayerConfig
from alertiger.src.model import AlerTigerForecastModelBuilder, AlerTigerClassificationModelBuilder


class TestModel(TestCase):

    def setUp(self) -> None:
        self.forecast_model_builder = AlerTigerForecastModelBuilder(
            mlp_config_for_historical_values=[MlpLayerConfig(units=8, activation='linear')],
            mlp_config_for_seasonality_score=[],
            mlp_config_for_concatenation=[MlpLayerConfig(units=8, activation='linear')],
            mlp_config_for_baseline=[MlpLayerConfig(units=4, activation='linear')],
            mlp_config_for_lower_bound=[MlpLayerConfig(units=4, activation='linear')],
            mlp_config_for_upper_bound=[MlpLayerConfig(units=4, activation='linear')],
            prediction_range=(0.005, 0.995),
            num_recent_skip_days=3
        )

        self.classification_model_builder = AlerTigerClassificationModelBuilder(
            mlp_layers=[MlpLayerConfig(units=14, activation="sigmoid"),
                        MlpLayerConfig(units=9, activation="sigmoid"),
                        MlpLayerConfig(units=9, activation="sigmoid")]
        )

    def test_deep_eye_forecast_model_builder_build(self):
        # model structure check.
        forecast_model: Model = self.forecast_model_builder.build()
        self.assertEquals(29, len(forecast_model.layers))
        self.assertEquals(2, len(forecast_model.inputs))
        self.assertEquals(1, len(forecast_model.outputs))

    def test_deep_eye_forecast_model_builder_loss_function(self):
        # the first column is observed value, the second column is anomaly label.
        # in this example, we have 5 data points, 3 normal and 2 abnormal.
        y_target: np.array = np.array([[0.1, 0.1, 0.1, 1.0, 1.0], [0, 0, 0, 1, 1]]).T
        # the first column is predicted value, the second and third columns are lower and upper bounds.
        y_predict: np.array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [-0.5, -0.5, -0.5, -0.5, -0.5], [0.5, 0.5, 0.5, 0.5, 0.5]]).T

        loss = self.forecast_model_builder._loss_function(y_target, y_predict)

        # We expect the loss to be a combination of quantile loss and MSE loss.
        # MSE_Loss = MSE([0,0,0,0,0],[0.1,0.1,0.1,0,0]) = 0.006
        # Lower Bound Quantile Loss = mean([0.005*0.6, 0.005*0.6, 0.005*0.6,0,0]) = 0.005*0.6 * (3/5)) = 0.003 * (3/5)
        # Upper Bound Quantile Loss = mean([0.005*0.4, 0.005*0.4, 0.005*0.4,0,0]) = 0.005*0.4 * (3/5)= 0.002 * (3/5)
        # So the overall loss is the summation of all losses = 0.006 + (0.002 + 0.003)*3/5 = 0.0009
        self.assertTrue(np.isclose(0.009, loss))

    def test_deep_eye_forecast_model_builder_norm(self):
        # the input tensor
        tsr: tf.Tensor = tf.constant(np.array([1.0, 2, 3, 4, 5]).reshape((1, 5)))
        tsr_norm: tf.Tensor = AlerTigerForecastModelBuilder._norm(tsr)
        self.assertListEqual(
            [[-1.414213562373095, -0.7071067811865475, 0.0, 0.7071067811865475, 1.414213562373095, 1.4142135623730951, 3.0]],
            tsr_norm.numpy().tolist()
        )

    def test_deep_eye_forecast_model_builder_inv_norm(self):
        # the input tensor
        tsr_norm: tf.Tensor = tf.constant(
            np.array([-1.414213562373095, -0.7071067811865475, 0.0, 0.7071067811865475, 1.414213562373095, 1.4142135623730951, 3.0]).reshape((1, 7)))
        tsr: tf.Tensor = AlerTigerForecastModelBuilder._inv_norm(tsr_norm)
        self.assertTrue(np.isclose(np.array([[1, 2, 3, 4, 5]]), tsr.numpy()).all())

    def test_deep_eye_forecast_model_builder_compute_regularity_score(self):
        value_input = np.sin(range(28)).reshape((1, 28))
        res: tf.Tensor = AlerTigerForecastModelBuilder._compute_regularity_score(value_input)
        self.assertListEqual(
            [[0.5709147453308105]],
            res.numpy().tolist()
        )

    def test_deep_eye_classification_model_builder_build(self):
        classification_model: Model = self.classification_model_builder.build(self.forecast_model_builder.build())
        self.assertEqual(25, len(classification_model.layers))
        self.assertEqual(3, len(classification_model.inputs))
        self.assertEquals(1, len(classification_model.outputs))

    def test_deep_eye_classification_model_builder_norm(self):
        tsr_orig = tf.constant(np.array([1, 2, 3, 4, 5]).reshape((1, 5)))
        tsr_std = tf.constant(np.array([5]).reshape((1, 1)))
        tsr_mean = tf.constant(np.array([5]).reshape((1, 1)))

        tsr_norm = AlerTigerClassificationModelBuilder._norm((tsr_orig, tsr_std, tsr_mean))
        self.assertListEqual(
            [[-0.8, -0.6, -0.4, -0.2, 0.0]],
            tsr_norm.numpy().tolist()
        )
