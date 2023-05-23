from dataclasses import dataclass


@dataclass
class MlpLayerConfig:
    """
    The MLP configuration for a single MLP layers
    """
    # number of neurons within a single MLP layer.
    units: int

    # the activation function to use, this can be "sigmoid" or "relu"
    activation: str

    # drop out rate.
    dropout_rate: float = 0.0

    # regularization method and weights for the kernel and bias weights.
    kernel_regularizer_method: str = "l1_l2"
    kernel_regularizer_l1_coeff: float = 0.0
    kernel_regularizer_l2_coeff: float = 0.0
    bias_regularizer_method: str = "l1_l2"
    bias_regularizer_l1_coeff: float = 0.0
    bias_regularizer_l2_coeff: float = 0.0
