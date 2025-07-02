from . import logger
from .window import WindowConfig
from os.path import join
from tensorflow import keras
import numpy as np

_logger = logger.get_logger(__name__)


def get_model_path(
    model_name: str,
    model_path: str,
    dataset_name: str,
    window_config: WindowConfig,
    seed: int,
) -> str:
    """
    Returns the path for saving or loading a model based.

    Parameters:
        model_name (str): Name of the model architecture.
        model_path (str): Base path where models are stored.
        dataset_name (str): Name of the dataset used for training.
        window_config (WindowConfig): Window configuration parameters.
        seed (int): Random seed used for reproducibility.

    Returns:
        str: A full model path string formatted accordingly.
    """
    return join(
        model_path,
        dataset_name,
        f"{seed}",
        f"ws{window_config.ws}_ts{window_config.ts}",
        model_name,
    )


def build_model(
    model_name: str, params: dict, window_config: WindowConfig, adapt_data: np.ndarray
) -> keras.Model:
    """
    Builds and returns a Keras model based on the specified model name and parameters.

    Parameters:
        model_name (str): Name of the model architecture to build (e.g., 'model1').
        params (dict): Dictionary containing model-specific parameters.
        window_config (WindowConfig): Window configuration parameters.
        adapt_data (np.ndarray): A portion of the training data to use for adaption preprocessing layers.

    Returns:
        keras.Model: A compiled Keras model corresponding to the selected architecture.

    Raises:
        Exception: If the provided `model_name` is not supported.
        AssertionError: If required parameters are missing in `params`.
    """
    match model_name:
        case "model1":
            assert "lstm_units" in params

            # create normalization layer
            normalization_layer = keras.layers.Normalization()
            normalization_layer.adapt(adapt_data)

            model = keras.Sequential()
            model.add(keras.layers.InputLayer((window_config.ws, 1)))
            model.add(normalization_layer)
            model.add(
                keras.layers.LSTM(
                    params["lstm_units"], activation="linear", return_sequences=False
                )
            )
            model.add(keras.layers.Dense(window_config.ts, "linear"))
            return model
        case "model2":
            assert "filters" in params
            assert "kernel_size" in params
            assert "lstm_units" in params

            # create normalization layer
            normalization_layer = keras.layers.Normalization()
            normalization_layer.adapt(adapt_data)

            model = keras.Sequential()
            model.add(keras.layers.InputLayer((window_config.ws, 1)))
            model.add(normalization_layer)
            model.add(
                keras.layers.Conv1D(
                    filters=params["filters"],
                    kernel_size=params["kernel_size"],
                    activation="linear",
                    padding="causal",
                )
            )
            model.add(
                keras.layers.Conv1D(
                    filters=params["filters"],
                    kernel_size=params["kernel_size"],
                    activation="linear",
                    padding="causal",
                )
            )
            model.add(keras.layers.MaxPooling1D(pool_size=2))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.RepeatVector(window_config.ts))
            model.add(
                keras.layers.LSTM(
                    params["lstm_units"], activation="linear", return_sequences=False
                )
            )
            model.add(keras.layers.Dense(window_config.ts, "linear"))
            return model
        case "model3":
            assert "lstm_units" in params
            assert "dense" in params

            # create normalization layer
            normalization_layer = keras.layers.Normalization()
            normalization_layer.adapt(adapt_data)

            model = keras.Sequential()
            model.add(keras.layers.InputLayer((window_config.ws, 1)))
            model.add(normalization_layer)
            model.add(keras.layers.LSTM(params["lstm_units"], activation="linear"))
            model.add(keras.layers.RepeatVector(window_config.ts))
            model.add(
                keras.layers.LSTM(
                    params["lstm_units"], activation="linear", return_sequences=True
                )
            )
            model.add(
                keras.layers.TimeDistributed(
                    keras.layers.Dense(params["dense"], activation="linear")
                )
            )
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
            return model
        case "model3unrolled":
            assert "lstm_units" in params
            assert "dense" in params

            model = keras.Sequential()
            model.add(keras.layers.InputLayer((window_config.ws, 1)))
            model.add(
                keras.layers.LSTM(
                    params["lstm_units"], activation="linear", return_sequences=True
                )
            )
            model.add(
                keras.layers.LSTM(
                    params["lstm_units"], activation="linear", return_sequences=False
                )
            )
            model.add(keras.layers.Dense(params["dense"], activation="linear"))
            model.add(keras.layers.Dense(window_config.ts))
            return model
        case _:
            _logger.fatal(f"unsupported model '{model_name}'")
            raise Exception(f"unsupported model '{model_name}'")
