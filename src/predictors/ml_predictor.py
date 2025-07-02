from .. import logger
from .predictor import BasePredictor
from ..ml_model import get_model_path
from ..window import WindowConfig
from tensorflow import keras
import numpy as np


class MLPredictor(BasePredictor):
    """
    Predictor that uses a machine learning model.
    """

    def __init__(
        self,
        model_name: str,
        models_path: str,
        dataset_name: str,
        window_config: WindowConfig,
        seed: int,
    ):
        """
        Initializes the ML predictor by loading a model from disk.

        Parameters:
            model_name (str): The name of the model.
            models_path (str): Base path to the directory containing trained models.
            dataset_name (str): Name of the dataset used during training.
            window_config (WindowConfig): Window configuration object.
            seed (int): Seed used to ensure reproducibility.
        """
        super().__init__()
        self._logger = logger.get_logger(self.__class__.__name__)

        self._model_name = model_name
        self._window_config = window_config

        self._full_model_path = get_model_path(
            model_name, models_path, dataset_name, window_config, seed
        )
        self._logger.debug(f"load model from '{self._full_model_path}'")
        self._inner_model = keras.models.load_model(self._full_model_path)

    def name(self) -> str:
        return self._model_name

    def predict(self, x) -> np.ndarray:
        return self._inner_model.predict(x, verbose=0).reshape(
            (x.shape[0], self._window_config.ts)
        )

    def update(self, sample):
        return None
