import numpy as np


class BasePredictor:
    """
    Abstract base class for all prediction models.

    Defines a common interface that all predictors must implement.
    """

    def name(self) -> str:
        """
        Returns the name of the prediction model.
        """
        raise NotImplementedError()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction based on the input data.

        Parameters:
            x (np.ndarray): Input data as 3-D array with shape(N, ts, 1).

        Returns:
            np.ndarray: Model predictions as a 2-D array with shape (N, ts).
        """
        raise NotImplementedError()

    def update(self, sample: float):
        """
        Updates the internal model based on the new sample data.

        Parameters:
            sample (float): The new sample data.
        """
        raise NotImplementedError()