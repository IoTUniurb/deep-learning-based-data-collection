from .predictor import BasePredictor
import numpy as np


class DBPPredictor(BasePredictor):
    """
    Predictor based on the DBP technique.

    This model estimates the trend (sigma) using the difference between averages
    of edge points in the input window and extrapolates future values accordingly.
    """

    def __init__(self, learning_phase: int, edge_points: int = 3):
        """
        Initializes the DBPPredictor.

        Parameters:
            learning_phase (int): Length of the learning phase.
            edge_points (int, optional): Number of values to use from the start and end of the window for trend estimation. Default is 3.
        """
        super().__init__()

        self._learning_phase = learning_phase
        self._edge_points = edge_points
        self._time_steps = 1

    def name(self) -> str:
        return "DBP"

    def predict(self, x):
        # compute sigma from
        sigma = (
            np.mean(x[:, -self._edge_points :], axis=1)
            - np.mean(x[:, : self._edge_points], axis=1)
        ) / (self._learning_phase - 1)

        # predict next time_steps
        ret = np.zeros((x.shape[0], self._time_steps, 1))
        for ts in range(self._time_steps):
            ret[:, ts] = np.mean(x[:, -self._edge_points :], axis=1) + (
                (ts + 1) * sigma
            )
        ret = ret.reshape((x.shape[0], self._time_steps))
        return ret

    def update(self, sample):
        return None
