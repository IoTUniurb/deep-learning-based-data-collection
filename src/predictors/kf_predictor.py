from .predictor import BasePredictor
from filterpy.kalman import KalmanFilter
import numpy as np


class KFPredictor(BasePredictor):
    """
    Predictor based on the Kalman Filter algorithm for one-step-ahead forecasting.

    This implementation uses the `filterpy` library to apply a basic linear Kalman filter
    to a time series. The predictor performs prediction and update steps explicitly and
    supports only single time-step forecasts.
    """

    def __init__(self, x_size):
        """
        Initializes the Kalman Filter with the specified state size.

        Parameters:
            x_size (int): Dimension of the internal state vector.
        """
        self.x_size = x_size
        self._time_steps = 1

        # TODO: Make it work with multi TS
        self._kf = KalmanFilter(dim_x=self.x_size, dim_z=self._time_steps)
        self._kf.x = np.zeros((self.x_size, self._time_steps))
        self._kf.F = np.eye(self.x_size)
        # self.kf.H = np.zeros((self.kf.dim_z, self.kf.x_size))
        # self.kf.H[0][0] = 1
        self._kf.P *= 10.0
        # q_var = 0#np.var(train_np)
        self._kf.Q = np.eye(self.x_size) * 0.001  # * q_var
        # self.kf.R = np.array([[0.1]])
        self._kf.R /= 10

        # self.kf.x = np.zeros(5)
        # self.kf.F = np.eye(5)
        # self.kf.P = np.eye(5) * 10.0
        # self.kf.Q = np.eye(5) * 0.001
        # self.kf.R = np.array([[0.1]])
        self._kf.H = np.ones((self._time_steps, self.x_size)) / self.x_size
        # self.kf.H = np.array([[1,1,1,1,1]])/5
        # self.kf.H = np.array([[1,0,0,0,0]])
        # self.kf.H = np.array([[0.05,0.05,0.1,0.3,0.5]])
        # self.kf.H = np.array([[0.5,0.3,0.1,0.05,0.05]])

    def name(self) -> str:
        return "KF"

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._kf.predict()
        return np.dot(self._kf.H, self._kf.x)

    def update(self, sample: float):
        self._kf.update(sample)
