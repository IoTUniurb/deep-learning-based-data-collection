from dataclasses import dataclass


@dataclass
class WindowConfig:
    """
    Class configuring the time series forecasting window parameters.

    Attributes:
        ws (int): Size of the window with past time steps used as input features.
        ts (int): Number of future time steps to predict (output horizon).
    """

    ws: int
    ts: int = 1
