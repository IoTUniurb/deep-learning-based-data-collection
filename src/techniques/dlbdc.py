from .. import logger
from ..dataset import Dataset
from ..window import WindowConfig
from ..progress import ProgressBar
from ..predictors.predictor import BasePredictor
from ..predictors.kf_predictor import KFPredictor
from .. import utils
import numpy as np

_logger = logger.get_logger(__name__)


def simulate(
    dataset: Dataset,
    output_path: str,
    predictor: BasePredictor,
    window_config: WindowConfig,
    error: int,
    seed: int,
    realign: str,
    alpha: float = 1.0,
):
    """
    Run a simulation of the DLBDC algorithm on a given dataset.

    Parameters
    ----------
    dataset (Dataset): Dataset to use for the simulation.
    output_path (str): Directory where the simulation metrics are saved.
    predictor (BasePredictor): Predictor.
    window_config (WindowConfig): Window configuration parameters.
    error (int): Allowed relative error (percentage).
    seed (int): Random seed used to split the dataset.
    realign (str): Realignment strategy for buffer update.
        Can be one of: ["simple-append", "scaled-distance"]
    alpha (float): Scaling factor for the "scaled-distance" strategy. Default is 1.0.
    """
    _logger.info(f"simulate technique using parameters:")
    _logger.info(f"  dataset       = '{dataset.name()}'")
    _logger.info(f"  predictor     = '{predictor.name()}'")
    _logger.info(f"  {window_config = }")
    _logger.info(f"  {error         = }")
    _logger.info(f"  {seed          = }")
    _logger.info(f"  {realign       = }")
    _logger.info(f"  {alpha         = }")

    # load simulation data
    _, test_data = dataset.train_test_split(type="random", seed=seed)
    test_data = test_data.reshape((test_data.shape[0] * test_data.shape[1]))
    _logger.debug(f"test data shape: {test_data.shape}")

    # create progressbar
    progress = ProgressBar(test_data.shape[0])

    # initialize buffer with the first few samples
    buffer = test_data[: window_config.ws].reshape((1, window_config.ws, 1)).copy()
    _logger.debug(f"buffer shape: {buffer.shape}")

    # initialize metrics counters
    sensing_count = window_config.ws
    inferences_count = 0
    send_count = window_config.ws
    skip_count = 0
    error_acc = 0
    error_percent_acc = 0

    # simulate
    idx = window_config.ws
    while idx < test_data.shape[0]:
        progress.update(idx)

        # 1. read real value from current iteration
        y_real = test_data[idx]
        # _logger.debug(f"predicted value: {y_real}")
        sensing_count += 1

        # 2. predict value from current iteration
        y_pred = predictor.predict(buffer)[0][0]
        # _logger.debug(f"predicted value: {y_pred}")
        inferences_count += 1

        # 3. compute error threshold
        eps = (y_real * error) / 100
        if y_pred >= (y_real - eps) and y_pred <= (y_real + eps):
            # 4.1. skip sending to the server
            skip_count += 1
            error_acc += abs(y_real - y_pred)
            error_percent_acc += abs(y_real - y_pred) / y_real

            # 5. update buffer with predicted value
            buffer = np.roll(buffer, -1)
            buffer[:, -1] = y_pred
        else:
            match realign:
                case "simple-append":
                    # send to the server
                    send_count += 1
                    # update buffer with predicted value
                    buffer = np.roll(buffer, -1)
                    buffer[:, -1] = y_real
                case "scaled-distance":
                    # send to the server
                    send_count += 1
                    # compute the scaled distance between the last point in the buffer and the new measured value
                    p1 = buffer[0][-1][0]
                    p2 = y_real
                    delta = (p2 - p1) * alpha

                    # update buffer with scaled distance
                    buffer = np.roll(buffer, -1)
                    buffer[:, -1] = p1 + delta
                    # _logger.error(f"{p1=} {p2=} {delta=} {p1+delta=}")
                case _:
                    _logger.fatal(f"unknown realign parameter '{realign}'")
                    raise Exception(f"unknown realign parameter '{realign}'")

        # 6. call update with the latest value inside the buffer
        predictor.update(buffer[:, -1])

        # 7. move to next iteration
        idx += 1

    # save metrics
    utils.save_metrics(
        "simulate.csv",
        output_path,
        {
            "dataset": dataset.name(),
            "seed": seed,
            "predictor_name": predictor.name(),
            "window_size": window_config.ws,
            "time_steps": window_config.ts,
            "error": error,
            "realign": realign,
            "alpha": alpha,
            "tot_samples": test_data.shape[0],
            "sensing_count": sensing_count,
            "inferences_count": inferences_count,
            "send_count": send_count,
            "skip_count": skip_count,
            "error_acc": error_acc,
            "error_percent_acc": error_percent_acc,
        },
    )
