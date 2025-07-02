from .. import logger
from ..dataset import Dataset
from ..window import WindowConfig
from ..progress import ProgressBar
from ..predictors.predictor import BasePredictor
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
    _logger.info(f"simulate DLDS technique using parameters:")
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
    # 0. start the simulation at ws
    idx = window_config.ws
    while idx + window_config.ts < test_data.shape[0]:
        progress.update(idx)

        # 1. predict values from buffer
        y_pred = predictor.predict(buffer)[0]
        inferences_count += 1

        # 2. compute individual errors of each predicted value
        pred_error_list = []
        pred_error_percent_list = []
        for i in range(window_config.ts):
            # compute error between predicted value and real value
            actual_val = test_data[idx + i]
            pred_error_list.append(abs(actual_val - y_pred[i]))
            pred_error_percent_list.append(abs(actual_val - y_pred[i]) / actual_val)

        # 3. read last value at ts-1
        y_real = test_data[idx + window_config.ts - 1]
        sensing_count += 1

        # 4. compute error threshold
        eps = (y_real * error) / 100
        if y_pred[-1] >= (y_real - eps) and y_pred[-1] <= (y_real + eps):
            # skip sending to the server
            skip_count += window_config.ts
            error_acc += sum(pred_error_list)
            error_percent_acc += sum(pred_error_percent_list)

            # update buffer with predicted values
            for i in range(window_config.ts):
                buffer = np.roll(buffer, -1)
                buffer[:, -1] = y_pred[i]
        else:
            # send to the server
            send_count += 1
            skip_count += window_config.ts - 1
            error_acc += sum(pred_error_list[:-1])
            error_percent_acc += sum(pred_error_percent_list[:-1])

            match realign:
                case "simple-append":
                    # update buffer by putting ts-1 pred values and the last real value
                    for i in range(window_config.ts):
                        buffer = np.roll(buffer, -1)
                        buffer[:, -1] = y_pred[i]
                    buffer[:, -1] = y_real
                case 'lerp':
                    # update buffer by putting ts pred values
                    for i in range(window_config.ts):
                        buffer = np.roll(buffer, -1)
                        buffer[:, -1] = y_pred[i]
                    buffer[:, -1] = y_real
                    
                    p1 = buffer[0][0][0]
                    p2 = buffer[0][-1][0]
                    omega = (p2-p1)/(window_config.ws-1)
                    for i in range(window_config.ws):
                        buffer[:,i] = p1 + i * omega
                case _:
                    _logger.fatal(f"unknown realign parameter '{realign}'")
                    raise Exception(f"unknown realign parameter '{realign}'")

        # 6. move to the next iteration
        idx += window_config.ts

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
