from . import logger
import os
import csv
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)

_logger = logger.get_logger(__name__)


def set_seed(seed: int):
    """
    Set the seed for python, numpy and tensorflow.
    """
    # TODO: It is possibe that using imports inside the function will break when using a GPU.
    import os

    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    import tensorflow as tf
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()
    if len(tf.config.list_physical_devices("GPU")) != 0:
        tf.config.experimental.enable_op_determinism()
        tf.config.experimental.set_memory_growth(
            tf.config.list_physical_devices("GPU")[0], True
        )


def save_metrics(file_name: str, out_path: str, metrics: dict):
    """
    Save metrics to a CSV file.

    Parameters:
        file_name (srt): Name of the output CSV file.
        out_path (str): Path to the output CSV directory.
        metrics (dict): Dictionary of metrics to save.
    """
    # TODO: Find a better way of storing metrics.
    #  An example could be having a directory for each script (train, simulate, etc.)
    #  and inside having multiple files with progressive manes.
    #  Another idea is storing the timestamp of each row to know when they where created.
    full_path = os.path.join(out_path, file_name)

    # Ensure output directory exists
    os.makedirs(out_path, exist_ok=True)

    with open(full_path, mode="a", newline="") as csvfile:
        _logger.debug(f"write metrics in '{out_path}'")

        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        if not os.path.exists(out_path):
            _logger.debug(f"new output file created '{out_path}'")
            writer.writeheader()
        writer.writerow(metrics)


def compute_metrics(real_values, pred_values):
    """
    Compute evaluation metrics between real and predicted values.

    Parameters:
        real_values (array-like): Ground truth values.
        pred_values (array-like): Predicted values from a model.

    Returns:
        dict: Dictionary containing each computed metric organized by its name.
    """
    mae = mean_absolute_error(real_values, pred_values)
    mape = mean_absolute_percentage_error(real_values, pred_values)
    mapes = mean_absolute_percentage_error(
        real_values, pred_values, multioutput="raw_values"
    )

    ret = {
        "mae": mae,
        "mape": mape,
    }
    if real_values.shape[1] > 1:
        ret["individual_mape"] = list(mapes)

    return ret
