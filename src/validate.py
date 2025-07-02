from . import utils
from . import logger
from .dataset import Dataset, to_supervised
from .window import WindowConfig
from .predictors.predictor import BasePredictor
from .predictors.ml_predictor import MLPredictor
from .predictors.dbp_predictor import DBPPredictor
from .predictors.kf_predictor import KFPredictor
import numpy as np

_logger = logger.get_logger(__name__)


def validate(
    dataset: Dataset,
    predictor: BasePredictor,
    window_config: WindowConfig,
    output_path: str,
    seed: int,
):
    """
    Validates a predictor on a given dataset and saves the results to a CSV file.

    Parameters:
        dataset (Dataset): The dataset to validate against.
        predictor (BasePredictor): The predictor.
        window_config (WindowConfig): Window configuration options.
        output_path (str): Path to the directory where the validation results will be saved.
        seed (int): Random seed for reproducibility.
    """
    _logger.info(f"validate model using parameters:")
    _logger.info(f"  dataset       = '{dataset.name()}'")
    _logger.info(f"  predictor     = '{predictor.name()}'")
    _logger.info(f"  {window_config = }")
    _logger.info(f"  {seed          = }")

    if isinstance(predictor, MLPredictor) or isinstance(predictor, DBPPredictor):
        metrics = _validate_batch(dataset, predictor, window_config, seed)
    elif isinstance(predictor, KFPredictor):
        metrics = _validate_unrolled(dataset, predictor, window_config, seed)
    else:
        _logger.fatal(f"unknown predictor {predictor}")
        raise Exception(f"unknown predictor {predictor}")

    utils.save_metrics(
        "validate.csv",
        output_path,
        {
            "dataset": dataset.name(),
            "predictor_name": predictor.name(),
            "seed": seed,
            "window_size": window_config.ws,
            "time_steps": window_config.ts,
            "metrics": metrics,
        },
    )


def _validate_batch(
    dataset: Dataset,
    predictor: BasePredictor,
    window_config: WindowConfig,
    seed: int,
):
    utils.set_seed(seed)

    # load validation data and convert it to supervised
    _, test_data = dataset.train_test_split(type="random", seed=seed)
    test_data = test_data.reshape((test_data.shape[0] * test_data.shape[1], 1))
    x_test, y_test = to_supervised(test_data, window_config)
    _logger.debug(f"test data shapes: {x_test.shape = } { y_test.shape = }")

    # predict the data
    y_pred = predictor.predict(x_test)
    _logger.debug(f"predicted data shape: {y_pred.shape = }")

    return utils.compute_metrics(y_test, y_pred)


def _validate_unrolled(
    dataset: Dataset,
    predictor: BasePredictor,
    window_config: WindowConfig,
    seed: int,
):
    utils.set_seed(seed)

    # load validation data
    _, test_data = dataset.train_test_split(type="random", seed=seed)
    test_data = test_data.reshape((test_data.shape[0] * test_data.shape[1], 1))
    x_test, y_test = to_supervised(test_data, window_config)
    _logger.debug(f"test data shapes: {x_test.shape = } { y_test.shape = }")

    y_pred = np.zeros(y_test.shape)

    for i in range(x_test.shape[0]):
        # load a new model
        if isinstance(predictor, KFPredictor):
            kf_predictor = KFPredictor(x_size=predictor.x_size)
        else:
            _logger.fatal(
                f"expected an object of KFPredictor class but got {predictor.__class__.__name__}"
            )
            raise Exception(
                f"expected an object of KFPredictor class but got {predictor.__class__.__name__}"
            )

        # initialize the model with the data on the buffer
        for val in x_test[i]:
            kf_predictor.predict(x_test[i])
            kf_predictor.update(val)

        # predict next value
        y_pred[i] = kf_predictor.predict(x_test[i])

    return utils.compute_metrics(y_test, y_pred)
