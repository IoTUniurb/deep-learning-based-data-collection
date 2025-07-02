from . import utils
from . import logger
from .dataset import Dataset, to_supervised
from .window import WindowConfig
from . import ml_model
from sklearn.model_selection import train_test_split
from tensorflow import keras

_logger = logger.get_logger(__name__)


def _train_model(
    dataset: Dataset,
    model_name: str,
    model_path: str,
    model_param: dict,
    output_path: str,
    window_config: WindowConfig,
    seed: int,
    optimizer: str = "adam",
    loss: str = "mse",
    metrics: list[str] = ["mean_absolute_error", "mean_absolute_percentage_error"],
):
    """
    Train a specific model on the provided dataset using supervised learning.

    Parameters:
        dataset (Dataset): The dataset to train on.
        model_name (str): Identifier of the model to be trained.
        model_path (str): Path where the trained model will be saved.
        model_param (dict): Dictionary containing model-specific parameters.
        output_path (str): Path where training metrics will be written.
        window_config (WindowConfig): Window configuration parameters.
        seed (int): Random seed for reproducibility.
        optimizer (str, optional): Optimizer to use during training. Defaults to "adam".
        loss (str, optional): Loss function. Defaults to "mse".
        metrics (list, optional): List of metrics for model evaluation. Defaults to MAE and MAPE.

    Returns:
        list: Evaluation scores on the test set, typically [loss, metric1, metric2, ...].
    """
    assert "epochs" in model_param
    # TODO: Make `batch_size` an optional field of model_params and use a default value when not present.
    assert "batch_size" in model_param

    _logger.debug(f"train model '{model_name}' with:")
    _logger.debug(f"  {seed          = }")
    _logger.debug(f"  {window_config = }")
    _logger.debug(f"  {model_param   = }")
    _logger.debug(f"  {optimizer     = }")
    _logger.debug(f"  {loss          = }")
    _logger.debug(f"  {metrics       = }")

    # set seed for training
    utils.set_seed(seed)

    # split dataset and convert it to supervised
    train_data, _ = dataset.train_test_split(type="random", seed=seed)
    train_data = train_data.reshape((train_data.shape[0] * train_data.shape[1], 1))
    train_sup_x, train_sup_y = to_supervised(train_data, window_config)

    # split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        train_sup_x, train_sup_y, random_state=seed, shuffle=True, train_size=0.80
    )
    _logger.debug(f"train data shapes: ")
    _logger.debug(f"  {x_train.shape = } {y_train.shape}")
    _logger.debug(f"  {x_test.shape = } {y_test.shape}")

    # load the model
    model_path = ml_model.get_model_path(
        model_name, model_path, dataset.name(), window_config, seed
    )
    _logger.debug(f"store model into '{model_path}'")
    model_save_cb = keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)

    # build the model
    model = ml_model.build_model(model_name, model_param, window_config, x_train)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # train the model
    model.fit(
        x_train,
        y_train,
        validation_split=0.10,
        epochs=model_param["epochs"],
        callbacks=[model_save_cb],
        batch_size=model_param["batch_size"],
        verbose=2,
    )

    # evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)

    utils.save_metrics(
        "train.csv",
        output_path,
        {
            "dataset": dataset.name(),
            "seed": seed,
            "model": model_name,
            "window_size": window_config.ws,
            "time_steps": window_config.ts,
            "param": model_param,
            "score": score,
        },
    )


def train_models(
    dataset: Dataset,
    window_config: WindowConfig,
    models_path: str,
    output_path: str,
    seed: int,
    params: dict,
):
    """
    Train multiple models on a given dataset with specified window configuration and parameters.

    Parameters:
        dataset (Dataset): The dataset to train on.
        window_config (WindowConfig): Window configuration parameters.
        models_path (str): Path where trained models will be saved.
        output_path (str): Path where training metrics will be written.
        seed (int): Seed for reproducibility during training.
        params (dict): Dictionary where keys are model names and values are parameter dictionaries.
    """
    _logger.info(f"train models using parameters:")
    _logger.info(f"  dataset       = '{dataset.name()}'")
    _logger.info(f"  {window_config = }")
    _logger.info(f"  {seed          = }")
    _logger.debug(f" {params        = }")

    scores = {}
    for model_name, model_param in params.items():
        score = _train_model(
            dataset,
            model_name,
            models_path,
            model_param,
            output_path,
            window_config,
            seed,
        )
        scores[model_name] = score
    _logger.debug(f"training scores= {scores}")
