from . import logger
from .window import WindowConfig
from .dataset_loader import DatasetLoader
from os.path import join
import pandas as pd
import numpy as np
from scipy.signal import convolve


class Dataset:
    def __init__(
        self,
        name: str,
        base_path: str,
        loader: DatasetLoader,
        normalize: bool = False,
        smooth: int = 4,
    ):
        """
        Initializes the dataset by loading it from a CSV file, optionally applying normalization and smoothing.

        Parameters:
            name (str): Name of the dataset file (without path), located under `base_path`.
            base_path (str): Directory where datasets are stored (default: 'datasets').
            normalize (bool): Whether to apply Z-score normalization to data (default: False).
            smooth (int): Window size for moving average smoothing. If None, smoothing is skipped (default: 4).
        """
        self._logger = logger.get_logger(self.__class__.__name__)

        self._ds_name = name
        self._dataset_path = join(base_path, self._ds_name)
        self._normalize = normalize
        self._smooth = smooth
        self._ds_loader = loader

        self.__load_dataset()

        # scale radioactivity dataset
        if self.name().startswith("rad"):
            self._full_data_df *= 1000

        if self._normalize:
            self.__normalize()

        if self._smooth != None:
            self.__smooth()

        # save the data as numpy arrays for better management
        self._full_data_np = self._full_data_df.values
        self._full_data_ts = self._full_data_df.index.values

    def __repr__(self):
        return f"Dataset(name={self.name()}, path={self._dataset_path})"

    def values(self) -> np.ndarray:
        """
        Returns the full dataset as a 2-D array with shape (N, 1).

        Returns:
            np.ndarray: The underlying time series data as a (N, 1) array.
        """
        return self._full_data_np

    def timestamps(self) -> np.ndarray:
        """
        Returns the full dataset's timestamps as a 2-D array with shape (N, 1).

        To retrieve the associated values, use the `values()` method.

        Returns:
            np.ndarray: The timestamps of the dataset as a (N, 1) array.
        """
        return self._full_data_ts

    def name(self) -> str:
        """
        Return the name of the dataset in a format readable to humans.
        """
        addition = self._ds_loader.name_addition()
        basename = self._ds_name.split(".")[0]
        if addition != "":
            return f"{basename} {addition}"
        else:
            return basename

    def train_test_split(
        self,
        type: str,
        train_split: float = 0.5,
        chunk_size: int = 500,
        test_chunks: int = 10,
        seed: int = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits the dataset into training and testing sets using one of two strategies:
        - 'sequential': Splits the full dataset sequentially at the specified ratio.
        - 'random': Splits the dataset into fixed-size chunks and randomly selects some for testing.

        Parameters:
            type (str): Splitting strategy, either 'sequential' or 'random'.
            train_split (float): Ratio of data to include in the training set (only for 'sequential').
            chunk_size (int): Length of each chunk (only for 'random').
            test_chunks (int): Number of chunks to use for testing (only for 'random').
            seed (int, optional): Random seed for reproducibility (only for 'random').

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing (train_data, test_data), where:
                - If 'sequential': arrays of shape (N,1)
                - If 'random': arrays of shape (M, chunk_size, 1)

        Raises:
            Exception: If the provided `type` is unknown.
        """
        if type == "sequential":
            self._logger.debug(
                f"split dataset using 'sequential' with {train_split} train-test split"
            )
            # split in train/test datasets
            split_index = int(len(self._full_data_np) * train_split)
            train_np = self._full_data_np[:split_index]
            test_np = self._full_data_np[split_index:]
        elif type == "random":
            self._logger.debug(
                f"split dataset using 'random' and keep {test_chunks} chunks for test"
            )
            # set custom seed
            if seed != None:
                np.random.seed(seed)

            chunk_num = len(self._full_data_np) // chunk_size
            chunks = self._full_data_np[: chunk_num * chunk_size].reshape(
                (chunk_num, chunk_size, 1)
            )
            self._logger.debug(
                f"split dataset into {chunk_num} chunks of size {chunk_size}"
            )

            all_idx = np.arange(chunk_num)
            test_idx = np.random.choice(chunk_num, size=test_chunks, replace=False)
            train_idx = np.setdiff1d(all_idx, test_idx)
            self._logger.debug(f"train idxs {train_idx}")
            self._logger.debug(f"train idxs {test_idx}")

            train_np = chunks[train_idx]
            test_np = chunks[test_idx]
        else:
            self._logger.fatal(f"train_test_split: unknown type '{type}'")
            raise Exception(f"train_test_split: unknown type '{type}'")

        return (train_np, test_np)

    def __load_dataset(self):
        """
        Loads a time series dataset using a DatasetLoader.
        """
        self._full_data_df = self._ds_loader.load(self._dataset_path)
        self._logger.debug(
            f"load dataset from '{self._dataset_path}' using loader {self._ds_loader}"
        )
        self._logger.debug(f"dataset shape: {self._full_data_df.shape}")

    def __smooth(self):
        """
        Applies a moving average smoothing over the dataset.
        """
        np_smooth = convolve(
            self._full_data_df, np.ones((self._smooth, 1)) / self._smooth, mode="valid"
        )
        self._full_data_df = self._full_data_df.drop(
            self._full_data_df.iloc[: (self._smooth - 1), :].index.tolist()
        )
        self._full_data_df["_value"] = np_smooth
        self._logger.debug(f"smooth dataset")

    def __normalize(self):
        """
        Normalize the dataset using a standard score normalization (Z-score).
        """
        self._ds_mean = np.mean(self._full_data_df)
        self._ds_std = np.std(self._full_data_df, axis=0).values[0]
        self._full_data_df = (self._full_data_df - self._ds_mean) / self._ds_std
        self._logger.debug(
            f"normalize dataset with mean={self._ds_mean}, std={self._ds_std}"
        )

    def __denormalize(self):
        """
        Denormalize the dataset using a standard score normalization (Z-score).
        """
        self._full_data_df = (self._full_data_df * self._ds_std) + self._ds_mean
        self._logger.debug(
            f"denormalize dataset with mean={self._ds_mean}, std={self._ds_std}"
        )


def to_supervised(data, window_config: WindowConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a univariate time series (numpy array) into a supervised learning format,
    using a fixed-size input window and a specified number of forecasting time steps.

    The function creates overlapping sequences where each input sample (X) consists of
    `window_size` past observations, and each target sample (Y) consists of `time_steps`
    future observations. The output is suitable for training models on multi-step
    forecasting tasks.

    Parameters:
        data (np.ndarray): A 2D numpy array of shape (N, 1), representing the time series.
        window_config (WindowConfig): Window configuration parameters.

    Returns:
        X (np.ndarray): Supervised input features of shape (samples, window_size, 1).
        Y (np.ndarray): Supervised targets of shape (samples, time_steps).
    """
    _logger = logger.get_logger(__name__)

    def __df_to_time_steps(df, window_size, time_steps):
        # Taken and modified from https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
        cols = []
        names = []
        # previous time steps
        for i in range(window_size, 0, -1):
            cols.append(df.shift(i))
            names += [("t-%d") % i]
        # current and forward time-steps
        for i in range(0, time_steps):
            cols.append(df.shift(-i))
            if i == 0:
                names += ["t"]
            else:
                names += [("t+%d") % i]

        df_timesteps = pd.concat(cols, axis=1)
        df_timesteps.columns = names
        # drop Nan values
        df_timesteps.dropna(inplace=True)

        return df_timesteps

    def __to_supervised(d_set, window_size, time_steps):
        # taken and modified from https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
        # flatten data
        data = d_set.reshape((d_set.shape[0], d_set.shape[1], 1))
        X, Y = [], []
        i = 0
        X = data[:, :window_size, :]
        Y = data[:, window_size : window_size + time_steps, :]
        Y = Y.reshape(Y.shape[0], Y.shape[1])
        return np.array(X), np.array(Y)

    _logger.debug(
        f"convert array with shape={data.shape} to supervised using window={window_config}"
    )
    data_df = pd.DataFrame(data)
    ts = __df_to_time_steps(data_df, window_config.ws, window_config.ts).values
    (x, y) = __to_supervised(ts, window_config.ws, window_config.ts)
    _logger.debug(f"{x.shape=}, {y.shape=}")

    return (x, y)
