import pandas as pd


class DatasetLoader:
    def load(self, path: str) -> pd.DataFrame:
        """
        Load the dataset contained in the given file.

        Parameters:
            path (str): Path to the dataset file to load.

        Returns:
            pd.DataFrame: The full dataset loaded.
        """
        raise NotImplementedError()

    def name_addition(self) -> str:
        """
        Additional string used to customize the name of the dataset.

        This string is used in some rare cases when using just the dataset path as name
        is not sufficient. The additional string is concatenated to the file name.

        Returns:
            str: The additional name string.
        """
        return ""


class NoWeekLoader(DatasetLoader):
    """
    Load one of the `_no_weekend` datasets obtained
    by removing weekends from the InfluxDB datasets.
    """

    def load(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        df = data[["_time", "_value"]]
        df = df[:-1]
        df = df.set_index("_time")
        return df


class InfluxLoader(DatasetLoader):
    """
    Load one of the original InfluxDB datasets.
    """

    def load(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path, header=3)
        df = data[["_time", "_value"]]
        df["_time"] = pd.to_datetime(df["_time"], format="ISO8601")
        df = df[:-1]
        df = df.set_index("_time")
        return df


class WeatherLoader(DatasetLoader):
    """
    Load the public weather dataset selecting one column to use.
    https://www.kaggle.com/datasets/giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets?select=electricity.csv
    https://www.bgc-jena.mpg.de/wetter/index.html

    Parameters:
        column (str): Name of the column to select when loading the dataset.
    """

    def __init__(self, column: str):
        self._column = column

    def __repr__(self):
        return f"WeatherLoader(column={self._column})"

    def load(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        df = data[["date", self._column]]
        df = df.rename({"date": "_time", self._column: "_value"}, axis=1)
        df["_time"] = pd.to_datetime(df["_time"])
        df = df[:-1]
        df = df.set_index("_time")

        match self._column:
            case "T (degC)":
                # Remove all values around 0 in range +- 0.01.
                df.loc[df["_value"].abs() < 0.01] = 0.01
            case "wv (m/s)":
                # Remove all sensor errors (negative values)
                df.loc[df["_value"] < -1000.0] = 0.08

        return df

    def name_addition(self):
        return self._column.replace("/", "_")


class TrafficLoader(DatasetLoader):
    """
    Load the public traffic dataset.
    https://www.kaggle.com/datasets/giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets?select=electricity.csv
    https://zenodo.org/records/4656132?utm_source=chatgpt.com
    """

    def load(self, path):
        data = pd.read_csv(path)
        df = data.rename({"date": "_time"}, axis=1)
        df["_time"] = pd.to_datetime(df["_time"])
        df = df[:-1]
        df = df.set_index("_time")

        mean = df.mean(axis=1)

        ret_df = pd.DataFrame(
            data={"_value": mean},
            index=df.index,
        )
        # Remove all values around 0 in range +-0.001 and multiply by 1000
        ret_df.loc[ret_df["_value"].abs() < 0.001] = 0.001
        ret_df *= 1000.0
        return ret_df


class ElectricityLoader(DatasetLoader):
    """
    Load the public electricity dataset.
    https://www.kaggle.com/datasets/giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets?select=electricity.csv
    https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
    """

    def load(self, path):
        data = pd.read_csv(path)
        df = data.rename({"date": "_time"}, axis=1)
        df["_time"] = pd.to_datetime(df["_time"])
        df = df[:-1]
        df = df.set_index("_time")

        sum = df.sum(axis=1)
        sum /= 4

        ret_df = pd.DataFrame(
            data={"_value": sum},
            index=df.index,
        )
        return ret_df
