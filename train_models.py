from src.dataset import Dataset
from src.dataset_loader import (
    NoWeekLoader,
    WeatherLoader,
    TrafficLoader,
    ElectricityLoader,
)
from src.training import train_models
from src.window import WindowConfig

DATASET_DIR = "/home/l.calisti/notebooks/dlds_paper/datasets"
MODELS_DIR = "/home/l.calisti/notebooks/dlds_paper/models"
OUTPUT_DIR = "/home/l.calisti/notebooks/dlds_paper/outputs"
MODELS_PARAM = {
    # "model1": {
    #     "lstm_units": 10,
    #     "epochs": 100,  # 20
    #     "batch_size": 32,
    # },
    # "model2": {
    #     "filters": 45,
    #     "kernel_size": 3,
    #     "lstm_units": 3,
    #     "epochs": 100,  # 40
    #     "batch_size": 64,
    # },
    "model3": {
        "lstm_units": 10,
        "dense": 30,
        "epochs": 100,
        "batch_size": 32,
    },
    # 'model4': {
    #     'filters': [32, 24],
    #     'kernel_size': [3,3],
    #     'dense': 10,
    #     'epochs': 30,
    #     'batch_size': 32,
    # }
    # 'model3unrolled': {
    #     'lstm_units': 7,
    #     'epochs': 30
    # }
}
SEEDS = [69]  # [42, 69, 911, 2020, 42069]
WS = [5]  # [3, 5, 7, 10, 15, 40]
TS = [2, 3, 5, 7]  # [2, 3, 5, 7, 10, 15]
DATASET_NAMES = [
    ("noweekend/co2_peano_no_weekend.csv", NoWeekLoader()),
    ("noweekend/pm2p5_peano_no_weekend.csv", NoWeekLoader()),
    ("noweekend/rad_peano_no_weekend.csv", NoWeekLoader()),
    ("noweekend/noise_peano_no_weekend.csv", NoWeekLoader()),
    # ("external/weather.csv", WeatherLoader("T (degC)")),
    # ("external/weather.csv", WeatherLoader("rh (%)")),
    # ("external/weather.csv", WeatherLoader("wv (m/s)")),
    # ("external/weather.csv", WeatherLoader("SWDR (W/m�)")),
    # ("external/traffic.csv", TrafficLoader()),
    # ("external/electricity.csv", ElectricityLoader()),
]

for dataset_name, dataset_loader in DATASET_NAMES:
    ds = Dataset(
        name=dataset_name,
        base_path=DATASET_DIR,
        loader=dataset_loader,
        smooth=None,
    )
    for seed in SEEDS:
        for ws in WS:
            for ts in TS:
                train_models(
                    dataset=ds,
                    window_config=WindowConfig(ws, ts),
                    models_path=MODELS_DIR,
                    output_path=OUTPUT_DIR,
                    seed=seed,
                    params=MODELS_PARAM,
                )
