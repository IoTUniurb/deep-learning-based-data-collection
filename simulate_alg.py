from src.dataset import Dataset
from src.dataset_loader import (
    NoWeekLoader,
    WeatherLoader,
    TrafficLoader,
    ElectricityLoader,
)
from src.window import WindowConfig
from src.techniques import dlbdc
from src.techniques import dlds
from src.predictors.ml_predictor import MLPredictor
from src.predictors.dbp_predictor import DBPPredictor
from src.predictors.kf_predictor import KFPredictor

DATASET_DIR = "/home/l.calisti/notebooks/dlds_paper/datasets"
MODELS_DIR = "/home/l.calisti/notebooks/dlds_paper/models"
OUTPUT_DIR = "/home/l.calisti/notebooks/dlds_paper/outputs"
SEEDS = [69]  # [42, 69, 911, 2020, 42069]
WS = [5]  # [3, 5, 7, 10, 15]
TS = [1, 2]
ERRORS = [3]  # [1, 3, 5, 7, 10]
ALPHAS = [0.5, 1.0, 1.0, 1.0]  # [0.25, 0.40, 0.50, 0.75, 0.90, 1.0]
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
        name=dataset_name, base_path=DATASET_DIR, loader=dataset_loader, smooth=None
    )
    for seed in SEEDS:
        for ws in WS:
            for ts in TS:
                for error in ERRORS:
                    wc = WindowConfig(ws, ts)
                    predictor = MLPredictor(
                        model_name="model3",
                        models_path=MODELS_DIR,
                        dataset_name=ds.name(),
                        window_config=wc,
                        seed=seed,
                    )
                    dlds.simulate(
                        dataset=ds,
                        output_path=OUTPUT_DIR,
                        predictor=predictor,
                        window_config=wc,
                        error=error,
                        seed=seed,
                        realign="lerp",
                    )

                    dlbdc.simulate(
                        dataset=ds,
                        output_path=OUTPUT_DIR,
                        predictor=predictor,
                        window_config=wc,
                        error=error,
                        seed=seed,
                        realign="simple-append",
                    )
                    predictor = DBPPredictor(20)
                    dlbdc.simulate(
                        dataset=ds,
                        output_path=OUTPUT_DIR,
                        predictor=predictor,
                        window_config=WindowConfig(20, 1),
                        error=error,
                        seed=seed,
                        realign="simple-append",
                    )
                    predictor = KFPredictor(3)
                    dlbdc.simulate(
                        dataset=ds,
                        output_path=OUTPUT_DIR,
                        predictor=predictor,
                        window_config=WindowConfig(3, 1),
                        error=error,
                        seed=seed,
                        realign="simple-append",
                    )
