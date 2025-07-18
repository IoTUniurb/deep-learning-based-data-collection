from src.dataset import Dataset
from src.dataset_loader import (
    NoWeekLoader,
    WeatherLoader,
    TrafficLoader,
    ElectricityLoader,
)
from src.window import WindowConfig
from src.validate import validate
from src.predictors.ml_predictor import MLPredictor
from src.predictors.dbp_predictor import DBPPredictor
from src.predictors.kf_predictor import KFPredictor

DATASET_DIR = "/home/l.calisti/notebooks/dlds_paper/datasets"
MODELS_DIR = "/home/l.calisti/notebooks/dlds_paper/models"
OUTPUT_DIR = "/home/l.calisti/notebooks/dlds_paper/outputs"
MODELS = ["model3"]
SEEDS = [69]  # [42, 69, 911, 2020, 42069]
WS = [5]  # [3,5,7, 10, 15]
TS = [1, 7]  # [1, 3, 5, 7, 10, 15]
DATASET_NAMES = [
    ("co2_peano_no_weekend.csv", NoWeekLoader()),
    ("pm2p5_peano_no_weekend.csv", NoWeekLoader()),
    ("rad_peano_no_weekend.csv", NoWeekLoader()),
    ("noise_peano_no_weekend.csv", NoWeekLoader()),
    # ("weather.csv", WeatherLoader("T (degC)")),
    # ("weather.csv", WeatherLoader("rh (%)")),
    # ("weather.csv", WeatherLoader("wv (m/s)")),
    # ("weather.csv", WeatherLoader("SWDR (W/m�)")),
    # ("traffic.csv", TrafficLoader()),
    # ("electricity.csv", ElectricityLoader()),
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
                for model_name in MODELS:
                    wc = WindowConfig(ws, ts)
                    ml_predictor = MLPredictor(
                        model_name, MODELS_DIR, ds.name(), wc, seed
                    )
                    validate(
                        dataset=ds,
                        predictor=ml_predictor,
                        window_config=wc,
                        seed=seed,
                        output_path=OUTPUT_DIR,
                    )
                    # dbp_predictor = DBPPredictor(20)
                    # validate(
                    #     dataset=ds,
                    #     predictor=dbp_predictor,
                    #     window_config=WindowConfig(20, 1),
                    #     seed=seed,
                    #     output_path=OUTPUT_DIR,
                    # )
                    # kf_predictor = KFPredictor(x_size=3)
                    # validate(
                    #     dataset=ds,
                    #     predictor=kf_predictor,
                    #     window_config=WindowConfig(3, 1),
                    #     seed=seed,
                    #     output_path=OUTPUT_DIR,
                    # )
