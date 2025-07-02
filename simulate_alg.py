from src.dataset import Dataset
from src.window import WindowConfig
from src.techniques import dlbdc
from src.techniques import dlds
from src.predictors.ml_predictor import MLPredictor
from src.predictors.dbp_predictor import DBPPredictor
from src.predictors.kf_predictor import KFPredictor

DATASET_DIR = "/home/l.calisti/notebooks/dlds_paper/datasets"
MODELS_DIR = "/home/l.calisti/notebooks/dlds_paper/models"
OUTPUT_DIR = "/home/l.calisti/notebooks/dlds_paper/outputs"
SEEDS = [69]  # 42,69,911,2020,42069]
WS = [5]  # 3,5,7,10,15]#,20,25]
TS = [3, 5, 7, 10, 15]
ERRORS = [1, 3, 5, 7, 10, 25]
ALPHAS = [0.5, 1.0, 1.0, 1.0]  # [0.25, 0.40, 0.50, 0.75, 0.90, 1.0]
DATASET_NAMES = [
    "co2_peano_no_weekend.csv",
    "pm2p5_peano_no_weekend.csv",
    "rad_peano_no_weekend.csv",
    "noise_peano_no_weekend.csv",
]

for dataset_name in DATASET_NAMES:
    ds = Dataset(dataset_name, base_path=DATASET_DIR)
    for seed in SEEDS:
        for ws in WS:
            for ts in TS:
                for error in ERRORS:
                    wc = WindowConfig(ws, ts)
                    predictor = MLPredictor(
                        "model3",
                        MODELS_DIR,
                        ds.name(),
                        wc,
                        seed,
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

# for dataset_name, alpha in zip(DATASET_NAMES, ALPHAS):
#     ds = Dataset(dataset_name, base_path=DATASET_DIR)
#     for seed in SEEDS:
#         for ws in WS:
#             for ts in TS:
#                 for error in ERRORS:
#                     wc = WindowConfig(ws, ts)
#                     predictor = MLPredictor(
#                         "model3",
#                         MODELS_DIR,
#                         ds.name(),
#                         wc,
#                         seed,
#                     )
#                     dlbdc.simulate(
#                         dataset=ds,
#                         output_path=OUTPUT_DIR,
#                         predictor=predictor,
#                         window_config=wc,
#                         error=error,
#                         seed=seed,
#                         realign="scaled-distance",
#                         alpha=alpha,
#                     )
#                     predictor = DBPPredictor(20)
#                     simulate(
#                         dataset=ds,
#                         output_path=OUTPUT_DIR,
#                         predictor=predictor,
#                         window_config=WindowConfig(20, 1),
#                         error=error,
#                         seed=seed,
#                         realign="simple-append",
#                     )
#                     predictor = KFPredictor(3)
#                     simulate(
#                         dataset=ds,
#                         output_path=OUTPUT_DIR,
#                         predictor=predictor,
#                         window_config=WindowConfig(3, 1),
#                         error=error,
#                         seed=seed,
#                         realign="simple-append",
#                     )
