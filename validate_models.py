from src.dataset import Dataset
from src.window import WindowConfig
from src.validate import validate
from src.predictors.ml_predictor import MLPredictor
from src.predictors.dbp_predictor import DBPPredictor
from src.predictors.kf_predictor import KFPredictor

DATASET_DIR = "/home/l.calisti/notebooks/dlds_paper/datasets"
MODELS_DIR = "/home/l.calisti/notebooks/dlds_paper/models"
OUTPUT_DIR = "/home/l.calisti/notebooks/dlds_paper/outputs"
MODELS = ["model1", "model2"]
SEEDS = [69]  # 42,69,911,2020,42069]
WS = [5]  # 3,5,7,10,15]#,20,25]
TS = [1]  # [3,5,7,10,15]
DATASET_NAMES = [
    "co2_peano_no_weekend.csv",
    # "pm2p5_peano_no_weekend.csv",
    # "rad_peano_no_weekend.csv",
    # "noise_peano_no_weekend.csv",
]

for dataset_name in DATASET_NAMES:
    ds = Dataset(dataset_name, base_path=DATASET_DIR)
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

# for dataset_name in DATASET_NAMES:
#     ds = Dataset(dataset_name, base_path=DATASET_DIR)
#     for seed in SEEDS:
#         for ws in WS:
#             for ts in TS:
#                 for model_name in MODELS:
#                     wc = WindowConfig(ws, ts)
#                     ml_predictor = MLPredictor(
#                         model_name, MODELS_DIR, ds.name(), wc, seed
#                     )
#                     validate(
#                         dataset=ds,
#                         predictor=ml_predictor,
#                         window_config=wc,
#                         seed=seed,
#                         output_path=OUTPUT_DIR,
#                     )
#                 dbp_predictor = DBPPredictor(20)
#                 validate(
#                     dataset=ds,
#                     predictor=dbp_predictor,
#                     window_config=WindowConfig(20, 1),
#                     seed=seed,
#                     output_path=OUTPUT_DIR,
#                 )
#                 kf_predictor = KFPredictor(x_size=3)
#                 validate(
#                     dataset=ds,
#                     predictor=kf_predictor,
#                     window_config=WindowConfig(3, 1),
#                     seed=seed,
#                     output_path=OUTPUT_DIR,
#                 )
