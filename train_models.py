from src.dataset import Dataset
from src.training import train_models
from src.window import WindowConfig

DATASET_DIR = "/home/l.calisti/notebooks/dlds_paper/datasets"
MODELS_DIR = "/home/l.calisti/notebooks/dlds_paper/models"
OUTPUT_DIR = "/home/l.calisti/notebooks/dlds_paper/outputs"
MODELS_PARAM = {
    # "model1": {
    #     "lstm_units": 10,
    #     "epochs": 100,  # 20
    #     "batch_size": 64,
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
        # 'lstm_units': 7,
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
SEEDS = [69]  # 42,69,911,2020,42069]
WS = [5]#[3, 5, 7, 10, 15, 40]
TS = [1]#,5,7,10,15]
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
                train_models(
                    dataset=ds,
                    window_config=WindowConfig(ws, ts),
                    models_path=MODELS_DIR,
                    output_path=OUTPUT_DIR,
                    seed=seed,
                    params=MODELS_PARAM,
                )
