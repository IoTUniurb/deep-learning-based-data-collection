from dataset import Dataset
from utils import printProgressBar
from models import DBPModel, MLModel
import numpy as np
import matplotlib.pyplot as plt
import time


DATASETS_FOLDER = "./dataset-new"
MODELS_FOLDER = "./models-new"
OUT_FOLDER = "./output"
DS = [
    "noweekend/co2_peano_no_weekend.csv",
    # 'noweekend/pm2p5_peano_no_weekend.csv',
    # 'noweekend/rad_peano_no_weekend.csv',
    # 'noweekend/noise_peano_no_weekend.csv'
]

dataset = Dataset(
    DATASETS_FOLDER, DS[0], normalize=True, smooth=True, train_split=3 / 4
)

data = dataset.df_test.values

ml_model = MLModel("model3", 7, 1, dataset, MODELS_FOLDER)
dbp_model = DBPModel(20, 3, 1, 1)

# !mkdir -p "./plots/model3_1_alt/"

# +
ml_buff = data[:20].reshape(1, 20, 1).copy()[:, -7:, :]
dbp_buff = data[:20].reshape(1, 20, 1).copy()
print(ml_buff.shape)
print(dbp_buff.shape)

ml_skip = 0
dbp_skip = 0
ml_counter = 0

for i in range(20, data.shape[0]):
    #     if i > 20:
    #         break
    printProgressBar(i, data.shape[0])

    plt.plot(dataset.inv_scale_data(data[i - 20 : i + 1]), ".--", label="real")
    plt.plot(
        np.arange(13, 20),
        dataset.inv_scale_data(ml_buff.reshape(7, 1)),
        ".-",
        label="ML",
    )
    plt.plot(
        np.arange(0, 20),
        dataset.inv_scale_data(dbp_buff.reshape(20, 1)),
        ".-",
        label="DBP",
    )
    #     print(i)
    real_val = data[i][0]

    ml_pred_val = ml_model.predict(ml_buff)[0][0]
    dbp_pred_val = dbp_model.predict(dbp_buff)[0][0]

    # inverse scale data
    real_val_invscale = dataset.inv_scale_data(real_val)
    ml_pred_val_invscale = dataset.inv_scale_data(ml_pred_val)
    dbp_pred_val_invscale = dataset.inv_scale_data(dbp_pred_val)
    #     print(f'{real_val_invscale     = }')
    #     print(f'{ml_pred_val_invscale  = }')
    #     print(f'{dbp_pred_val_invscale = }')
    # print(f'{real_val_invscale =} {pred_val_invscale =}')

    epsilon = (real_val_invscale * 1) / 100
    # #     print(f"{epsilon=}")

    #     print(f"range [{real_val_invscale-epsilon}, {real_val_invscale+epsilon}]")
    if ml_counter == 0:
        if ml_pred_val_invscale >= (
            real_val_invscale - epsilon
        ) and ml_pred_val_invscale <= (real_val_invscale + epsilon):
            ml_skip += 1
            ml_buff = np.roll(ml_buff, -1)
            ml_buff[:, -1] = ml_pred_val
        else:
            print("ML: send")
            ml_buff = np.roll(ml_buff, -1)
            ml_buff[:, -1] = real_val
            #             for j in range(1,7):
            #                 ml_buff = np.roll(ml_buff,-1)
            #                 ml_buff[:,-1] = data[i+j][0]
            #             i += 7-1
            ml_counter = 7 - 1

    else:
        print("ML: send")
        ml_buff = np.roll(ml_buff, -1)
        ml_buff[:, -1] = real_val

        ml_counter -= 1

    #         ml_buff = np.roll(ml_buff,-1)
    #         p1 = ml_buff[0][0][0]
    #         p2 = real_val
    # #         print(f"{p1=} {p2=}")
    #         omega = (p2-p1)/14
    # #         print(omega)
    #         for j in range(15):
    #             ml_buff[:,j] = p1 + j * omega
    # #         print(ml_buff)
    # #         break
    #         ml_buff[:,-1] = real_val

    if dbp_pred_val_invscale >= (
        real_val_invscale - epsilon
    ) and dbp_pred_val_invscale <= (real_val_invscale + epsilon):
        dbp_skip += 1
        new_val = dbp_pred_val
    else:
        print("DBP: send")
        new_val = real_val
    dbp_buff = np.roll(dbp_buff, -1)
    dbp_buff[:, -1] = new_val

    #     plt.plot(20,real_val,'.',label='real')
    plt.plot(20, ml_pred_val_invscale, ".", label="ML pred")
    plt.plot(20, dbp_pred_val_invscale, ".", label="DBP pred")
    plt.legend()
    plt.title(f"{i = } {ml_skip = } {dbp_skip = }")

    plt.savefig(f"plots/model3_1_alt/{i}.png")
    plt.cla()
# -

print(ml_skip)
print(dbp_skip)
