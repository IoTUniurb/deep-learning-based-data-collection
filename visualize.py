from src.dataset import Dataset, to_supervised
from src.dataset_loader import WeatherLoader, TrafficLoader, ElectricityLoader
from src.ml_model import get_model_path
from src.window import WindowConfig
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_DIR = "/home/l.calisti/notebooks/dlds_paper/datasets"
MODELS_DIR = "/home/l.calisti/notebooks/dlds_paper/models"
OUTPUT_DIR = "/home/l.calisti/notebooks/dlds_paper/outputs"
DATASET_NAMES = [
    # ("co2_peano_no_weekend.csv", NoWeekLoader()),
    # ("pm2p5_peano_no_weekend.csv", NoWeekLoader()),
    # ("rad_peano_no_weekend.csv", NoWeekLoader()),
    # ("noise_peano_no_weekend.csv", NoWeekLoader()),
    # ("weather.csv", WeatherLoader("T (degC)")),
    # ("weather.csv", WeatherLoader("rh (%)")),
    # ("weather.csv", WeatherLoader("wv (m/s)")),
    # ("weather.csv", WeatherLoader("SWDR (W/m�)")),
    # ("traffic.csv", TrafficLoader()),
    ("electricity.csv", ElectricityLoader()),
]
SEED = 69
dbp_window_size = 20
ml_window_size = 48
time_steps = 2
window_config = WindowConfig(ml_window_size, time_steps)
print(f"{dbp_window_size = }")
print(f"{window_config  = }")

ds = Dataset(
    name=DATASET_NAMES[0][0],
    base_path=DATASET_DIR,
    loader=DATASET_NAMES[0][1],
    smooth=None,
)
data = ds.values()

print(f"{data=}")
print(f"{data.shape=}")
print(f"{np.min(data)=} {np.max(data)=}")
print(f"{np.mean(data)=}")

d = DATASET_NAMES[0][1].load(f"{DATASET_DIR}/weather.csv").values
np.where(np.abs(d) <= 0.01)

# +
th = 0.01
# data[np.where(np.abs(data) < th)[0]]=th
less = data[np.where(np.abs(data) <= th)[0]]

print(f"{less.shape=}")
print(f"{less=}")
# -

train_data, test_data = ds.train_test_split(type="random", seed=SEED)
train_data = train_data.reshape((train_data.shape[0] * train_data.shape[1], 1))
test_data = test_data.reshape((test_data.shape[0] * test_data.shape[1], 1))
print(f"{train_data.shape=} {test_data.shape=}")

train_sup_x, train_sup_y = to_supervised(train_data, window_config)
test_sup_x, test_sup_y = to_supervised(test_data, window_config)
x_train, x_test, y_train, y_test = train_test_split(
    train_sup_x,
    train_sup_y,
    random_state=SEED,
    shuffle=True,
    train_size=0.80,
)
print(f"{x_train.shape=} {y_train.shape=}")
print(f"{x_test.shape=} {y_test.shape=}")

plt.subplot(2, 1, 1)
plt.plot(train_data)
plt.subplot(2, 1, 2)
plt.plot(test_data)

model_path = get_model_path(
    model_name="model3",
    dataset_name=ds.name(),
    model_path=MODELS_DIR,
    seed=SEED,
    window_config=window_config,
)
model = keras.models.load_model(model_path)

print(f"{model.evaluate(test_sup_x, test_sup_y,verbose=0)=}")
print(f"{model.evaluate(train_sup_x, train_sup_y,verbose=0)=}")
print(f"{model.evaluate(x_test, y_test,verbose=0)=}")
# pred_y = model.predict(test_sup_x)

# +
from sklearn.metrics import mean_absolute_percentage_error

idx = np.where(test_sup_y >= 1000)[0]
pred = model.predict(test_sup_x[idx]).reshape(-1)
mean_absolute_percentage_error(test_sup_y[idx], pred)
# -


plt.plot(test_data[1010:1070])
pred = model.predict(test_data[1010:1058].reshape(1, 48, 1)).reshape(2)
print(pred)
plt.scatter(np.arange(49, 51), pred)
plt.scatter(np.arange(0, 48), test_data[1010:1058])
plt.scatter(np.arange(49, 51), test_data[1059:1061])


# +
# plt.subplot(2, 1, 1)
# plt.scatter(np.arange(5),x_test[0])
# plt.subplot(2, 1, 2)
# plt.scatter(0,y_test[0])
y_pred = model.predict(x_test).reshape(-1)
print(y_pred.shape)
from src.utils import compute_metrics

print(y_test.reshape(-1))
print(y_pred.reshape(-1))

compute_metrics(y_test, y_pred)
# -

lim = 3840
step = 10
plt.subplot(2, 1, 1)
plt.plot(y_test[lim : lim + step])
plt.subplot(2, 1, 2)
plt.plot(y_pred[lim : lim + step])
print(compute_metrics(y_test[lim : lim + step], y_pred[lim : lim + step]))
print(y_test[lim : lim + step])
print(f"{np.max(y_test)=} {np.max(y_pred)=}")
print(f"{np.min(y_test)=} {np.min(y_pred)=}")
print(f"{np.min(np.abs(y_test))=} {np.min(y_pred)=}")

i = np.where(np.abs(y_train) < 0.01)[0]
y_train[i]

# take a part of the dataset
# window = df_train[184:214]
window = ds.df_test[0:30]
# window = df_test[5134:5165]

data = Dataset.df_to_time_steps(window, dbp_window_size, time_steps).values
buff, real_vals = Dataset.to_supervised(data, dbp_window_size, time_steps)
print(f"{buff.shape      = }")
print(f"{real_vals.shape = }")

# ## Predict


# +
def dbp_predict(buffer):
    sigma = (np.mean(buffer[:, -3:], axis=1) - np.mean(buffer[:, :3], axis=1)) / (
        dbp_window_size - 1
    )
    ret = np.zeros((buffer.shape[0], time_steps, 1))
    for ts in range(time_steps):
        ret[:, ts] = np.mean(buffer[:, -3:], axis=1) + ((ts + 1) * sigma)
    ret = ret.reshape(-1, time_steps)
    return ret


def ml_predict(model, buffer):
    preds = []
    b = buffer.copy()
    for i in range(time_steps):
        pred = model.predict(b)[0][0]
        preds.append(pred)
        b = np.roll(b, -1)
        b[:, -1] = pred
    #         b[:,-1] = real_vals[0][i]
    return np.array(preds)


# -
# dbp_model = DBPModel(dbp_window_size,3,None,time_steps)
dbp_preds = dbp_predict(buff[0].reshape((1, dbp_window_size, 1)))[0]
avg_points = [np.mean(buff[0, :3]), np.mean(buff[0, -3:])]
print(dbp_preds)
print(avg_points)

# +
ml_model1 = MLModel("model1", ml_window_size, time_steps, ds, MODELS_FOLDER)
# ml_model1_preds = ml_predict(ml_model1,buff[0,-ml_window_size:].reshape(1,ml_window_size,1))
ml_model1_preds = ml_model1.predict(
    buff[0, -ml_window_size:].reshape(1, ml_window_size, 1)
)[0]
print(f"{ml_model1_preds = }")

ml_model2 = MLModel("model2", ml_window_size, time_steps, ds, MODELS_FOLDER)
# ml_model2_preds = ml_predict(ml_model2,buff[0,-ml_window_size:].reshape(1,ml_window_size,1))
ml_model2_preds = ml_model2.predict(
    buff[0, -ml_window_size:].reshape(1, ml_window_size, 1)
)[0]
print(f"{ml_model2_preds = }")

ml_model3 = MLModel("model3", ml_window_size, time_steps, ds, MODELS_FOLDER)
# ml_model3_preds = ml_predict(ml_model3,buff[0,-ml_window_size:].reshape(1,ml_window_size,1))
ml_model3_preds = ml_model3.predict(
    buff[0, -ml_window_size:].reshape(1, ml_window_size, 1)
)[0]
print(f"{ml_model3_preds = }")

# +
real_vals_invscale = ds.inv_scale_data(real_vals)

dbp_preds_invscale = ds.inv_scale_data(dbp_preds)
avg_points_invscale = ds.inv_scale_data(np.array(avg_points))

ml_model1_preds_invscale = ds.inv_scale_data(ml_model1_preds)
ml_model2_preds_invscale = ds.inv_scale_data(ml_model2_preds)
ml_model3_preds_invscale = ds.inv_scale_data(ml_model3_preds)

window_invscale = ds.inv_scale_data(window)
# -

preds = np.array(dbp_preds_invscale)  # .reshape(7,1)
mae = mean_absolute_error(real_vals_invscale[0], preds)
mape = mean_absolute_percentage_error(real_vals_invscale[0], preds)
print(f"{mae=} {mape=}")

# +
mae = mean_absolute_error(real_vals_invscale[0], ml_model1_preds_invscale)
mape = mean_absolute_percentage_error(real_vals_invscale[0], ml_model1_preds_invscale)
print(f"model 1 {mae=} {mape=}")

mae = mean_absolute_error(real_vals_invscale[0], ml_model2_preds_invscale)
mape = mean_absolute_percentage_error(real_vals_invscale[0], ml_model2_preds_invscale)
print(f"model 2 {mae=} {mape=}")

mae = mean_absolute_error(real_vals_invscale[0], ml_model3_preds_invscale)
mape = mean_absolute_percentage_error(real_vals_invscale[0], ml_model3_preds_invscale)
print(f"model 3 {mae=} {mape=}")
# -

# plot some of the data along with the predictions of the DBP
plt.plot(np.arange(0, len(window)), window_invscale, ".-")
plt.plot([0, dbp_window_size - 1], avg_points_invscale, ".-r")
plt.plot(
    np.arange(dbp_window_size, len(dbp_preds) + dbp_window_size),
    dbp_preds_invscale,
    ".--r",
    label="DBP",
)
plt.plot(
    np.arange(dbp_window_size, len(ml_model1_preds) + dbp_window_size),
    ml_model1_preds_invscale,
    ".--",
    label="model1",
)
plt.plot(
    np.arange(dbp_window_size, len(ml_model2_preds) + dbp_window_size),
    ml_model2_preds_invscale,
    ".--",
    label="model2",
)
plt.plot(
    np.arange(dbp_window_size, len(ml_model3_preds) + dbp_window_size),
    ml_model3_preds_invscale,
    ".--",
    label="model3",
)
plt.legend()

full_data = Dataset.df_to_time_steps(ds.df_test, dbp_window_size, time_steps).values
full_buff, full_real_vals = Dataset.to_supervised(
    full_data, dbp_window_size, time_steps
)

# +
full_dbp_preds = dbp_predict(full_buff)
full_model3_preds = ml_model3.predict(full_buff[:, -ml_window_size:, :])

full_dbp_preds = full_dbp_preds.reshape(-1)[1000:1020]
full_model3_preds = full_model3_preds.reshape(-1)[1000:1020]

plt.plot(ds.df_test.values[1000:1040], "r")
plt.plot(
    np.arange(dbp_window_size, len(full_dbp_preds) + dbp_window_size),
    full_dbp_preds,
    label="DBP",
)
plt.plot(
    np.arange(dbp_window_size, len(full_model3_preds) + dbp_window_size),
    full_model3_preds,
    label="model3",
)
plt.legend()
