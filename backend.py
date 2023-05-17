import io
import os
import json
import numpy as np
import pandas as pd
from pickle import load as pickle_load
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def loss(y_true, y_pred, seasons=1):
    NMSE = round((((y_true - y_pred) ** 2) / (y_true.mean() * y_pred.mean())).mean(), 2)
    RMSE = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
    MAE = round(mean_absolute_error(y_true, y_pred), 2)
    try:
        MAPE = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)
    except Exception:
        MAPE = None
    try:
        SMAPE = round(
            np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))
            * 100,
            2,
        )
    except Exception:
        SMAPE = None

    return [NMSE, RMSE, MAE, MAPE, SMAPE]


class Timeseries:
    def __init__(self, files, forecast_num, test_file=None):
        names = [i.name for i in files]
        model_h5_file = files[names.index("model.h5")]
        model_json_file = files[names.index("model.json")]
        last_values_file = files[names.index("last_values.npy")]
        lags_file = files[names.index("lags.npy")]
        self.forecast_num = forecast_num
        if test_file.name.endswith(".csv"):
            self.test_file = pd.read_csv(test_file)
        else:
            self.test_file = pd.read_excel(test_file)

        with open("model.h5", "wb") as model_path:
            model_path.write(model_h5_file.getvalue())
        self.model = load_model("model.h5")
        os.remove("model.h5")

        params = json.load(model_json_file)
        self.params = params
        self.last = np.load(last_values_file)
        self.lags = np.load(lags_file)

        self.scale_type = params.get("scale_type")
        if self.scale_type != "None":
            self.feature_scaler = pickle_load(files[names.index("feature_scaler.pkl")])
            self.label_scaler = pickle_load(files[names.index("label_scaler.pkl")])

        self.difference_choice = params.get("difference_choice", 0)
        if self.difference_choice == 1:
            self.interval = params["interval"]
            self.fill_values = np.load(files[names.index("fill.npy")])

        self.s_difference_choice = params.get("second_difference_choice", 0)
        if self.s_difference_choice == 1:
            self.s_interval = params["second_interval"]
            self.s_fill_values = np.load(files[names.index("s_fill.npy")])

        self.predictor_names = params["predictor_names"]
        self.label_name = params["label_name"]
        self.is_round = params.get("is_round", True)
        self.is_negative = params.get("is_negative", False)

    def difference(self, data, diff, interval=None, fill_values=None):
        if diff:
            return np.array(
                [data[i] - data[i - interval] for i in range(interval, len(data))]
            )
        else:
            for i in range(len(data)):
                if i >= interval:
                    data[i] = data[i] + data[i - interval]
                else:
                    data[i] = data[i] + fill_values[(len(fill_values) - interval) + i]

    def forecast(self):
        input_value = self.last
        steps, features = input_value.shape[0], input_value.shape[1]
        shape = (1, steps, features)
        pred = []

        for _ in range(self.forecast_num):
            output = self.model.predict(
                input_value.reshape(shape)[:, self.lags], verbose=0
            )
            pred = np.append(pred, output)
            input_value = np.append(input_value, output)[-shape[1] :]

        self.pred = np.array(pred).reshape(-1, 1)

        if self.s_difference_choice:
            self.difference(self.pred, False, self.s_interval, self.s_fill_values)

        if self.difference_choice:
            self.difference(self.pred, False, self.interval, self.fill_values)

        if self.scale_type != "None":
            self.pred = self.label_scaler.inverse_transform(self.pred)

        if not self.is_negative:
            self.pred = self.pred.clip(0, None)
        if self.is_round:
            self.pred = np.round(self.pred).astype(int)

    def plot_prediction(self):
        fig, ax = plt.subplots()
        if self.test_file is not []:
            y_test = self.test_file[self.label_name]
            y_test = np.asarray(y_test)[: self.forecast_num]
            ax.plot(y_test, label=["Real"])
        ax.plot(self.pred, label=["Prediction"])
        return fig

    def get_loss(self):
        if self.test_file is not []:
            y_test = self.test_file[self.label_name]
            y_test = np.asarray(y_test)[: self.forecast_num]

            return dict(
                zip(["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"], loss(y_test, self.pred))
            )


def renew_last_values(lag_file, data, scaler):
    old_lags = np.load(lag_file)
    data = data.iloc[-len(old_lags) :].values
    if scaler is not None:
        scaler = pickle_load(scaler)
        data = scaler.transform(data.reshape(-1, 1)).ravel()

    buffer = io.BytesIO()
    np.save(buffer, data)
    return buffer
