import json
import numpy as np
import pandas as pd
from joblib import load as joblib_load
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
    def __init__(self, forecast_num, test_file=None):
        base_path = "TimeseriesModel"
        model_h5_file = f"{base_path}/model.h5"
        model_json_file = f"{base_path}/model.json"
        last_values_file = f"{base_path}/last_values.npy"
        lags_file = f"{base_path}/lags.npy"
        self.forecast_num = forecast_num
        self.test_file = test_file
        if self.test_file is not None:
            if self.test_file.name.endswith(".csv"):
                self.test_data = pd.read_csv(self.test_file)
            else:
                self.test_data = pd.read_excel(self.test_file)

        self.model = load_model(model_h5_file)

        with open(model_json_file) as f:
            params = json.loads(f.read())
        self.last = np.load(last_values_file)
        self.lags = np.load(lags_file)

        self.scale_type = params.get("scale_type")
        if self.scale_type != "None":
            self.feature_scaler = joblib_load(f"{base_path}/feature_scaler.pkl")
            self.label_scaler = joblib_load(f"{base_path}/label_scaler.pkl")

        self.difference_choice = params.get("difference_choice", 0)
        if self.difference_choice == 1:
            self.interval = params["interval"]
            self.fill_values = np.load(f"{base_path}/fill.npy")

        self.s_difference_choice = params.get("second_difference_choice", 0)
        if self.s_difference_choice == 1:
            self.s_interval = params["second_interval"]
            self.s_fill_values = np.load(f"{base_path}/s_fill.npy")

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
        
        self.pred = self.pred.round(2).ravel()


    def plot_prediction(self):
        fig, ax = plt.subplots()
        df = pd.DataFrame(
            data=self.pred, columns=["Predict"], index=range(1, len(self.pred) + 1)
        )
        if self.test_file is not None:
            y_test = self.test_data[self.label_name]
            y_test = np.asarray(y_test)[: self.forecast_num]
            ax.plot(y_test, label=["Real"])
            df["Real"] = y_test
        ax.plot(self.pred, label=["Prediction"])
        return df, fig

    def get_loss(self):
        if self.test_file is not None:
            y_test = self.test_data[self.label_name]
            y_test = np.asarray(y_test)[: self.forecast_num]

            return dict(
                zip(["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"], loss(y_test, self.pred))
            )


class Supervised:
    def __init__(self, forecast_num, test_file):
        base_path = "SupervisedModel"
        model_joblib_file = f"{base_path}/model.joblib"
        model_json_file = f"{base_path}/model.json"
        last_values_file = f"{base_path}/last_values.npy"

        self.forecast_num = forecast_num
        if test_file.name.endswith(".csv"):
            self.test_df = pd.read_csv(test_file)
        else:
            self.test_df = pd.read_excel(test_file)

        self.model = joblib_load(model_joblib_file)

        with open(model_json_file) as f:
            params = json.loads(f.read())

        self.scale_type = params.get("scale_type")
        if self.scale_type != "None":
            self.feature_scaler = joblib_load(f"{base_path}/feature_scaler.pkl")
            self.label_scaler = joblib_load(f"{base_path}/label_scaler.pkl")

        self.sliding = params.get("sliding", -1)
        self.lookback_option = params.get("lookback_option", 0)
        if self.lookback_option:
            self.lookback_value = params.get("lookback_value", 0)
            last_values_file = f"{base_path}/last_values.npy"
            self.last = np.load(last_values_file)

        self.seasonal_lookback_option = params.get("seasonal_lookback_option", 0)
        if self.seasonal_lookback_option == 1:
            self.seasonal_period = params.get("seasonal_period", 0)
            self.seasonal_value = params.get("seasonal_value", 0)
            seasonal_last_values_file = f"{base_path}/seasonal_last_values.npy"
            self.seasonal_last = np.load(seasonal_last_values_file)

        self.predictor_names = params["predictor_names"]
        self.label_name = params["label_name"]
        self.is_round = params.get("is_round", True)
        self.is_negative = params.get("is_negative", False)

    def forecastLookback(
        self, num, lookback=0, seasons=0, seasonal_lookback=0, sliding=-1
    ):
        pred = []
        if sliding == 0:
            last = self.last
            for i in range(num):
                X_test = self.test_df[self.predictor_names].iloc[i]
                if self.scale_type != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(
                        X_test.values.reshape(1, -1)
                    ).reshape(-1)
                for j in range(1, lookback + 1):
                    X_test[f"t-{j}"] = last[-j]
                to_pred = X_test.to_numpy().reshape(1, -1)
                out = self.model.predict(to_pred)
                last = np.append(last, out)[-lookback:]
                pred.append(out)

        elif sliding == 1:
            seasonal_last = self.seasonal_last
            for i in range(num):
                X_test = self.test_df[self.predictor_names].iloc[i]
                if self.scale_type != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(
                        X_test.values.reshape(1, -1)
                    ).reshape(-1)
                for j in range(1, seasons + 1):
                    X_test[f"t-{j*seasonal_last}"] = seasonal_last[
                        -j * seasonal_lookback
                    ]
                to_pred = X_test.to_numpy().reshape(1, -1)
                out = self.model.predict(to_pred)
                seasonal_last = np.append(seasonal_last, out)[1:]
                pred.append(out)

        elif sliding == 2:
            last = self.last
            seasonal_last = self.seasonal_last
            for i in range(num):
                X_test = self.test_df[self.predictor_names].iloc[i]
                if self.scale_type != "None":
                    X_test.iloc[:] = self.feature_scaler.transform(
                        X_test.values.reshape(1, -1)
                    ).reshape(-1)
                for j in range(1, lookback + 1):
                    X_test[f"t-{j}"] = last[-j]
                for j in range(1, seasons + 1):
                    X_test[f"t-{j*seasonal_lookback}"] = seasonal_last[
                        -j * seasonal_lookback
                    ]
                to_pred = X_test.to_numpy().reshape(1, -1)
                out = self.model.predict(to_pred)
                last = np.append(last, out)[-lookback:]
                seasonal_last = np.append(seasonal_last, out)[1:]
                pred.append(out)

        return np.array(pred).reshape(-1)

    def forecast(self):
        num = self.forecast_num
        lookback_option = self.lookback_option
        seasonal_lookback_option = self.seasonal_lookback_option
        X_test = self.test_df[self.predictor_names][:num].to_numpy()

        if lookback_option == 0 and seasonal_lookback_option == 0:
            if self.scale_type != "None":
                X_test = self.feature_scaler.transform(X_test)
            self.pred = self.model.predict(X_test).reshape(-1)
        else:
            sliding = self.sliding
            lookback = self.lookback_value
            seasonal_lookback = self.seasonal_value
            seasons = self.seasonal_period

            self.pred = self.forecastLookback(
                num, lookback, seasons, seasonal_lookback, sliding
            )

        if self.scale_type != "None":
            self.pred = self.label_scaler.inverse_transform(
                self.pred.reshape(-1, 1)
            ).reshape(-1)

        if not self.is_negative:
            self.pred = self.pred.clip(0, None)
        if self.is_round:
            self.pred = np.round(self.pred).astype(int)

        self.pred = self.pred.round(2).ravel()

    def plot_prediction(self):
        fig, ax = plt.subplots()
        df = pd.DataFrame(
            data=self.pred, columns=["Predict"], index=range(1, len(self.pred) + 1)
        )
        if self.label_name in self.test_df:
            y_test = (
                self.test_df[self.label_name]
                .iloc[: self.forecast_num]
                .to_numpy()
                .reshape(-1)
            )
            ax.plot(y_test, label=["Real"])
            df["Real"] = y_test
        ax.plot(self.pred, label=["Prediction"])
        return df, fig

    def get_loss(self):
        if self.label_name in self.test_df:
            y_test = (
                self.test_df[self.label_name]
                .iloc[: self.forecast_num]
                .to_numpy()
                .reshape(-1)
            )
            return dict(
                zip(["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"], loss(y_test, self.pred))
            )


def renew_last_values(data):
    base_path = "TimeseriesModel"
    model_json_file = f"{base_path}/model.json"
    with open(model_json_file) as f:
        params = json.loads(f.read())
    
    lag = int(params["lag_number"].split(",")[-1]) + 1
    data = data.iloc[-lag:].values
    scale_type = params.get("scale_type")
    if scale_type != "None":
        scaler = joblib_load(f"{base_path}/label_scaler.pkl")
        data = scaler.transform(data.reshape(-1, 1))

    np.save(f"{base_path}/last_values.npy", data)

def renew_lookback_values(data):
    base_path = "SupervisedModel"
    model_json_file = f"{base_path}/model.json"
    with open(model_json_file) as f:
        params = json.loads(f.read())
    sliding = params.get("sliding", -1)
    lookback_value = params.get("lookback_value", 0)
    seasonal_value = params.get("seasonal_value", 0)

    if sliding == 0:
        data = data.iloc[-lookback_value:].values
    elif sliding == 1:
        return None
    elif sliding == 2:
        data = data[-(lookback_value + seasonal_value) : -seasonal_value].values
    
    scale_type = params.get("scale_type")
    if scale_type != "None":
        scaler = joblib_load(f"{base_path}/label_scaler.pkl")
        data = scaler.transform(data.reshape(-1, 1))

    np.save(f"{base_path}/last_values.npy", data)


def renew_seasonal_lookback_values(data):
    base_path = "SupervisedModel"
    model_json_file = f"{base_path}/model.json"
    with open(model_json_file) as f:
        params = json.loads(f.read())
    sliding = params.get("sliding", -1)
    seasonal_value = params.get("seasonal_value", 0)
    seasonal_period = params.get("seasonal_period", 0)

    if sliding == 0:
        return None
    data = data[-seasonal_value * seasonal_period :].values

    scale_type = params.get("scale_type")
    if scale_type != "None":
        scaler = joblib_load(f"{base_path}/label_scaler.pkl")
        data = scaler.transform(data.reshape(-1, 1))
    
    np.save(f"{base_path}/seasonal_last_values.npy", data)
