"""XGBoost model for autoregressive forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor


def _create_dataset(series: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(len(series) - look_back):
        x.append(series[i : i + look_back])
        y.append(series[i + look_back])
    return np.array(x), np.array(y)


def _rolling_forecast(model: XGBRegressor, x_train: np.ndarray, steps: int, look_back: int) -> np.ndarray:
    current = x_train[-1].copy()
    preds = []
    for _ in range(steps):
        p = model.predict(current.reshape(1, look_back))[0]
        preds.append(p)
        current = np.roll(current, -1)
        current[-1] = p
    return np.array(preds)


def tune_and_forecast(train: pd.Series, val: pd.Series, test: pd.Series, param_grid: dict) -> dict:
    best = {"rmse": float("inf"), "params": None, "model": None, "val_pred": None}

    for params in ParameterGrid(param_grid):
        look_back = int(params["look_back"])
        x_train, y_train = _create_dataset(train.values, look_back)
        model_params = {k: v for k, v in params.items() if k != "look_back"}
        model = XGBRegressor(objective="reg:squarederror", verbosity=0, **model_params)
        model.fit(x_train, y_train)

        val_pred = _rolling_forecast(model, x_train, len(val), look_back)
        rmse = float(np.sqrt(np.mean((val.values - val_pred) ** 2)))

        if rmse < best["rmse"]:
            best = {"rmse": rmse, "params": params, "model": model, "val_pred": val_pred}

    # Refit with best params on train+val.
    best_params = best["params"]
    look_back = int(best_params["look_back"])
    train_val = pd.concat([train, val])
    x_tv, y_tv = _create_dataset(train_val.values, look_back)
    model_params = {k: v for k, v in best_params.items() if k != "look_back"}
    final_model = XGBRegressor(objective="reg:squarederror", verbosity=0, **model_params)
    final_model.fit(x_tv, y_tv)
    test_pred = _rolling_forecast(final_model, x_tv, len(test), look_back)

    return {
        "name": "XGBoost",
        "best_params": dict(best_params),
        "validation_pred": best["val_pred"],
        "test_pred": test_pred,
    }
