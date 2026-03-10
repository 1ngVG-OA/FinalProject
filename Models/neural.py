"""MLP model for autoregressive forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from utils import seed_everything


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, activation: str, dropout: float) -> None:
        super().__init__()
        act = {"relu": nn.ReLU(), "tanh": nn.Tanh()}.get(activation, nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _create_dataset(series: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(len(series) - look_back):
        x.append(series[i : i + look_back])
        y.append(series[i + look_back])
    return np.array(x), np.array(y)


def _fit_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int = 10,
) -> None:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    no_improve = 0

    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)

        avg_loss = total / len(loader.dataset)
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break


def _recursive_forecast(model: nn.Module, window: torch.Tensor, steps: int) -> np.ndarray:
    model.eval()
    preds = []
    current = window.clone()
    for _ in range(steps):
        with torch.no_grad():
            pred = model(current.unsqueeze(0)).item()
        preds.append(pred)
        current = torch.cat([current[1:], torch.tensor([pred], dtype=torch.float32)])
    return np.array(preds)


def tune_and_forecast(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
    param_grid: dict,
    epochs: int,
    seed: int,
) -> dict:
    seed_everything(seed)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).ravel()
    val_scaled = scaler.transform(val.values.reshape(-1, 1)).ravel()

    best = {"rmse": float("inf"), "params": None, "val_pred": None}

    for params in ParameterGrid(param_grid):
        look_back = int(params["look_back"])
        x_train, y_train = _create_dataset(train_scaled, look_back)
        x_t = torch.tensor(x_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        model = MLP(look_back, int(params["hidden_size"]), str(params["activation"]), float(params["dropout"]))
        _fit_model(
            model,
            x_t,
            y_t,
            epochs=epochs,
            lr=float(params["lr"]),
            batch_size=int(params["batch_size"]),
        )

        init_window = torch.tensor(train_scaled[-look_back:], dtype=torch.float32)
        val_pred_scaled = _recursive_forecast(model, init_window, len(val_scaled))
        val_pred = scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
        rmse = float(np.sqrt(np.mean((val.values - val_pred) ** 2)))

        if rmse < best["rmse"]:
            best = {"rmse": rmse, "params": params, "val_pred": val_pred}

    # Refit on train+validation with best params and forecast test.
    best_params = best["params"]
    train_val = pd.concat([train, val])
    scaler_final = StandardScaler()
    train_val_scaled = scaler_final.fit_transform(train_val.values.reshape(-1, 1)).ravel()
    test_scaled = scaler_final.transform(test.values.reshape(-1, 1)).ravel()

    look_back = int(best_params["look_back"])
    x_tv, y_tv = _create_dataset(train_val_scaled, look_back)
    x_tv_t = torch.tensor(x_tv, dtype=torch.float32)
    y_tv_t = torch.tensor(y_tv, dtype=torch.float32).unsqueeze(1)

    model_final = MLP(look_back, int(best_params["hidden_size"]), str(best_params["activation"]), float(best_params["dropout"]))
    _fit_model(
        model_final,
        x_tv_t,
        y_tv_t,
        epochs=epochs,
        lr=float(best_params["lr"]),
        batch_size=int(best_params["batch_size"]),
    )

    init_window = torch.tensor(train_val_scaled[-look_back:], dtype=torch.float32)
    test_pred_scaled = _recursive_forecast(model_final, init_window, len(test_scaled))
    test_pred = scaler_final.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()

    return {
        "name": "MLP",
        "best_params": dict(best_params),
        "validation_pred": best["val_pred"],
        "test_pred": test_pred,
    }
