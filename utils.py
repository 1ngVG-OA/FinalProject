"""Shared utilities for plotting, metrics and reproducibility."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def ensure_directories(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def evaluate_metrics(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def run_adf(series: pd.Series) -> dict:
    stat, pvalue, *_ = adfuller(series.dropna())
    return {"adf_stat": float(stat), "pvalue": float(pvalue)}


def plot_acf_pacf(series: pd.Series, title: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), ax=axes[0])
    axes[0].set_title("ACF")
    plot_pacf(series.dropna(), ax=axes[1])
    axes[1].set_title("PACF")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_predictions(
    full_series: pd.Series,
    val_pred: pd.Series,
    test_pred: pd.Series,
    title: str,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(full_series.index, full_series.values, label="series", alpha=0.7)
    plt.plot(val_pred.index, val_pred.values, label="validation forecast")
    plt.plot(test_pred.index, test_pred.values, label="test forecast")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def to_series(values: np.ndarray | list[float], index: pd.Index) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=index)
