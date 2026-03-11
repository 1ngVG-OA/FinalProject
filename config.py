"""Central configuration for the forecasting pipeline.

This module defines:
1) project paths,
2) reproducibility seed,
3) dataset-specific settings,
4) hyperparameter grids for MLP and XGBoost.

Notes
-----
- All paths are resolved from the repository root.
- `SERIES_CONFIG` controls data loading and split strategy.
- Hyperparameter grids are consumed by model tuning functions in `Models/`.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datasets"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Global reproducibility seed used across numpy/torch/model training.
SEED = 1234

# Per-series configuration block used by `Mains/main.py`.
#
# Field semantics:
# - csv_path: source dataset path
# - date_col: datetime column to parse and use as index
# - value_col: target numeric column
# - freq: expected temporal frequency (regularization)
# - split: tuple (cut1, cut2) for train/validation/test slicing
# - seasonal: whether to enable seasonal statistical model settings
# - seasonal_period: seasonality period (for example 12 for monthly)
# - diff_order: differencing order used by auto_arima
SERIES_CONFIG = {
    "demographic": {
        "csv_path": DATA_DIR / "demographic.csv",
        "date_col": "date",
        "value_col": "values",
        "freq": "ME",
        "split": (111, 123),
        "seasonal": True,
        "seasonal_period": 12,
        "diff_order": 2,
    },
    "ogrin": {
        "csv_path": DATA_DIR / "ogrin.csv",
        "date_col": "date",
        "value_col": "values",
        "freq": "D",
        "split": (106, 136),
        "seasonal": False,
        "seasonal_period": 7,
        "diff_order": 0,
    },
}

# Hyperparameter search space for the autoregressive MLP model.
MLP_PARAM_GRID = {
    "look_back": [14, 28],
    "hidden_size": [4, 8],
    "lr": [1e-3, 1e-2],
    "activation": ["relu", "tanh"],
    "dropout": [0.0, 0.1],
    "batch_size": [16, 32],
}

# Hyperparameter search space for the autoregressive XGBoost model.
XGB_PARAM_GRID = {
    "look_back": [14, 28],
    "n_estimators": [30, 80],
    "max_depth": [2, 4],
    "eta": [0.1, 0.3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "seed": [SEED],
}

# Maximum epochs for MLP fitting loops (with internal early stopping).
MLP_EPOCHS = 300
