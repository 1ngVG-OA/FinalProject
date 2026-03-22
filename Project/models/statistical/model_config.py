"""Shared configuration and utility functions for statistical models (Step 3).

All items in this module are used by both SarimaRunner and HoltWintersRunner,
keeping the individual model files free from duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Metric utilities
# ---------------------------------------------------------------------------

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE ignoring near-zero denominators."""
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    ape = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(ape) * 100.0)


def _mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error (positive => model over-forecasting)."""
    return float(np.nanmean(y_pred - y_true))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Return RMSE/MAE/MAPE and bias metrics on aligned series."""
    pred = pd.Series(np.asarray(y_pred, dtype=float), index=y_true.index)
    yt = y_true.astype(float).to_numpy()
    yp = pred.astype(float).to_numpy()
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "mape": _safe_mape(yt, yp),
        "mbe": _mean_bias_error(yt, yp),
        "abs_mbe": float(abs(_mean_bias_error(yt, yp))),
    }


def _compute_metrics_aligned(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute metrics after index alignment and NaN filtering."""
    y_true_num = pd.to_numeric(y_true, errors="coerce")
    y_pred_num = pd.to_numeric(y_pred, errors="coerce")
    aligned = pd.concat(
        [y_true_num.rename("y_true"), y_pred_num.rename("y_pred")], axis=1
    ).dropna()
    if aligned.empty:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "mape": float("nan"),
            "mbe": float("nan"),
            "abs_mbe": float("nan"),
        }
    return compute_metrics(aligned["y_true"], aligned["y_pred"])


def _aicc(aic: float, n: int, k: int) -> float:
    """Small-sample corrected AIC."""
    if (n - k - 1) <= 0:
        return float("nan")
    return float(aic + (2.0 * k * (k + 1)) / (n - k - 1))


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatisticalStepConfig:
    """Configuration for the statistical model grid search (SARIMA + HW)."""

    p_values: tuple[int, ...] = (0, 1, 2)
    d_values: tuple[int, ...] = (2,)
    q_values: tuple[int, ...] = (0, 1, 2)
    p_seasonal_values: tuple[int, ...] = (0, 1)
    d_seasonal_values: tuple[int, ...] = (0,)
    q_seasonal_values: tuple[int, ...] = (0, 1)
    seasonal_period: int = 1
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    maxiter: int = 200
    ljung_box_lags: int = 10


# ---------------------------------------------------------------------------
# Split validation helpers
# ---------------------------------------------------------------------------

def validate_split(series: pd.Series, name: str) -> pd.Series:
    """Validate and clean a time-series split."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) < 8:
        raise ValueError(f"{name} split is too short")
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    s.name = "value"
    return s


def validate_original_series(series: pd.Series | None) -> pd.Series | None:
    """Validate and clean the original (untransformed) series, or return None."""
    if series is None:
        return None
    if not isinstance(series, pd.Series):
        raise TypeError("original_series must be a pandas Series or None")
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        return None
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    return s


# ---------------------------------------------------------------------------
# Inverse-transform helpers (log1p + differencing)
# ---------------------------------------------------------------------------

def invert_diff2_log1p(pred_d2: pd.Series, seed_d1: float, seed_log: float) -> pd.Series:
    """Invert double-differenced log1p predictions back to original scale."""
    d1_pred = seed_d1 + pred_d2.cumsum()
    log_pred = seed_log + d1_pred.cumsum()
    return np.expm1(log_pred)


def build_original_scale_context(
    original_series: pd.Series | None,
    use_log1p: bool,
    diff_order: int,
    val_start: Any,
) -> dict[str, Any] | None:
    """Build the seed context needed to invert log1p+diff predictions.

    Returns None if inversion is not applicable (no original series, or
    transformation is not log1p+diff).
    """
    if original_series is None or not use_log1p or diff_order not in (1, 2):
        return None

    raw = pd.to_numeric(original_series, errors="coerce").dropna().astype(float)
    if raw.empty:
        return None

    x_log = np.log1p(raw)

    if diff_order == 1:
        try:
            seed_log_val = float(x_log[x_log.index < val_start].iloc[-1])
        except Exception:
            return None
        return {"raw": raw, "seed_log_val": seed_log_val}

    # diff_order == 2
    x_d1 = x_log.diff().dropna()
    try:
        seed_d1_val = float(x_d1[x_d1.index < val_start].iloc[-1])
        seed_log_val = float(x_log[x_log.index < val_start].iloc[-1])
    except Exception:
        return None
    return {"raw": raw, "seed_d1_val": seed_d1_val, "seed_log_val": seed_log_val}


def validation_original_metrics(
    pred_val: pd.Series,
    orig_context: dict[str, Any] | None,
    diff_order: int,
) -> dict[str, float] | None:
    """Compute validation metrics in original scale when inversion is available."""
    if orig_context is None:
        return None

    if diff_order == 1:
        log_pred = orig_context["seed_log_val"] + pred_val.cumsum()
        pred_orig = np.expm1(log_pred)
    elif diff_order == 2:
        pred_orig = invert_diff2_log1p(
            pred_val,
            orig_context["seed_d1_val"],
            orig_context["seed_log_val"],
        )
    else:
        return None

    true_orig = orig_context["raw"].reindex(pred_orig.index)
    return _compute_metrics_aligned(true_orig, pred_orig)


# ---------------------------------------------------------------------------
# Holt-Winters candidate configurations
# ---------------------------------------------------------------------------

def build_hw_candidate_configs(seasonal_period: int) -> list[dict[str, Any]]:
    """Return the list of HW configurations to evaluate.

    When seasonal_period > 1 additive and multiplicative seasonal variants are
    included; for annual/non-seasonal data only trend variants are evaluated.
    """
    if seasonal_period > 1:
        return [
            {"trend": "add", "seasonal": "add", "damped_trend": False},
            {"trend": "add", "seasonal": "add", "damped_trend": True},
            {"trend": "add", "seasonal": "mul", "damped_trend": False},
            {"trend": "add", "seasonal": "mul", "damped_trend": True},
            {"trend": None, "seasonal": "add", "damped_trend": False},
            {"trend": None, "seasonal": "mul", "damped_trend": False},
        ]
    return [
        {"trend": "add", "seasonal": None, "damped_trend": False},
        {"trend": "add", "seasonal": None, "damped_trend": True},
        {"trend": None, "seasonal": None, "damped_trend": False},
    ]


# ---------------------------------------------------------------------------
# Index frequency utilities
# ---------------------------------------------------------------------------

def infer_seasonal_period_from_index(index: pd.Index) -> int:
    """Infer seasonal period from datetime index frequency.

    Returns conservative defaults when frequency is unknown:
    - monthly  -> 12
    - quarterly -> 4
    - weekly   -> 52
    - daily    -> 7
    - annual/unknown -> 1 (no seasonal component)
    """
    if not isinstance(index, pd.DatetimeIndex):
        return 1

    freq = pd.infer_freq(index)
    if not freq:
        return 1

    freq = freq.upper()
    if freq.startswith("M"):
        return 12
    if freq.startswith("Q"):
        return 4
    if freq.startswith("W"):
        return 52
    if freq.startswith("D"):
        return 7
    if freq.startswith("A") or freq.startswith("Y"):
        return 1
    return 1
