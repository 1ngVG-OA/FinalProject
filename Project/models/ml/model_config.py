"""Configurazione e utility condivise per lo Step 4 (ML non neurale)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ------------------------------------------------------------------
# Utility metriche
# ------------------------------------------------------------------

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcola il MAPE ignorando denominatori prossimi a zero."""
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    ape = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(ape) * 100.0)


def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Errore medio con segno (positivo => sovrastima media)."""
    return float(np.nanmean(y_pred - y_true))


def compute_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Restituisce RMSE/MAE/MAPE/MBE su indici allineati."""
    pred = pd.Series(np.asarray(y_pred, dtype=float), index=y_true.index)
    yt = y_true.astype(float).to_numpy()
    yp = pred.astype(float).to_numpy()
    mbe = mean_bias_error(yt, yp)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "mape": safe_mape(yt, yp),
        "mbe": mbe,
        "abs_mbe": float(abs(mbe)),
    }


def compute_metrics_aligned(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcola le metriche dopo allineamento indice e filtro NaN."""
    y_true_num = pd.to_numeric(y_true, errors="coerce")
    y_pred_num = pd.to_numeric(y_pred, errors="coerce")
    aligned = pd.concat([y_true_num.rename("y_true"), y_pred_num.rename("y_pred")], axis=1).dropna()
    if aligned.empty:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "mape": float("nan"),
            "mbe": float("nan"),
            "abs_mbe": float("nan"),
        }
    return compute_metrics(aligned["y_true"], aligned["y_pred"])


def invert_diff2_log1p(pred_d2: pd.Series, seed_d1: float, seed_log: float) -> pd.Series:
    """Inverte previsioni da diff2(log1p(y)) alla scala originale."""
    d1_pred = seed_d1 + pred_d2.cumsum()
    log_pred = seed_log + d1_pred.cumsum()
    return np.expm1(log_pred)


def original_scale_metrics_for_segment(
    pred_segment: pd.Series,
    original_series: pd.Series | None,
    use_log1p: bool,
    diff_order: int,
) -> dict[str, float] | None:
    """Calcola metriche in scala originale per un segmento predetto.

    Gestisce le trasformazioni: solo log1p (diff_order=0), diff1(log1p)
    e diff2(log1p).
    """
    if original_series is None or not use_log1p or diff_order not in (0, 1, 2):
        return None

    raw = pd.to_numeric(original_series, errors="coerce").dropna().astype(float)
    if raw.empty or pred_segment.empty:
        return None

    if diff_order == 0:
        # Only log1p applied — direct inversion
        pred_orig = np.expm1(pred_segment)
    else:
        x_log = np.log1p(raw)
        seg_start = pred_segment.index.min()

        if diff_order == 1:
            try:
                seed_log = float(x_log[x_log.index < seg_start].iloc[-1])
            except Exception:
                return None
            pred_orig = np.expm1(seed_log + pred_segment.cumsum())
        else:
            x_d1 = x_log.diff().dropna()
            try:
                seed_d1 = float(x_d1[x_d1.index < seg_start].iloc[-1])
                seed_log = float(x_log[x_log.index < seg_start].iloc[-1])
            except Exception:
                return None
            pred_orig = invert_diff2_log1p(pred_segment, seed_d1, seed_log)

    true_orig = raw.reindex(pred_orig.index)
    return compute_metrics_aligned(true_orig, pred_orig)


@dataclass(frozen=True)
class MLStepConfig:
    """Configurazione dei modelli ML non neurali dello Step 4."""

    lookback_values: tuple[int, ...] = (6, 8, 12)
    feature_selection: str = "importance"  # one of: none, rfe, importance
    selected_feature_count: int = 6
    cv_folds: int | None = None
    overfitting_lambda: float = 0.0
    random_state: int = 42
    use_xgboost: bool = True

    # Griglie parametri per modello.
    dt_max_depth: tuple[int | None, ...] = (3, 5, None)
    dt_min_samples_leaf: tuple[int, ...] = (1, 2, 4)

    rf_n_estimators: tuple[int, ...] = (200, 400)
    rf_max_depth: tuple[int | None, ...] = (4, 8, None)
    rf_min_samples_leaf: tuple[int, ...] = (1, 2)

    gbr_n_estimators: tuple[int, ...] = (200, 400)
    gbr_learning_rate: tuple[float, ...] = (0.03, 0.05, 0.1)
    gbr_max_depth: tuple[int, ...] = (2, 3)

    xgb_n_estimators: tuple[int, ...] = (300, 600)
    xgb_learning_rate: tuple[float, ...] = (0.03, 0.05, 0.1)
    xgb_max_depth: tuple[int, ...] = (2, 3, 4)
    xgb_subsample: tuple[float, ...] = (0.8, 1.0)
    xgb_colsample_bytree: tuple[float, ...] = (0.8, 1.0)

    def __post_init__(self) -> None:
        if self.cv_folds is not None and int(self.cv_folds) < 2:
            raise ValueError("cv_folds must be None or >= 2")
        if float(self.overfitting_lambda) < 0.0:
            raise ValueError("overfitting_lambda must be >= 0")

    @staticmethod
    def validate_split(series: pd.Series, name: str) -> pd.Series:
        """Valida e pulisce una serie di split."""
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series")
        s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        if len(s) < 10:
            raise ValueError(f"{name} split is too short for ML lag modeling")
        if not s.index.is_monotonic_increasing:
            s = s.sort_index()
        s.name = "value"
        return s

    @staticmethod
    def validate_original_series(series: pd.Series | None) -> pd.Series | None:
        """Valida la serie originale non trasformata usata per metriche inverse."""
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


def parse_model_name(cfg: dict[str, Any]) -> str:
    """Restituisce il nome famiglia modello a partire da una config."""
    return str(cfg["model"])
