"""Configurazione e utility per modelli di forecasting neurali basati su torch."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Project.preprocessing.time_series_preprocessor import PreprocessingConfig, TimeSeriesPreprocessor


# ------------------------------------------------------------------
# Utility training e device
# ------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Imposta seed Python, NumPy e Torch per training riproducibile."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_torch_device(device_name: str) -> torch.device:
    """Risolve il device torch con fallback esplicito a CPU."""

    requested = str(device_name).lower().strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------------------------------------------------
# Utility metriche e inversione trasformazioni
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
    return pd.Series(np.expm1(log_pred.to_numpy(dtype=float)), index=pred_d2.index, name="pred_orig")


def _fit_scaler(train_series: pd.Series, scale_method: str) -> StandardScaler | MinMaxScaler | None:
    method = str(scale_method).lower()
    if method == "none":
        return None
    if method == "standard":
        scaler: StandardScaler | MinMaxScaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scale_method must be one of: none, standard, minmax")

    scaler.fit(train_series.to_numpy().reshape(-1, 1))
    return scaler


def invert_preprocessed_segment(
    pred_segment: pd.Series,
    original_series: pd.Series | None,
    preprocessing_config: PreprocessingConfig | None,
) -> pd.Series | None:
    """Inverte scaling e trasformazioni deterministiche alla scala originale."""

    if original_series is None or preprocessing_config is None:
        return None

    raw = NeuralStepConfig.validate_original_series(original_series)
    if raw is None or pred_segment.empty:
        return None

    transform_cfg = preprocessing_config.transform
    if transform_cfg.power_exponent is not None:
        return None

    preproc = TimeSeriesPreprocessor(raw, preprocessing_config)
    transformed_full = preproc.apply_deterministic_transforms(raw)
    transformed_splits = preproc.split_series(transformed_full)
    scaler = _fit_scaler(transformed_splits["train"], transform_cfg.scale_method)

    pred = pd.Series(pd.to_numeric(pred_segment, errors="coerce"), index=pred_segment.index).dropna()
    if pred.empty:
        return None

    if scaler is not None:
        pred_transformed = pd.Series(
            scaler.inverse_transform(pred.to_numpy().reshape(-1, 1)).ravel(),
            index=pred.index,
            name=pred.name,
        )
    else:
        pred_transformed = pred

    if transform_cfg.diff_order == 0:
        if transform_cfg.use_log1p:
            return pd.Series(np.expm1(pred_transformed.to_numpy()), index=pred_transformed.index)
        return pred_transformed

    if not transform_cfg.use_log1p:
        return None

    x_log = pd.Series(np.log1p(raw.to_numpy(dtype=float)), index=raw.index, name="log1p")
    seg_start = pred_transformed.index.min()

    if transform_cfg.diff_order == 1:
        try:
            seed_log = float(x_log[x_log.index < seg_start].iloc[-1])
        except Exception:
            return None
        pred_log = seed_log + pred_transformed.cumsum()
        return pd.Series(
            np.expm1(pred_log.to_numpy(dtype=float)),
            index=pred_transformed.index,
        )

    if transform_cfg.diff_order == 2:
        x_d1 = x_log.diff().dropna()
        try:
            seed_d1 = float(x_d1[x_d1.index < seg_start].iloc[-1])
            seed_log = float(x_log[x_log.index < seg_start].iloc[-1])
        except Exception:
            return None
        return invert_diff2_log1p(pred_transformed, seed_d1, seed_log)

    return None


def original_scale_metrics_for_segment(
    pred_segment: pd.Series,
    original_series: pd.Series | None,
    preprocessing_config: PreprocessingConfig | None,
) -> dict[str, float] | None:
    """Calcola metriche in scala originale dopo inversione del preprocessing."""

    pred_orig = invert_preprocessed_segment(pred_segment, original_series, preprocessing_config)
    if pred_orig is None:
        return None

    raw = NeuralStepConfig.validate_original_series(original_series)
    if raw is None:
        return None

    true_orig = raw.reindex(pred_orig.index)
    return compute_metrics_aligned(true_orig, pred_orig)


@dataclass(frozen=True)
class NeuralStepConfig:
    """Configurazione dello step neurale di forecasting."""

    candidate_models: tuple[str, ...] = ("mlp", "lstm")
    lookback_values: tuple[int, ...] = (6, 12)

    mlp_hidden_sizes: tuple[int, ...] = (8, 16)
    mlp_activations: tuple[str, ...] = ("relu", "tanh")
    mlp_dropouts: tuple[float, ...] = (0.0, 0.1)

    lstm_hidden_sizes: tuple[int, ...] = (8, 16)
    lstm_num_layers: tuple[int, ...] = (1,)
    lstm_dropouts: tuple[float, ...] = (0.0,)

    batch_sizes: tuple[int, ...] = (16, 32)
    learning_rates: tuple[float, ...] = (1e-3,)
    weight_decays: tuple[float, ...] = (0.0,)

    max_epochs: int = 200
    patience: int = 20
    seed: int = 42
    device: str = "auto"

    @staticmethod
    def validate_split(series: pd.Series, name: str) -> pd.Series:
        """Valida uno split in ingresso per il windowing neurale."""

        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series")
        x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        if len(x) < 10:
            raise ValueError(f"{name} split is too short for neural modeling")
        if not x.index.is_monotonic_increasing:
            x = x.sort_index()
        x.name = "value"
        return x

    @staticmethod
    def validate_original_series(series: pd.Series | None) -> pd.Series | None:
        """Valida la serie originale non trasformata usata per metriche inverse."""

        if series is None:
            return None
        if not isinstance(series, pd.Series):
            raise TypeError("original_series must be a pandas Series or None")
        x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        if x.empty:
            return None
        if not x.index.is_monotonic_increasing:
            x = x.sort_index()
        return x


def build_compact_neural_config() -> NeuralStepConfig:
    """Restituisce la configurazione neurale compatta consigliata."""

    return NeuralStepConfig(
        candidate_models=("mlp", "lstm"),
        lookback_values=(6, 12),
        mlp_hidden_sizes=(8, 16),
        mlp_activations=("relu", "tanh"),
        mlp_dropouts=(0.0, 0.1),
        lstm_hidden_sizes=(8, 16),
        lstm_num_layers=(1,),
        lstm_dropouts=(0.0,),
        batch_sizes=(16, 32),
        learning_rates=(1e-3,),
        weight_decays=(0.0,),
        max_epochs=200,
        patience=20,
        seed=42,
        device="cpu",
    )


def build_extended_neural_config() -> NeuralStepConfig:
    """Restituisce lo spazio di ricerca neurale esteso per esplorazione ampia."""

    return NeuralStepConfig(
        candidate_models=("mlp", "lstm"),
        lookback_values=(6, 8, 12),
        mlp_hidden_sizes=(8, 16),
        mlp_activations=("relu", "tanh"),
        mlp_dropouts=(0.0,),
        lstm_hidden_sizes=(8, 16),
        lstm_num_layers=(1, 2),
        lstm_dropouts=(0.0,),
        batch_sizes=(16,),
        learning_rates=(1e-3, 5e-4),
        weight_decays=(0.0,),
        max_epochs=120,
        patience=20,
        seed=42,
        device="cpu",
    )