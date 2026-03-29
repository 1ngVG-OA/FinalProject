"""Utility a finestre mobili per modelli di forecasting neurali."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowedSplits:
    """Dataset supervisionati e seed ricorsivi per uno specifico lookback."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_seed: pd.Series
    train_val_seed: pd.Series


def build_training_windows(series: pd.Series, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Crea finestre supervisionate one-step da un singolo segmento."""

    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(x) <= lookback:
        raise ValueError("Series is too short for the requested lookback")

    values = x.to_numpy()
    X_rows = [values[i - lookback:i] for i in range(lookback, len(values))]
    y_rows = [values[i] for i in range(lookback, len(values))]
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float)


def build_segment_windows(
    source_series: pd.Series,
    target_index: pd.Index,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crea finestre per target che appartengono a segmenti successivi della serie."""

    x = pd.to_numeric(source_series, errors="coerce").dropna().astype(float)
    index_to_pos = {idx: pos for pos, idx in enumerate(x.index)}

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for target in target_index:
        if target not in index_to_pos:
            raise KeyError(f"Target index {target!r} not found in source series")
        pos = index_to_pos[target]
        if pos < lookback:
            raise ValueError("Insufficient history to build target window")
        X_rows.append(x.iloc[pos - lookback:pos].to_numpy(dtype=float))
        y_rows.append(float(x.iloc[pos]))

    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float)


def build_windowed_splits(
    train: pd.Series,
    validation: pd.Series,
    test: pd.Series,
    lookback: int,
) -> WindowedSplits:
    """Costruisce dataset supervisionati train/validation/test e seed ricorsivi."""

    train_series = pd.to_numeric(train, errors="coerce").dropna().astype(float)
    val_series = pd.to_numeric(validation, errors="coerce").dropna().astype(float)
    test_series = pd.to_numeric(test, errors="coerce").dropna().astype(float)

    if len(train_series) <= lookback:
        raise ValueError("Train split is too short for neural windowing")

    train_val = pd.concat([train_series, val_series], axis=0)
    full = pd.concat([train_series, val_series, test_series], axis=0)

    X_train, y_train = build_training_windows(train_series, lookback)
    X_val, y_val = build_segment_windows(train_val, val_series.index, lookback)
    X_test, y_test = build_segment_windows(full, test_series.index, lookback)

    return WindowedSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_seed=train_series.iloc[-lookback:].copy(),
        train_val_seed=train_val.iloc[-lookback:].copy(),
    )