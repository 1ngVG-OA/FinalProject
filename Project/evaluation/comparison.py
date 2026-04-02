"""Utility di confronto cross-family per la pipeline di forecasting.

Il modulo aggrega i risultati dei vincitori statistical/ml/neural,
costruisce tabelle comparative e identifica il global winner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from Project.models.ml.model_config import invert_diff2_log1p
from Project.models.neural.model_config import invert_preprocessed_segment


def _invert_log_diff_segment(
    pred_segment: pd.Series,
    original_series: pd.Series | None,
    use_log1p: bool,
    diff_order: int,
) -> pd.Series | None:
    """Inverte un segmento forecast log1p/differenziato in scala originale."""

    if original_series is None or pred_segment.empty:
        return None

    raw = pd.to_numeric(original_series, errors="coerce").dropna().astype(float)
    if raw.empty:
        return None

    pred = pd.Series(pd.to_numeric(pred_segment, errors="coerce"), index=pred_segment.index).dropna()
    if pred.empty:
        return None

    if not use_log1p:
        return pred

    if diff_order == 0:
        return pd.Series(np.expm1(pred.to_numpy(dtype=float)), index=pred.index, name="pred_orig")

    x_log = pd.Series(np.log1p(raw.to_numpy(dtype=float)), index=raw.index, name="log1p")
    seg_start = pred.index.min()

    if diff_order == 1:
        seed_candidates = x_log[x_log.index < seg_start]
        if seed_candidates.empty:
            return None
        seed_log = float(seed_candidates.iloc[-1])
        pred_log = seed_log + pred.cumsum()
        return pd.Series(np.expm1(pred_log.to_numpy(dtype=float)), index=pred.index, name="pred_orig")

    if diff_order == 2:
        x_d1 = x_log.diff().dropna()
        seed_d1_candidates = x_d1[x_d1.index < seg_start]
        seed_log_candidates = x_log[x_log.index < seg_start]
        if seed_d1_candidates.empty or seed_log_candidates.empty:
            return None
        seed_d1 = float(seed_d1_candidates.iloc[-1])
        seed_log = float(seed_log_candidates.iloc[-1])
        return invert_diff2_log1p(pred, seed_d1, seed_log)

    return None


def _with_family_metadata(summary: pd.DataFrame, family: str) -> pd.DataFrame:
    """Aggiunge la colonna family a una summary di modello."""
    df = summary.copy()
    df.insert(0, "family", family)
    return df


def _winner_row(summary: pd.DataFrame, family: str, winner: str) -> pd.Series:
    """Estrae la riga del modello vincitore per una specifica family."""
    row = summary.loc[summary["model"].astype(str) == str(winner)]
    if row.empty:
        raise KeyError(f"Winner '{winner}' not found in summary for family '{family}'")
    winner_row = row.iloc[0].copy()
    winner_row.loc["family"] = family
    return winner_row


def _extract_statistical_preds(output: dict[str, Any], winner: str) -> tuple[pd.Series, pd.Series, pd.Series | None, pd.Series | None]:
    """Estrae predizioni statistical (scala trasformata e scala originale)."""
    val_pred = pd.Series(output[f"{winner}_val_pred"], copy=True)
    test_pred = pd.Series(output[f"{winner}_test_pred"], copy=True)
    val_orig = _invert_log_diff_segment(val_pred, output.get("original_series"), output.get("use_log1p", False), int(output.get("diff_order", 0)))
    test_orig = _invert_log_diff_segment(test_pred, output.get("original_series"), output.get("use_log1p", False), int(output.get("diff_order", 0)))
    return val_pred, test_pred, val_orig, test_orig


def _extract_ml_preds(output: dict[str, Any], winner: str) -> tuple[pd.Series, pd.Series, pd.Series | None, pd.Series | None]:
    """Estrae predizioni ML (scala trasformata e scala originale)."""
    pred_series = output["pred_series"][winner]
    val_pred = pd.Series(pred_series["val"], copy=True)
    test_pred = pd.Series(pred_series["test"], copy=True)
    val_orig = _invert_log_diff_segment(val_pred, output.get("original_series"), output.get("use_log1p", False), int(output.get("diff_order", 0)))
    test_orig = _invert_log_diff_segment(test_pred, output.get("original_series"), output.get("use_log1p", False), int(output.get("diff_order", 0)))
    return val_pred, test_pred, val_orig, test_orig


def _extract_neural_preds(output: dict[str, Any], winner: str) -> tuple[pd.Series, pd.Series, pd.Series | None, pd.Series | None]:
    """Estrae predizioni neural (scala trasformata e scala originale)."""
    pred_series = output["pred_series"][winner]
    val_pred = pd.Series(pred_series["val"], copy=True)
    test_pred = pd.Series(pred_series["test"], copy=True)
    val_orig = invert_preprocessed_segment(val_pred, output.get("original_series"), output.get("preprocessing_config"))
    test_orig = invert_preprocessed_segment(test_pred, output.get("original_series"), output.get("preprocessing_config"))
    return val_pred, test_pred, val_orig, test_orig


def _build_winner_forecasts(
    statistical_output: dict[str, Any],
    ml_output: dict[str, Any],
    neural_output: dict[str, Any],
) -> pd.DataFrame:
    """Costruisce la tabella forecast unificata in scala originale dei vincitori."""

    stat_winner = str(statistical_output["winner"])
    ml_winner = str(ml_output["winner"])
    neural_winner = str(neural_output["winner"])

    _, _, stat_val_orig, stat_test_orig = _extract_statistical_preds(statistical_output, stat_winner)
    _, _, ml_val_orig, ml_test_orig = _extract_ml_preds(ml_output, ml_winner)
    _, _, neural_val_orig, neural_test_orig = _extract_neural_preds(neural_output, neural_winner)

    if any(x is None for x in (stat_val_orig, stat_test_orig, ml_val_orig, ml_test_orig, neural_val_orig, neural_test_orig)):
        raise ValueError("Original-scale inversion failed for at least one winner forecast")
    # Dopo la guardia su None, questi assert rendono esplicito (a runtime e per il type-checker)
    # che le serie invertite esistono, evitando accessi ambigui a reindex/to_numpy nelle righe successive.
    assert stat_val_orig is not None and stat_test_orig is not None
    assert ml_val_orig is not None and ml_test_orig is not None
    assert neural_val_orig is not None and neural_test_orig is not None

    val_index = statistical_output["validation_actual"].index
    test_index = statistical_output["test_actual"].index
    original_series = statistical_output.get("original_series")
    if not isinstance(original_series, pd.Series):
        raise TypeError("statistical_output['original_series'] must be a pandas Series")
    raw = pd.to_numeric(original_series, errors="coerce").dropna().astype(float)

    comparison = pd.DataFrame(
        {
            "split": ["validation"] * len(val_index) + ["test"] * len(test_index),
            "timestamp": list(val_index) + list(test_index),
        }
    )
    comparison["actual"] = raw.reindex(comparison["timestamp"]).to_numpy()
    comparison["statistical_pred"] = list(stat_val_orig.reindex(val_index).to_numpy()) + list(stat_test_orig.reindex(test_index).to_numpy())
    comparison["ml_pred"] = list(ml_val_orig.reindex(val_index).to_numpy()) + list(ml_test_orig.reindex(test_index).to_numpy())
    comparison["neural_pred"] = list(neural_val_orig.reindex(val_index).to_numpy()) + list(neural_test_orig.reindex(test_index).to_numpy())

    for family in ("statistical", "ml", "neural"):
        comparison[f"{family}_error"] = comparison[f"{family}_pred"] - comparison["actual"]

    comparison.insert(0, "scale", "original")
    return comparison


def build_cross_family_comparison(
    statistical_output: dict[str, Any],
    ml_output: dict[str, Any],
    neural_output: dict[str, Any],
) -> dict[str, Any]:
    """Costruisce tabelle complete e tabelle winner-only cross-family."""

    # ------------------------------------------------------------------
    # Tabella globale: tutti i modelli
    # ------------------------------------------------------------------

    all_models = pd.concat(
        [
            _with_family_metadata(statistical_output["summary"], "statistical"),
            _with_family_metadata(ml_output["summary"], "ml"),
            _with_family_metadata(neural_output["summary"], "neural"),
        ],
        ignore_index=True,
        sort=False,
    )
    all_models = all_models.assign(rank_rmse_val_global=all_models["rmse_val_orig"].fillna(all_models["rmse_val"]))
    all_models = all_models.assign(rank_abs_mbe_val_global=all_models["abs_mbe_val_orig"].fillna(all_models["abs_mbe_val"]))
    all_models = all_models.sort_values(["rank_rmse_val_global", "rank_abs_mbe_val_global"], ascending=[True, True]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Tabella family winners e ranking globale
    # ------------------------------------------------------------------

    family_winners = pd.DataFrame(
        [
            _winner_row(statistical_output["summary"], "statistical", str(statistical_output["winner"])),
            _winner_row(ml_output["summary"], "ml", str(ml_output["winner"])),
            _winner_row(neural_output["summary"], "neural", str(neural_output["winner"])),
        ]
    )
    family_winners = family_winners.assign(rank_rmse_val_global=family_winners["rmse_val_orig"].fillna(family_winners["rmse_val"]))
    family_winners = family_winners.assign(rank_abs_mbe_val_global=family_winners["abs_mbe_val_orig"].fillna(family_winners["abs_mbe_val"]))
    family_winners = family_winners.sort_values(["rank_rmse_val_global", "rank_abs_mbe_val_global"], ascending=[True, True]).reset_index(drop=True)
    family_winners.insert(0, "global_rank", np.arange(1, len(family_winners) + 1, dtype=int))

    # ------------------------------------------------------------------
    # Forecast unificati del global winner set
    # ------------------------------------------------------------------

    winner_forecasts = _build_winner_forecasts(statistical_output, ml_output, neural_output)
    global_winner = family_winners.iloc[0]

    return {
        "all_models": all_models,
        "family_winners": family_winners,
        "winner_forecasts": winner_forecasts,
        "global_winner": {
            "family": str(global_winner["family"]),
            "model": str(global_winner["model"]),
            "global_rank": int(global_winner["global_rank"]),
            "rank_rmse_val_global": float(global_winner["rank_rmse_val_global"]),
            "rank_abs_mbe_val_global": float(global_winner["rank_abs_mbe_val_global"]),
        },
    }