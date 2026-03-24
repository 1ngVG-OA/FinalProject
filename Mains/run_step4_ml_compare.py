"""Compare Step 3 (statistical) vs Step 4 ML (baseline v1 and extended v2).

Reads existing result CSVs and produces a unified ranking table without
re-running any experiment. Computes test-set original-scale metrics for
statistical models on-the-fly from the saved forecast CSV.
Saves a comparison CSV for reporting.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Project.preprocessing.descriptive_analysis import load_target_series

METRICS_DIR = ROOT / "Results" / "metrics"
ARTIFACTS_DIR = ROOT / "Results" / "artifacts"
TARGET_SERIES_KEY = "consumption_total"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mbe = float(np.nanmean(y_pred - y_true))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": _safe_mape(y_true, y_pred),
        "abs_mbe": abs(mbe),
    }


def _invert_diff1_log1p(
    pred: pd.Series,
    raw: pd.Series,
    segment_start: int,
) -> pd.Series:
    """Invert diff1(log1p(y)) predictions to original scale."""
    x_log = np.log1p(raw.astype(float))
    seed_log = float(x_log[x_log.index < segment_start].iloc[-1])
    return np.expm1(seed_log + pred.cumsum())


def _test_orig_metrics_stat(
    raw: pd.Series,
    forecasts_path: Path,
    model_col: str,
) -> dict[str, float]:
    """Compute test original-scale metrics for a statistical model from its saved forecast CSV."""
    fc = pd.read_csv(forecasts_path)
    test_fc = fc[fc["split"] == "test"].copy()
    test_fc["timestamp"] = test_fc["timestamp"].astype(int)
    pred = pd.Series(
        test_fc[model_col].astype(float).to_numpy(),
        index=test_fc["timestamp"].to_numpy(),
    )
    pred_orig = _invert_diff1_log1p(pred, raw, int(pred.index.min()))
    true_orig = raw.reindex(pred_orig.index).astype(float)
    mask = ~(np.isnan(true_orig.to_numpy()) | np.isnan(pred_orig.to_numpy()))
    return _metrics(true_orig.to_numpy()[mask], pred_orig.to_numpy()[mask])


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_stat(raw: pd.Series) -> pd.DataFrame:
    path = METRICS_DIR / "tavola_1_14_stat_summary_v1.csv"
    fc_path = METRICS_DIR / "tavola_1_14_stat_forecasts_v1.csv"
    df = pd.read_csv(path)
    df["experiment"] = "step3_statistical"
    df["step"] = "step3"
    df["lookback"] = "-"

    # Compute test_orig metrics on-the-fly from the saved forecast table.
    for model_name, pred_col in [("sarima", "sarima_pred"), ("holt_winters", "hw_pred")]:
        row_mask = df["model"] == model_name
        if row_mask.any() and fc_path.exists():
            try:
                m = _test_orig_metrics_stat(raw, fc_path, pred_col)
                df.loc[row_mask, "rmse_test_orig"] = m["rmse"]
                df.loc[row_mask, "mae_test_orig"] = m["mae"]
                df.loc[row_mask, "mape_test_orig"] = m["mape"]
                df.loc[row_mask, "abs_mbe_test_orig"] = m["abs_mbe"]
            except Exception:
                pass

    for col in ("rmse_val_orig", "mae_val_orig", "mape_val_orig", "abs_mbe_val_orig",
                "rmse_test_orig", "mae_test_orig", "mape_test_orig", "abs_mbe_test_orig"):
        if col not in df.columns:
            df[col] = float("nan")
    return df


def _load_ml(csv_path: Path, experiment_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["experiment"] = experiment_label
    df["step"] = "step4"
    df["lookback"] = df["lookback"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Build and display
# ---------------------------------------------------------------------------

COLS_DISPLAY = [
    "experiment", "model", "lookback",
    "rmse_val_orig", "mae_val_orig", "mape_val_orig", "abs_mbe_val_orig",
    "rmse_test_orig", "mae_test_orig", "mape_test_orig", "abs_mbe_test_orig",
]

RENAME = {
    "rmse_val_orig":  "RMSE_val(GWh)",
    "mae_val_orig":   "MAE_val(GWh)",
    "mape_val_orig":  "MAPE_val(%)",
    "abs_mbe_val_orig": "|MBE|_val(GWh)",
    "rmse_test_orig": "RMSE_test(GWh)",
    "mae_test_orig":  "MAE_test(GWh)",
    "mape_test_orig": "MAPE_test(%)",
    "abs_mbe_test_orig": "|MBE|_test(GWh)",
}


def build_comparison(raw: pd.Series) -> pd.DataFrame:
    stat = _load_stat(raw)
    ml_v1 = _load_ml(METRICS_DIR / "tavola_1_14_ml_summary_v1.csv", "step4_baseline (no XGBoost)")
    ml_v2 = _load_ml(METRICS_DIR / "tavola_1_14_ml_summary_xgb_v2.csv", "step4_extended (XGBoost on)")

    combined = pd.concat([stat, ml_v1, ml_v2], ignore_index=True, sort=False)

    # Sort: step3 first, then within each group by test RMSE ascending.
    combined["_step_order"] = combined["step"].map({"step3": 0, "step4": 1}).fillna(2)
    combined["_sort_rmse"] = pd.to_numeric(combined["rmse_test_orig"], errors="coerce")
    combined = combined.sort_values(["_step_order", "_sort_rmse"]).reset_index(drop=True)

    cols = [c for c in COLS_DISPLAY if c in combined.columns]
    result = combined[cols + ["_step_order", "_sort_rmse", "step"]].copy()

    numeric = [c for c in cols if pd.api.types.is_float_dtype(result[c])]
    result[numeric] = result[numeric].round(1)

    return combined, result.rename(columns=RENAME)


def main() -> None:
    raw = load_target_series(ROOT / "Datasets" / "Tavola_1.14.csv", target=TARGET_SERIES_KEY)
    combined, display = build_comparison(raw)

    print_cols = [c for c in display.columns if not c.startswith("_") and c != "step"]
    out_df = display[print_cols]

    print("\n" + "=" * 120)
    print("CONFRONTO MODELLI — Step 3 Statistico vs Step 4 ML — scala originale (GWh)")
    print("=" * 120)
    print(out_df.to_string(index=False))
    print("=" * 120)
    print("\nChiave:")
    print("  RMSE / MAE      — scala originale (GWh). Più basso = migliore.")
    print("  MAPE            — errore percentuale medio.")
    print("  |MBE|           — bias assoluto. Vicino a 0 = previsore non distorto sistematicamente.")
    print("  _val            — validation (tuning).  _test — test (valutazione finale).")
    print()

    # Overall winner on test RMSE original scale.
    winner_pool = combined.dropna(subset=["_sort_rmse"])
    if not winner_pool.empty:
        best = winner_pool.loc[winner_pool["_sort_rmse"].idxmin()]
        print("  Miglior modello su RMSE test (scala originale):")
        print(f"    {best['model']} [{best['experiment']}] -> {best['_sort_rmse']:,.1f} GWh")
        print()

    # Save.
    out_path = METRICS_DIR / "tavola_1_14_comparison_all_models.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Tabella comparativa salvata in: {out_path}")


if __name__ == "__main__":
    main()
