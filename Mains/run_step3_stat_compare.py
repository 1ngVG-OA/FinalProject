"""Compare Step 3 statistical baseline vs extended SARIMA grid search.

Reads both baseline (standard grid) and extended (wider grid) result CSVs
and produces a unified comparison table showing whether deeper exploration
improved the statistical model winner. Computes test-set original-scale metrics
for both on-the-fly from forecast CSV.
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


def _load_summary(suffix: str, raw: pd.Series) -> pd.DataFrame:
    path = METRICS_DIR / f"tavola_1_14_stat_summary_{suffix}.csv"
    fc_path = METRICS_DIR / f"tavola_1_14_stat_forecasts_{suffix}.csv"
    
    df = pd.read_csv(path)
    df["experiment"] = f"step3_{suffix}"

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

    return df


def build_comparison(raw: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = _load_summary("stat_baseline", raw)
    extended = _load_summary("stat_extended", raw)

    combined = pd.concat([baseline, extended], ignore_index=True, sort=False)

    # Select and rename display columns.
    cols = [
        "experiment",
        "model",
        "rmse_val_orig",
        "mae_val_orig",
        "mape_val_orig",
        "abs_mbe_val_orig",
        "rmse_test_orig",
        "mae_test_orig",
        "mape_test_orig",
        "abs_mbe_test_orig",
    ]
    cols = [c for c in cols if c in combined.columns]

    rename = {
        "rmse_val_orig":      "RMSE_val(GWh)",
        "mae_val_orig":       "MAE_val(GWh)",
        "mape_val_orig":      "MAPE_val(%)",
        "abs_mbe_val_orig":   "|MBE|_val(GWh)",
        "rmse_test_orig":     "RMSE_test(GWh)",
        "mae_test_orig":      "MAE_test(GWh)",
        "mape_test_orig":     "MAPE_test(%)",
        "abs_mbe_test_orig":  "|MBE|_test(GWh)",
    }

    result = combined[cols].copy()
    numeric = [c for c in result.columns if result[c].dtype == float]
    result[numeric] = result[numeric].round(1)
    result = result.rename(columns=rename)

    return combined, result


def main() -> None:
    raw = load_target_series(ROOT / "Datasets" / "Tavola_1.14.csv")
    combined, display = build_comparison(raw)

    print("\n" + "=" * 120)
    print("CONFRONTO STEP 3 STATISTICO \u2014 Baseline vs Extended (griglia SARIMA più densa)")
    print("=" * 120)
    print(display.to_string(index=False))
    print("=" * 120)
    print("\nChiave:")
    print("  Baseline        \u2014 configurazione standard (p=0-2, d=0-1, q=0-2).")
    print("  Extended        \u2014 griglia estesa (p=0-4, d=0-1, q=0-4) per più esplorazione parametrica.")
    print("  RMSE / MAE      \u2014 scala originale (GWh). Più basso = migliore.")
    print("  MAPE            \u2014 errore percentuale medio.")
    print("  |MBE|           \u2014 bias assoluto. Vicino a 0 = previsore non distorto sistematicamente.")
    print("  _val            \u2014 validation (tuning).  _test \u2014 test (valutazione finale).")
    print()

    # Overall winners.
    combined_clean = combined.dropna(subset=["rmse_test_orig"])
    if not combined_clean.empty:
        baseline_pool = combined_clean[combined_clean["experiment"] == "step3_stat_baseline"]
        extended_pool = combined_clean[combined_clean["experiment"] == "step3_stat_extended"]

        if not baseline_pool.empty:
            base_best = baseline_pool.loc[baseline_pool["rmse_test_orig"].idxmin()]
            print(f"  Baseline winner (RMSE test originale): {base_best['model']} → {base_best['rmse_test_orig']:,.1f} GWh")

        if not extended_pool.empty:
            ext_best = extended_pool.loc[extended_pool["rmse_test_orig"].idxmin()]
            print(f"  Extended winner (RMSE test originale): {ext_best['model']} → {ext_best['rmse_test_orig']:,.1f} GWh")

        if not baseline_pool.empty and not extended_pool.empty:
            base_rmse = baseline_pool["rmse_test_orig"].min()
            ext_rmse = extended_pool["rmse_test_orig"].min()
            improvement = base_rmse - ext_rmse
            pct_change = (improvement / base_rmse * 100) if base_rmse > 0 else 0
            if improvement > 0:
                print(f"  Miglioramento con Extended: {improvement:,.1f} GWh ({pct_change:+.1f}%) ✓")
            else:
                print(f"  Extended peggiore: {abs(improvement):,.1f} GWh ({pct_change:+.1f}%) ✗")
        print()

    # Save comparison CSV.
    out_path = METRICS_DIR / "tavola_1_14_stat_comparison_baseline_vs_extended.csv"
    display.drop(columns="experiment", errors="ignore").to_csv(out_path, index=False)
    print(f"  Tabella comparativa salvata in: {out_path}")


if __name__ == "__main__":
    main()
