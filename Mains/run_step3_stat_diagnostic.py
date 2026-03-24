"""Run Step 3 statistical diagnostics for baseline vs extended experiments.

This script reads existing Step 3 outputs and produces:
- a detailed comparison table for baseline/extended and both models,
- generalization-gap ratios (validation -> test),
- a robust ranking that balances test accuracy and stability.
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

METRICS_DIR = ROOT / "Results" / "metrics"
TARGET_SERIES_KEY = "production_total"


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mbe = float(np.nanmean(y_pred - y_true))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": _safe_mape(y_true, y_pred),
        "mbe": mbe,
        "abs_mbe": abs(mbe),
    }


def _invert_diff1_log1p(pred: pd.Series, raw: pd.Series, segment_start: int) -> pd.Series:
    x_log = np.log1p(raw.astype(float))
    seed_log = float(x_log[x_log.index < segment_start].iloc[-1])
    return np.expm1(seed_log + pred.cumsum())


def _compute_test_orig_metrics(raw: pd.Series, suffix: str, model_col: str) -> dict[str, float]:
    forecasts_path = METRICS_DIR / f"tavola_1_14_stat_forecasts_{suffix}.csv"
    if not forecasts_path.exists():
        return {}

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
    if not mask.any():
        return {}

    return _metrics(true_orig.to_numpy()[mask], pred_orig.to_numpy()[mask])


def _load_summary(suffix: str, raw: pd.Series) -> pd.DataFrame:
    path = METRICS_DIR / f"tavola_1_14_stat_summary_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    df = pd.read_csv(path)
    df["experiment"] = suffix

    # Some summaries do not persist *_test_orig metrics; rebuild from forecast table.
    model_to_col = {
        "sarima": "sarima_pred",
        "holt_winters": "hw_pred",
    }
    for model_name, pred_col in model_to_col.items():
        row_mask = df["model"] == model_name
        if not row_mask.any():
            continue
        try:
            m = _compute_test_orig_metrics(raw, suffix, pred_col)
            if m:
                df.loc[row_mask, "rmse_test_orig"] = m["rmse"]
                df.loc[row_mask, "mae_test_orig"] = m["mae"]
                df.loc[row_mask, "mape_test_orig"] = m["mape"]
                df.loc[row_mask, "mbe_test_orig"] = m["mbe"]
                df.loc[row_mask, "abs_mbe_test_orig"] = m["abs_mbe"]
        except Exception:
            continue

    return df


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den_clean = den.replace(0.0, np.nan)
    return num / den_clean


def _paired_scale_columns(
    df: pd.DataFrame,
    val_col: str,
    test_col: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return val/test metric columns on a consistent common scale.

    Preference order:
    1) original scale only if BOTH val_orig and test_orig exist,
    2) transformed scale otherwise.
    """
    val_orig_col = f"{val_col}_orig"
    test_orig_col = f"{test_col}_orig"

    can_use_orig = val_orig_col in df.columns and test_orig_col in df.columns
    if can_use_orig:
        val_series = df[val_orig_col].fillna(df[val_col])
        test_series = df[test_orig_col].fillna(df[test_col])
        scale = pd.Series(["original"] * len(df), index=df.index)
        return val_series, test_series, scale

    val_series = df[val_col]
    test_series = df[test_col]
    scale = pd.Series(["transformed"] * len(df), index=df.index)
    return val_series, test_series, scale


def _build_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rmse_val, rmse_test, rmse_scale = _paired_scale_columns(out, "rmse_val", "rmse_test")
    mae_val, mae_test, _ = _paired_scale_columns(out, "mae_val", "mae_test")
    mape_val, mape_test, _ = _paired_scale_columns(out, "mape_val", "mape_test")

    out["rank_rmse_val"] = rmse_val
    out["rank_rmse_test"] = rmse_test
    out["rank_mae_val"] = mae_val
    out["rank_mae_test"] = mae_test
    out["rank_mape_val"] = mape_val
    out["rank_mape_test"] = mape_test
    out["ranking_scale"] = rmse_scale

    # Generalization ratios: > 1 means degradation from val to test.
    out["gap_rmse_ratio"] = _safe_ratio(out["rank_rmse_test"], out["rank_rmse_val"])
    out["gap_mae_ratio"] = _safe_ratio(out["rank_mae_test"], out["rank_mae_val"])
    out["gap_mape_ratio"] = _safe_ratio(out["rank_mape_test"], out["rank_mape_val"])

    # Stability penalty: prioritize models with smaller degradation out-of-sample.
    # Ratio 1.0 -> no penalty, 2.0 -> +100% penalty.
    out["stability_penalty"] = (out["gap_rmse_ratio"].fillna(10.0) - 1.0).clip(lower=0.0)

    # Robust score on test scale with stability penalty.
    # Lower is better.
    out["robust_score"] = out["rank_rmse_test"] * (1.0 + 0.35 * out["stability_penalty"])

    return out


def _build_shortlist(diag: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "experiment",
        "model",
        "ranking_scale",
        "best_params",
        "rank_rmse_val",
        "rank_rmse_test",
        "rank_mae_test",
        "rank_mape_test",
        "gap_rmse_ratio",
        "gap_mae_ratio",
        "gap_mape_ratio",
        "stability_penalty",
        "robust_score",
    ]

    shortlisted = diag[cols].copy().sort_values(
        ["robust_score", "rank_rmse_test", "gap_rmse_ratio"],
        ascending=[True, True, True],
    )
    return shortlisted.reset_index(drop=True)


def _print_summary(shortlist: pd.DataFrame) -> None:
    top_robust = shortlist.iloc[0]
    top_accuracy = shortlist.sort_values(
        ["rank_rmse_test", "rank_mae_test", "rank_mape_test"],
        ascending=[True, True, True],
    ).iloc[0]

    print("\n" + "=" * 120)
    print("STEP 3 STATISTICAL DIAGNOSTICS - BASELINE VS EXTENDED")
    print("=" * 120)
    print(shortlist.to_string(index=False))
    print("=" * 120)

    print("\nBest candidate by ACCURACY (test metrics):")
    print(
        f"- Experiment: {top_accuracy['experiment']}\n"
        f"- Model: {top_accuracy['model']}\n"
        f"- RMSE val: {top_accuracy['rank_rmse_val']:.1f}\n"
        f"- RMSE test: {top_accuracy['rank_rmse_test']:.1f}\n"
        f"- Gap RMSE ratio (test/val): {top_accuracy['gap_rmse_ratio']:.2f}x\n"
        f"- Robust score: {top_accuracy['robust_score']:.1f}"
    )

    print("\nBest candidate by ROBUSTNESS (stability-aware):")
    print(
        f"- Experiment: {top_robust['experiment']}\n"
        f"- Model: {top_robust['model']}\n"
        f"- RMSE val: {top_robust['rank_rmse_val']:.1f}\n"
        f"- RMSE test: {top_robust['rank_rmse_test']:.1f}\n"
        f"- Gap RMSE ratio (test/val): {top_robust['gap_rmse_ratio']:.2f}x\n"
        f"- Robust score: {top_robust['robust_score']:.1f}"
    )

    print("\nInterpretation notes:")
    print("- gap ratio near 1.0x means strong generalization")
    print("- robust score penalizes models that collapse from validation to test")


def main() -> None:
    from Project.preprocessing.descriptive_analysis import load_target_series

    raw = load_target_series(ROOT / "Datasets" / "Tavola_1.14.csv", target=TARGET_SERIES_KEY)

    baseline = _load_summary("stat_baseline", raw)
    extended = _load_summary("stat_extended", raw)

    combined = pd.concat([baseline, extended], ignore_index=True)
    diagnostic = _build_diagnostic(combined)
    shortlist = _build_shortlist(diagnostic)

    out_diag = METRICS_DIR / "tavola_1_14_stat_diagnostic_baseline_vs_extended.csv"
    out_shortlist = METRICS_DIR / "tavola_1_14_stat_shortlist_baseline_vs_extended.csv"

    diagnostic.to_csv(out_diag, index=False)
    shortlist.to_csv(out_shortlist, index=False)

    _print_summary(shortlist)

    print("\nSaved outputs:")
    print(f"- diagnostic: {out_diag}")
    print(f"- shortlist:  {out_shortlist}")


if __name__ == "__main__":
    main()
