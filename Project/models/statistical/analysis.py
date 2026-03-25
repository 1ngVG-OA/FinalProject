"""Analysis helpers for Step 3 statistical baseline and extended experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


def compute_test_orig_metrics(raw: pd.Series, forecasts_path: Path, model_col: str) -> dict[str, float]:
    """Compute test metrics on original scale from a saved forecast table."""

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


def load_variant_summary(metrics_dir: Path, variant: str, raw: pd.Series) -> pd.DataFrame:
    """Load a baseline/extended statistical summary and enrich it with test original-scale metrics."""

    summary_path = Path(metrics_dir) / f"summary_{variant}.csv"
    forecasts_path = Path(metrics_dir) / f"forecasts_{variant}.csv"

    df = pd.read_csv(summary_path)
    df["experiment"] = variant

    for model_name, pred_col in (("sarima", "sarima_pred"), ("holt_winters", "hw_pred")):
        row_mask = df["model"] == model_name
        if not row_mask.any() or not forecasts_path.exists():
            continue
        try:
            m = compute_test_orig_metrics(raw, forecasts_path, pred_col)
            if m:
                df.loc[row_mask, "rmse_test_orig"] = m["rmse"]
                df.loc[row_mask, "mae_test_orig"] = m["mae"]
                df.loc[row_mask, "mape_test_orig"] = m["mape"]
                df.loc[row_mask, "mbe_test_orig"] = m["mbe"]
                df.loc[row_mask, "abs_mbe_test_orig"] = m["abs_mbe"]
        except Exception:
            continue

    return df


def build_baseline_extended_comparison(raw: pd.Series, metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a side-by-side baseline vs extended comparison table."""

    baseline = load_variant_summary(metrics_dir, "baseline", raw)
    extended = load_variant_summary(metrics_dir, "extended", raw)
    combined = pd.concat([baseline, extended], ignore_index=True, sort=False)

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

    display = combined[cols].copy()
    numeric = [c for c in display.columns if display[c].dtype == float]
    display[numeric] = display[numeric].round(1)
    display = display.rename(
        columns={
            "rmse_val_orig": "RMSE_val(GWh)",
            "mae_val_orig": "MAE_val(GWh)",
            "mape_val_orig": "MAPE_val(%)",
            "abs_mbe_val_orig": "|MBE|_val(GWh)",
            "rmse_test_orig": "RMSE_test(GWh)",
            "mae_test_orig": "MAE_test(GWh)",
            "mape_test_orig": "MAPE_test(%)",
            "abs_mbe_test_orig": "|MBE|_test(GWh)",
        }
    )
    return combined, display


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den_clean = den.replace(0.0, np.nan)
    return num / den_clean


def _paired_scale_columns(
    df: pd.DataFrame,
    val_col: str,
    test_col: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
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


def build_diagnostic_tables(raw: pd.Series, metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build detailed diagnostic and shortlist tables for baseline vs extended experiments."""

    baseline = load_variant_summary(metrics_dir, "baseline", raw)
    extended = load_variant_summary(metrics_dir, "extended", raw)
    combined = pd.concat([baseline, extended], ignore_index=True)

    diagnostic = combined.copy()

    rmse_val, rmse_test, rmse_scale = _paired_scale_columns(diagnostic, "rmse_val", "rmse_test")
    mae_val, mae_test, _ = _paired_scale_columns(diagnostic, "mae_val", "mae_test")
    mape_val, mape_test, _ = _paired_scale_columns(diagnostic, "mape_val", "mape_test")

    diagnostic["rank_rmse_val"] = rmse_val
    diagnostic["rank_rmse_test"] = rmse_test
    diagnostic["rank_mae_val"] = mae_val
    diagnostic["rank_mae_test"] = mae_test
    diagnostic["rank_mape_val"] = mape_val
    diagnostic["rank_mape_test"] = mape_test
    diagnostic["ranking_scale"] = rmse_scale
    diagnostic["gap_rmse_ratio"] = _safe_ratio(diagnostic["rank_rmse_test"], diagnostic["rank_rmse_val"])
    diagnostic["gap_mae_ratio"] = _safe_ratio(diagnostic["rank_mae_test"], diagnostic["rank_mae_val"])
    diagnostic["gap_mape_ratio"] = _safe_ratio(diagnostic["rank_mape_test"], diagnostic["rank_mape_val"])
    diagnostic["stability_penalty"] = (diagnostic["gap_rmse_ratio"].fillna(10.0) - 1.0).clip(lower=0.0)
    diagnostic["robust_score"] = diagnostic["rank_rmse_test"] * (1.0 + 0.35 * diagnostic["stability_penalty"])

    shortlist_cols = [
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
    shortlist = diagnostic[shortlist_cols].copy().sort_values(
        ["robust_score", "rank_rmse_test", "gap_rmse_ratio"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return diagnostic, shortlist


def format_comparison_summary(combined: pd.DataFrame, display: pd.DataFrame) -> str:
    """Return a printable textual summary for baseline vs extended comparison."""

    lines = [
        "",
        "=" * 120,
        "CONFRONTO STEP 3 STATISTICO - Baseline vs Extended",
        "=" * 120,
        display.to_string(index=False),
        "=" * 120,
        "",
    ]

    combined_clean = combined.dropna(subset=["rmse_test_orig"])
    if not combined_clean.empty:
        baseline_pool = combined_clean[combined_clean["experiment"] == "baseline"]
        extended_pool = combined_clean[combined_clean["experiment"] == "extended"]
        if not baseline_pool.empty:
            base_best = baseline_pool.loc[baseline_pool["rmse_test_orig"].idxmin()]
            lines.append(f"Baseline winner (RMSE test originale): {base_best['model']} -> {base_best['rmse_test_orig']:,.1f} GWh")
        if not extended_pool.empty:
            ext_best = extended_pool.loc[extended_pool["rmse_test_orig"].idxmin()]
            lines.append(f"Extended winner (RMSE test originale): {ext_best['model']} -> {ext_best['rmse_test_orig']:,.1f} GWh")
        if not baseline_pool.empty and not extended_pool.empty:
            base_rmse = baseline_pool["rmse_test_orig"].min()
            ext_rmse = extended_pool["rmse_test_orig"].min()
            improvement = base_rmse - ext_rmse
            pct_change = (improvement / base_rmse * 100) if base_rmse > 0 else 0.0
            if improvement > 0:
                lines.append(f"Miglioramento con Extended: {improvement:,.1f} GWh ({pct_change:+.1f}%)")
            else:
                lines.append(f"Extended peggiore: {abs(improvement):,.1f} GWh ({pct_change:+.1f}%)")

    return "\n".join(lines)


def format_diagnostic_summary(shortlist: pd.DataFrame) -> str:
    """Return a printable textual summary for diagnostic shortlist."""

    top_robust = shortlist.iloc[0]
    top_accuracy = shortlist.sort_values(
        ["rank_rmse_test", "rank_mae_test", "rank_mape_test"],
        ascending=[True, True, True],
    ).iloc[0]

    return "\n".join(
        [
            "",
            "=" * 120,
            "STEP 3 STATISTICAL DIAGNOSTICS - BASELINE VS EXTENDED",
            "=" * 120,
            shortlist.to_string(index=False),
            "=" * 120,
            "",
            "Best candidate by ACCURACY (test metrics):",
            f"- Experiment: {top_accuracy['experiment']}",
            f"- Model: {top_accuracy['model']}",
            f"- RMSE val: {top_accuracy['rank_rmse_val']:.1f}",
            f"- RMSE test: {top_accuracy['rank_rmse_test']:.1f}",
            f"- Gap RMSE ratio (test/val): {top_accuracy['gap_rmse_ratio']:.2f}x",
            f"- Robust score: {top_accuracy['robust_score']:.1f}",
            "",
            "Best candidate by ROBUSTNESS (stability-aware):",
            f"- Experiment: {top_robust['experiment']}",
            f"- Model: {top_robust['model']}",
            f"- RMSE val: {top_robust['rank_rmse_val']:.1f}",
            f"- RMSE test: {top_robust['rank_rmse_test']:.1f}",
            f"- Gap RMSE ratio (test/val): {top_robust['gap_rmse_ratio']:.2f}x",
            f"- Robust score: {top_robust['robust_score']:.1f}",
        ]
    )