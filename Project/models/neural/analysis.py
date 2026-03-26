"""Comparison helpers for neural baseline and extended experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


COLS_DISPLAY = [
    "experiment",
    "model",
    "lookback",
    "best_epoch",
    "rmse_val_orig",
    "mae_val_orig",
    "mape_val_orig",
    "abs_mbe_val_orig",
    "rmse_test_orig",
    "mae_test_orig",
    "mape_test_orig",
    "abs_mbe_test_orig",
]

RENAME = {
    "best_epoch": "best_epoch",
    "rmse_val_orig": "RMSE_val(GWh)",
    "mae_val_orig": "MAE_val(GWh)",
    "mape_val_orig": "MAPE_val(%)",
    "abs_mbe_val_orig": "|MBE|_val(GWh)",
    "rmse_test_orig": "RMSE_test(GWh)",
    "mae_test_orig": "MAE_test(GWh)",
    "mape_test_orig": "MAPE_test(%)",
    "abs_mbe_test_orig": "|MBE|_test(GWh)",
}


def _load_summary(csv_path: Path, experiment_label: str) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["experiment"] = experiment_label
    df["lookback"] = df["lookback"].astype(str)
    return df


def build_baseline_extended_comparison(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build baseline-vs-extended neural comparison tables."""

    baseline = _load_summary(metrics_dir / "summary_baseline.csv", "step5_baseline")
    extended = _load_summary(metrics_dir / "summary_extended.csv", "step5_extended")

    available = [df for df in (baseline, extended) if not df.empty]
    if not available:
        raise FileNotFoundError("No neural summary files were found for comparison.")

    combined = pd.concat(available, ignore_index=True, sort=False)
    combined["_experiment_order"] = combined["experiment"].map(
        {"step5_baseline": 0, "step5_extended": 1}
    ).fillna(2)
    combined["_sort_rmse"] = pd.to_numeric(combined["rmse_test_orig"], errors="coerce")
    combined = combined.sort_values(
        ["_sort_rmse", "_experiment_order", "model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    cols = [c for c in COLS_DISPLAY if c in combined.columns]
    display = combined[cols].copy()
    numeric_cols = [c for c in cols if pd.api.types.is_float_dtype(display[c])]
    display[numeric_cols] = display[numeric_cols].round(1)
    display = display.rename(columns=RENAME)
    return combined, display


def format_comparison_summary(combined: pd.DataFrame, display: pd.DataFrame) -> str:
    """Format a console summary for neural baseline/extended comparison."""

    lines = [
        "",
        "=" * 110,
        "CONFRONTO MODELLI — Step 5 Neural baseline vs extended — scala originale (GWh)",
        "=" * 110,
        display.to_string(index=False),
        "=" * 110,
        "",
        "Chiave:",
        "  RMSE / MAE      — scala originale (GWh). Più basso = migliore.",
        "  MAPE            — errore percentuale medio.",
        "  |MBE|           — bias assoluto. Vicino a 0 = previsore non distorto sistematicamente.",
        "  best_epoch      — epoca effettiva selezionata tramite early stopping.",
        "",
    ]

    winner_pool = combined.dropna(subset=["_sort_rmse"])
    if not winner_pool.empty:
        best = winner_pool.loc[winner_pool["_sort_rmse"].idxmin()]
        lines.extend(
            [
                "Miglior modello su RMSE test (scala originale):",
                f"  {best['model']} [{best['experiment']}] -> {best['_sort_rmse']:,.1f} GWh",
                "",
            ]
        )

    return "\n".join(lines)