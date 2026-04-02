"""Utility di plotting per i modelli ML non neurali dello Step 4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model_config import invert_diff2_log1p


def _suffix_name(base_name: str, suffix: str | None) -> str:
    if not suffix:
        return base_name
    return f"{base_name}_{suffix}"


def save_ml_plots(output: dict[str, Any], out_dir: Path, suffix: str | None = None) -> dict[str, Path]:
    """Salva grafici di confronto in scala trasformata e originale."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "ml_plot_forecasts": out_dir / f"{_suffix_name('forecast_comparison', suffix)}.png",
        "ml_plot_forecasts_original_scale": out_dir / f"{_suffix_name('forecast_original_scale', suffix)}.png",
    }

    summary = output["summary"]
    forecasts = output["forecast_table"]
    val_actual = output["validation_actual"]
    test_actual = output["test_actual"]

    model_names = list(summary["model"].astype(str))
    val_len = len(val_actual)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(val_actual.index, val_actual.values, color="black", linewidth=2, label="actual_val")
    ax.plot(test_actual.index, test_actual.values, color="dimgray", linewidth=2, label="actual_test")

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown"]
    for i, model_name in enumerate(model_names):
        color = palette[i % len(palette)]
        pred_col = f"{model_name}_pred"
        pred_all = forecasts[pred_col].to_numpy()
        val_pred = pred_all[:val_len]
        test_pred = pred_all[val_len:]
        ax.plot(val_actual.index, val_pred, color=color, linestyle="--", linewidth=1.6, label=f"{model_name}_val")
        ax.plot(test_actual.index, test_pred, color=color, linewidth=2.0, label=f"{model_name}_test")

    ax.axvline(val_actual.index.max(), color="gray", linestyle=":", linewidth=1)
    ax.set_title("Step 4 - Non-Neural ML Forecasts")
    ax.set_xlabel("Time")
    ax.set_ylabel("Transformed value")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(paths["ml_plot_forecasts"], dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Forecast in scala originale (quando inversione disponibile)
    # ------------------------------------------------------------------

    original_series = output.get("original_series")
    use_log1p = bool(output.get("use_log1p", False))
    diff_order = int(output.get("diff_order", 0))

    if (
        isinstance(original_series, pd.Series)
        and not original_series.empty
        and use_log1p
        and diff_order in (0, 1, 2)
    ):
        raw = pd.to_numeric(original_series, errors="coerce").dropna().astype(float)
        x_log = pd.Series(np.log1p(raw.to_numpy(dtype=float)), index=raw.index, name="log1p")

        val_start = int(val_actual.index.min())
        test_start = int(test_actual.index.min())

        def invert_segment(pred: pd.Series, segment: str) -> pd.Series:
            if diff_order == 0:
                return pd.Series(np.expm1(pred.to_numpy(dtype=float)), index=pred.index, name="pred_orig")
            if diff_order == 1:
                start = val_start if segment == "val" else test_start
                seed_log = float(x_log[x_log.index < start].iloc[-1])
                pred_log = seed_log + pred.cumsum()
                return pd.Series(np.expm1(pred_log.to_numpy(dtype=float)), index=pred.index, name="pred_orig")

            x_d1 = x_log.diff().dropna()
            start = val_start if segment == "val" else test_start
            seed_d1 = float(x_d1[x_d1.index < start].iloc[-1])
            seed_log = float(x_log[x_log.index < start].iloc[-1])
            return invert_diff2_log1p(pred, seed_d1, seed_log)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(raw.index, raw.to_numpy(dtype=float), color="black", linewidth=2.2, label="serie_originale", zorder=5)
        ax.axvspan(int(val_actual.index.min()), int(val_actual.index.max()) + 1, alpha=0.07, color="tab:blue", label="_nolegend_")
        ax.axvspan(int(test_actual.index.min()), int(test_actual.index.max()) + 1, alpha=0.09, color="tab:orange", label="_nolegend_")

        for i, model_name in enumerate(model_names):
            color = palette[i % len(palette)]
            pred_col = f"{model_name}_pred"
            pred_all = forecasts[pred_col].to_numpy()
            val_pred = pd.Series(pred_all[:val_len], index=val_actual.index)
            test_pred = pd.Series(pred_all[val_len:], index=test_actual.index)

            val_orig = invert_segment(val_pred, "val")
            test_orig = invert_segment(test_pred, "test")
            ax.plot(val_orig.index, val_orig.to_numpy(dtype=float), color=color, linestyle="--", linewidth=1.6, label=f"{model_name}_val_orig")
            ax.plot(test_orig.index, test_orig.to_numpy(dtype=float), color=color, linewidth=2.0, label=f"{model_name}_test_orig")

        ax.set_title("Step 4 - Non-Neural ML Forecasts on Original Scale")
        ax.set_xlabel("Time")
        ax.set_ylabel("Original value")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", ncol=2)
        fig.tight_layout()
        fig.savefig(paths["ml_plot_forecasts_original_scale"], dpi=150)
        plt.close(fig)

    return paths
