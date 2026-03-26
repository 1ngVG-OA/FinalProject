"""Plot utilities for torch-based neural forecasting models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .model_config import invert_preprocessed_segment


def _suffix_name(base_name: str, suffix: str | None) -> str:
    if not suffix:
        return base_name
    return f"{base_name}_{suffix}"


def save_neural_plots(output: dict[str, Any], out_dir: Path, suffix: str | None = None) -> dict[str, Path]:
    """Save transformed-scale and original-scale forecast plots for Step 5."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "neural_plot_forecasts": out_dir / f"{_suffix_name('forecast_comparison', suffix)}.png",
        "neural_plot_forecasts_original_scale": out_dir / f"{_suffix_name('forecast_original_scale', suffix)}.png",
    }

    summary = output["summary"]
    forecasts = output["forecast_table"]
    val_actual = output["validation_actual"]
    test_actual = output["test_actual"]

    model_names = list(summary["model"].astype(str))
    val_len = len(val_actual)
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(val_actual.index, val_actual.values, color="black", linewidth=2.0, label="actual_val")
    ax.plot(test_actual.index, test_actual.values, color="dimgray", linewidth=2.0, label="actual_test")

    for idx, model_name in enumerate(model_names):
        color = palette[idx % len(palette)]
        pred_col = f"{model_name}_pred"
        pred_all = forecasts[pred_col].to_numpy()
        ax.plot(val_actual.index, pred_all[:val_len], color=color, linestyle="--", linewidth=1.6, label=f"{model_name}_val")
        ax.plot(test_actual.index, pred_all[val_len:], color=color, linewidth=2.0, label=f"{model_name}_test")

    ax.axvline(val_actual.index.max(), color="gray", linestyle=":", linewidth=1.0)
    ax.set_title("Step 5 - Neural Forecasts")
    ax.set_xlabel("Time")
    ax.set_ylabel("Preprocessed value")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(paths["neural_plot_forecasts"], dpi=150)
    plt.close(fig)

    original_series = output.get("original_series")
    preprocessing_config = output.get("preprocessing_config")
    if isinstance(original_series, pd.Series) and original_series is not None and preprocessing_config is not None:
        raw = pd.to_numeric(original_series, errors="coerce").dropna().astype(float)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(raw.index, raw.values, color="black", linewidth=2.2, label="serie_originale", zorder=5)
        ax.axvspan(int(val_actual.index.min()), int(val_actual.index.max()) + 1, alpha=0.07, color="tab:blue", label="_nolegend_")
        ax.axvspan(int(test_actual.index.min()), int(test_actual.index.max()) + 1, alpha=0.09, color="tab:orange", label="_nolegend_")

        plotted_any = False
        for idx, model_name in enumerate(model_names):
            color = palette[idx % len(palette)]
            pred_col = f"{model_name}_pred"
            pred_all = forecasts[pred_col].to_numpy()
            val_pred = pd.Series(pred_all[:val_len], index=val_actual.index)
            test_pred = pd.Series(pred_all[val_len:], index=test_actual.index)

            val_orig = invert_preprocessed_segment(val_pred, original_series, preprocessing_config)
            test_orig = invert_preprocessed_segment(test_pred, original_series, preprocessing_config)
            if val_orig is None or test_orig is None:
                continue

            plotted_any = True
            ax.plot(val_orig.index, val_orig.values, color=color, linestyle="--", linewidth=1.6, label=f"{model_name}_val_orig")
            ax.plot(test_orig.index, test_orig.values, color=color, linewidth=2.0, label=f"{model_name}_test_orig")

        if plotted_any:
            ax.set_title("Step 5 - Neural Forecasts on Original Scale")
            ax.set_xlabel("Time")
            ax.set_ylabel("Original value")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", ncol=2)
            fig.tight_layout()
            fig.savefig(paths["neural_plot_forecasts_original_scale"], dpi=150)
        plt.close(fig)

    return paths