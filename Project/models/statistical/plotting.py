"""Utility di plotting per gli output statistici dello Step 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .model_config import invert_diff2_log1p


def save_statistical_plots(
    output: dict[str, Any],
    out_dir: Path,
    suffix: str | None = None,
) -> dict[str, Path]:
    """Salva i grafici di confronto forecast e diagnostica dei residui."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name_suffix = f"_{suffix}" if suffix else ""

    paths = {
        "stat_plot_forecasts": out_dir / f"forecast_comparison{name_suffix}.png",
        "stat_plot_residuals": out_dir / f"residuals_diagnostics{name_suffix}.png",
        "stat_plot_forecasts_original_scale": out_dir / f"forecast_original_scale{name_suffix}.png",
    }

    val_idx = output["validation_actual"].index
    test_idx = output["test_actual"].index

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(val_idx, output["validation_actual"].values, color="black", linewidth=2, label="actual_val")
    ax.plot(test_idx, output["test_actual"].values, color="dimgray", linewidth=2, label="actual_test")
    ax.plot(val_idx, output["sarima_val_pred"].values, color="tab:blue", linestyle="--", label="sarima_val")
    ax.plot(test_idx, output["sarima_test_pred"].values, color="tab:blue", label="sarima_test")
    ax.axvline(val_idx.max(), color="gray", linestyle=":", linewidth=1)
    ax.set_title("Step 3 - SARIMA Forecasts")
    ax.set_xlabel("Time")
    ax.set_ylabel("Transformed value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["stat_plot_forecasts"], dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Diagnostica residui (ACF/PACF)
    # ------------------------------------------------------------------

    def _acf_lags(residuals: pd.Series) -> int:
        return max(1, min(20, len(residuals) - 1))

    def _pacf_lags(residuals: pd.Series) -> int:
        return max(1, min(20, (len(residuals) // 2) - 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(output["sarima_validation_residuals"], lags=_acf_lags(output["sarima_validation_residuals"]), ax=axes[0])
    axes[0].set_title("SARIMA Residuals ACF")
    plot_pacf(
        output["sarima_validation_residuals"],
        lags=_pacf_lags(output["sarima_validation_residuals"]),
        ax=axes[1],
        method="ywm",
    )
    axes[1].set_title("SARIMA Residuals PACF")

    fig.tight_layout()
    fig.savefig(paths["stat_plot_residuals"], dpi=150)
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
        and diff_order in (1, 2)
    ):
        raw = pd.to_numeric(original_series, errors="coerce").dropna().astype(float)
        x_log = np.log1p(raw)

        val_start = int(val_idx.min())
        test_start = int(test_idx.min())

        if diff_order == 1:
            seed_log_val = float(x_log[x_log.index < val_start].iloc[-1])
            seed_log_test = float(x_log[x_log.index < test_start].iloc[-1])

            sarima_val_orig = np.expm1(seed_log_val + output["sarima_val_pred"].cumsum())
            sarima_test_orig = np.expm1(seed_log_test + output["sarima_test_pred"].cumsum())
        else:
            x_d1 = x_log.diff().dropna()
            seed_d1_val = float(x_d1[x_d1.index < val_start].iloc[-1])
            seed_log_val = float(x_log[x_log.index < val_start].iloc[-1])
            seed_d1_test = float(x_d1[x_d1.index < test_start].iloc[-1])
            seed_log_test = float(x_log[x_log.index < test_start].iloc[-1])

            sarima_val_orig = invert_diff2_log1p(output["sarima_val_pred"], seed_d1_val, seed_log_val)
            sarima_test_orig = invert_diff2_log1p(output["sarima_test_pred"], seed_d1_test, seed_log_test)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(raw.index, raw.values, color="black", linewidth=2.2, label="serie_originale", zorder=4)
        ax.axvspan(int(val_idx.min()), int(val_idx.max()) + 1, alpha=0.07, color="tab:blue", label="_nolegend_")
        ax.axvspan(int(test_idx.min()), int(test_idx.max()) + 1, alpha=0.09, color="tab:orange", label="_nolegend_")

        ax.plot(sarima_val_orig.index, sarima_val_orig.values, color="tab:blue", linestyle="--", linewidth=1.8, label="sarima_val_orig")
        ax.plot(sarima_test_orig.index, sarima_test_orig.values, color="tab:blue", linewidth=2.2, label="sarima_test_orig")

        ax.set_title("Step 3 - SARIMA Forecasts on Original Scale")
        ax.set_xlabel("Time")
        ax.set_ylabel("Original value")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(paths["stat_plot_forecasts_original_scale"], dpi=150)
        plt.close(fig)

    return paths
