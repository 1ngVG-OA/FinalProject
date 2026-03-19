from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def _adf_report(series: pd.Series, name: str) -> dict:
    stat, pvalue, *_ = adfuller(series.dropna())
    return {
        "series": name,
        "adf_stat": stat,
        "pvalue": pvalue,
        "stationary_at_5pct": pvalue < 0.05,
    }


def run_step2_analysis(series_df: pd.DataFrame, target_col_name: str) -> tuple[pd.Series, dict[str, pd.Series], pd.DataFrame]:
    """Compute candidate transformations and ADF summary."""
    target_series = series_df[target_col_name].astype(float)

    candidate_series = {
        "level": target_series,
        "log": np.log(target_series),
        "diff_1": target_series.diff(),
        "log_diff_1": np.log(target_series).diff(),
        "diff_2": target_series.diff().diff(),
    }

    adf_results = pd.DataFrame(
        [_adf_report(series, name) for name, series in candidate_series.items()]
    ).set_index("series")

    return target_series, candidate_series, adf_results


def plot_step2_transforms(candidate_series: dict[str, pd.Series]) -> plt.Figure:
    """Plot level/log/diff variants used in Step 2."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=False)

    candidate_series["level"].plot(ax=axes[0, 0], color="tab:blue", title="Serie in livello")
    axes[0, 0].set_ylabel("GWh")

    candidate_series["log"].plot(ax=axes[0, 1], color="tab:green", title="Log-serie")
    axes[0, 1].set_ylabel("log(GWh)")

    candidate_series["diff_1"].plot(ax=axes[1, 0], color="tab:orange", title="Prima differenza")
    axes[1, 0].axhline(0, color="black", lw=1, ls="--")

    candidate_series["log_diff_1"].plot(ax=axes[1, 1], color="tab:red", title="Prima differenza della log-serie")
    axes[1, 1].axhline(0, color="black", lw=1, ls="--")

    candidate_series["diff_2"].plot(ax=axes[2, 0], color="tab:purple", title="Seconda differenza")
    axes[2, 0].axhline(0, color="black", lw=1, ls="--")

    fig.tight_layout()
    return fig


def plot_step2_acf_pacf_grid(candidate_series: dict[str, pd.Series], lags: int = 20) -> plt.Figure:
    """Plot ACF/PACF grid for all candidate transformations."""
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))

    plot_acf(candidate_series["level"].dropna(), lags=lags, ax=axes[0, 0])
    axes[0, 0].set_title("ACF - livello")
    plot_pacf(candidate_series["level"].dropna(), lags=lags, ax=axes[0, 1], method="ywm")
    axes[0, 1].set_title("PACF - livello")

    plot_acf(candidate_series["log"].dropna(), lags=lags, ax=axes[1, 0])
    axes[1, 0].set_title("ACF - log-serie")
    plot_pacf(candidate_series["log"].dropna(), lags=lags, ax=axes[1, 1], method="ywm")
    axes[1, 1].set_title("PACF - log-serie")

    plot_acf(candidate_series["log_diff_1"].dropna(), lags=lags, ax=axes[2, 0])
    axes[2, 0].set_title("ACF - log diff(1)")
    plot_pacf(candidate_series["log_diff_1"].dropna(), lags=lags, ax=axes[2, 1], method="ywm")
    axes[2, 1].set_title("PACF - log diff(1)")

    plot_acf(candidate_series["diff_1"].dropna(), lags=lags, ax=axes[3, 0])
    axes[3, 0].set_title("ACF - diff(1)")
    plot_pacf(candidate_series["diff_1"].dropna(), lags=lags, ax=axes[3, 1], method="ywm")
    axes[3, 1].set_title("PACF - diff(1)")

    plot_acf(candidate_series["diff_2"].dropna(), lags=lags, ax=axes[4, 0])
    axes[4, 0].set_title("ACF - diff(2)")
    plot_pacf(candidate_series["diff_2"].dropna(), lags=lags, ax=axes[4, 1], method="ywm")
    axes[4, 1].set_title("PACF - diff(2)")

    fig.tight_layout()
    return fig


def plot_diff2_focus(candidate_series: dict[str, pd.Series], lags: int = 20) -> tuple[plt.Figure, plt.Figure]:
    """Return dedicated figures for diff(2) line and diff(2) ACF/PACF."""
    fig1, ax = plt.subplots(figsize=(14, 4))
    candidate_series["diff_2"].plot(ax=ax, color="tab:purple", title="Seconda differenza")
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Anno")
    ax.set_ylabel("Delta^2 GWh")
    fig1.tight_layout()

    fig2, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(candidate_series["diff_2"].dropna(), lags=lags, ax=axes[0])
    axes[0].set_title("ACF - diff(2)")
    plot_pacf(candidate_series["diff_2"].dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title("PACF - diff(2)")
    fig2.tight_layout()

    return fig1, fig2
