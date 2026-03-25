"""Descriptive analysis utilities for Tavola_1.14.

This module implements Step 1 of the project pipeline and provides a
script-friendly alternative to exploratory notebooks.

Main responsibilities:
1) load and clean the target annual series from the ISTAT-style CSV,
2) compute descriptive statistics (frequency, tendency, dispersion),
3) detect global and local outliers,
4) generate static plots and persist all outputs under ``Results/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


TARGET_COLUMN_INDEX = {
    "production_total": 1,
}

TARGET_SERIES_NAME = {
    "production_total": "produzione_lorda_totale",
}


@dataclass(frozen=True)
class DescriptivePaths:
    """Filesystem paths required by :func:`run_descriptive_analysis`.

    Attributes:
        dataset_path: Input CSV path.
        results_metrics_dir: Output directory for CSV metric tables.
        results_plots_dir: Output directory for PNG plots.
    """

    dataset_path: Path
    results_metrics_dir: Path
    results_plots_dir: Path


def _parse_istat_number(value: str) -> float:
    """Parse ISTAT-style numeric strings into float.

    Args:
        value: Raw string value from CSV.

    Returns:
        Parsed numeric value as float, or ``np.nan`` for missing/non-parseable
        tokens.

    Examples:
        - ``"1.150"`` -> ``1150.0``
        - ``"2.575"`` -> ``2575.0``
        - ``"...."``  -> ``np.nan``
        - ``"-"``     -> ``np.nan``
    """

    if value is None:
        return np.nan

    s = str(value).strip()
    if s in {"", "....", "-"}:
        return np.nan

    # The dataset uses dot as thousands separator.
    s = s.replace(".", "")
    s = s.replace(",", ".")

    try:
        return float(s)
    except ValueError:
        return np.nan


def load_target_series(dataset_path: Path, target: str = "production_total") -> pd.Series:
    """Load annual target series from Tavola_1.14.csv.

    The CSV contains multi-row headers; valid data rows are identified by a
    4-digit year in column 0.

    Args:
        dataset_path: Absolute or relative path to ``Tavola_1.14.csv``.
        target: Logical target key. Supported value: ``"production_total"``.

    Returns:
        Clean annual series indexed by year.

    Raises:
        ValueError: If ``target`` is not supported.

    Supported target:
    - production_total: "Produzione lorda - Totale" (column index 1).
    """

    if target not in TARGET_COLUMN_INDEX:
        options = ", ".join(sorted(TARGET_COLUMN_INDEX))
        raise ValueError(f"Unsupported target '{target}'. Available targets: {options}")

    raw = pd.read_csv(dataset_path, sep=";", header=None, dtype=str)

    year_mask = raw[0].astype(str).str.fullmatch(r"\d{4}")
    target_col = TARGET_COLUMN_INDEX[target]
    data = raw.loc[year_mask, [0, target_col]].copy()
    data.columns = ["year", "value"]

    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    data["value"] = data["value"].map(_parse_istat_number)

    data = data.dropna(subset=["year", "value"]).copy()
    data["year"] = data["year"].astype(int)

    series_name = TARGET_SERIES_NAME.get(target, "target_series")
    series = pd.Series(data["value"].values, index=data["year"].values, name=series_name)
    series.index.name = "year"
    return series


def _frequency_distribution(series: pd.Series, n_bins: int | None = None) -> pd.DataFrame:
    """Build binned absolute and relative frequency distribution.

    Args:
        series: Input numeric series.
        n_bins: Number of bins. If ``None``, Sturges' rule is used.

    Returns:
        DataFrame with class interval, absolute frequency and relative frequency.
    """

    x = series.dropna()
    if n_bins is None:
        n_bins = int(np.ceil(np.log2(len(x)) + 1))

    categories = pd.cut(x, bins=n_bins, include_lowest=True)
    abs_freq = categories.value_counts(sort=False)
    rel_freq = abs_freq / abs_freq.sum()

    return pd.DataFrame(
        {
            "class_interval": abs_freq.index.astype(str),
            "absolute_frequency": abs_freq.values,
            "relative_frequency": rel_freq.values,
        }
    )


def _central_tendency(series: pd.Series) -> pd.DataFrame:
    """Compute central tendency measures.

    Args:
        series: Input numeric series.

    Returns:
        One-row DataFrame containing mean, median and mode.
    """

    x = series.dropna()
    modes = x.mode()
    mode_value = float(modes.iloc[0]) if not modes.empty else np.nan

    return pd.DataFrame(
        [
            {
                "mean": float(x.mean()),
                "median": float(x.median()),
                "mode": mode_value,
            }
        ]
    )


def _dispersion_measures(series: pd.Series) -> pd.DataFrame:
    """Compute dispersion indicators.

    Args:
        series: Input numeric series.

    Returns:
        One-row DataFrame with range, variance, standard deviation,
        coefficient of variation and IQR.
    """

    x = series.dropna()
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1

    mean_value = float(x.mean())
    std_value = float(x.std(ddof=1))
    cv_value = std_value / mean_value if mean_value != 0 else np.nan

    return pd.DataFrame(
        [
            {
                "range": float(x.max() - x.min()),
                "variance": float(x.var(ddof=1)),
                "std_dev": std_value,
                "coefficient_of_variation": cv_value,
                "iqr": iqr,
            }
        ]
    )


def _outlier_table_iqr(series: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect global outliers using the IQR rule.

    Args:
        series: Input numeric series.

    Returns:
        A tuple ``(details, summary)`` where:
        - ``details`` lists outlier years and values,
        - ``summary`` stores fences and aggregate outlier counts/ratios.
    """

    x = series.dropna()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (x < lower) | (x > upper)
    outliers = x.loc[outlier_mask]

    details = pd.DataFrame(
        {
            "year": outliers.index,
            "value": outliers.values,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "lower_fence": float(lower),
                "upper_fence": float(upper),
                "num_outliers": int(outlier_mask.sum()),
                "outlier_ratio": float(outlier_mask.mean()),
            }
        ]
    )

    return details, summary


def _trend_validation(series: pd.Series) -> pd.DataFrame:
    """Evaluate long-run trend with linear and rank-based statistics.

    Args:
        series: Input annual series on original scale.

    Returns:
        One-row DataFrame with slope, p-values, R^2, Spearman correlation and
        year-over-year sign shares.
    """

    x = series.dropna().astype(float)
    years = x.index.to_numpy(dtype=float)
    values = x.to_numpy(dtype=float)

    slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
    spearman_rho, spearman_p = stats.spearmanr(years, values)

    diff = x.diff().dropna()
    positive_share = float((diff > 0).mean()) if len(diff) else np.nan
    negative_share = float((diff < 0).mean()) if len(diff) else np.nan
    zero_share = float((diff == 0).mean()) if len(diff) else np.nan

    return pd.DataFrame(
        [
            {
                "n_observations": int(len(x)),
                "start_year": int(x.index.min()),
                "end_year": int(x.index.max()),
                "slope_per_year": float(slope),
                "slope_p_value": float(p_value),
                "r_squared": float(r_value**2),
                "spearman_rho": float(spearman_rho),
                "spearman_p_value": float(spearman_p),
                "positive_yoy_share": positive_share,
                "negative_yoy_share": negative_share,
                "zero_yoy_share": zero_share,
                "intercept": float(intercept),
                "slope_std_err": float(std_err),
            }
        ]
    )


def _local_outliers_on_variation(
    series: pd.Series,
    window: int = 11,
    threshold: float = 3.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Detect local anomalies on YoY changes using robust local scoring.

    The method uses rolling MAD on residuals from a rolling median baseline,
    with a rolling-std fallback when MAD is near zero.

    Args:
        series: Input annual level series.
        window: Centered rolling window size.
        threshold: Absolute score threshold for anomaly flagging.

    Returns:
        A tuple ``(details, local_outliers, summary)``.
    """

    x = series.dropna().astype(float)
    yoy = x.diff().dropna()

    min_periods = max(5, window // 2)
    rolling_median = yoy.rolling(window=window, center=True, min_periods=min_periods).median()
    residual = yoy - rolling_median

    rolling_mad = residual.abs().rolling(window=window, center=True, min_periods=min_periods).median()
    rolling_std = residual.rolling(window=window, center=True, min_periods=min_periods).std(ddof=0)
    eps = 1e-9
    modified_z = 0.6745 * residual / (rolling_mad + eps)
    rolling_z = residual / (rolling_std + eps)

    use_std_fallback = rolling_mad.fillna(0.0) < 1e-6
    combined_score = modified_z.mask(use_std_fallback, rolling_z)

    flag = combined_score.abs() > threshold

    details = pd.DataFrame(
        {
            "year": yoy.index,
            "yoy_change": yoy.values,
            "local_median_change": rolling_median.values,
            "residual_vs_local": residual.values,
            "rolling_mad": rolling_mad.values,
            "local_score": combined_score.values,
            "is_local_outlier": flag.fillna(False).values,
        }
    )

    local_outliers = details.loc[details["is_local_outlier"]].copy()

    summary = pd.DataFrame(
        [
            {
                "window": int(window),
                "threshold": float(threshold),
                "n_yoy_points": int(len(yoy)),
                "num_local_outliers": int(details["is_local_outlier"].sum()),
                "local_outlier_ratio": float(details["is_local_outlier"].mean()),
                "yoy_mean": float(yoy.mean()),
                "yoy_std": float(yoy.std(ddof=1)),
                "yoy_q05": float(yoy.quantile(0.05)),
                "yoy_q95": float(yoy.quantile(0.95)),
            }
        ]
    )

    return details, local_outliers, summary


def _save_distribution_plots(series: pd.Series, freq_df: pd.DataFrame, out_dir: Path) -> None:
    """Generate and save descriptive plots to disk.

    Args:
        series: Clean target series.
        freq_df: Frequency distribution table used by frequency plot.
        out_dir: Destination directory for PNG files.

    Returns:
        None. Files are written as side effects.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    x = series.dropna()

    # 0) Baseline time-series plot to inspect raw trend/scale over years.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x.index, x.values, color="tab:blue", linewidth=2)
    series_label = str(series.name or "target_series").replace("_", " ").strip()
    ax.set_title(f"Base Time Series - {series_label}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "series_base.png", dpi=150)
    plt.close(fig)

    # 1) Frequency distribution bar chart (empirical absolute + relative).
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    idx = np.arange(len(freq_df))

    bars = ax1.bar(idx, freq_df["absolute_frequency"], alpha=0.75, label="Absolute")
    ax2.plot(idx, freq_df["relative_frequency"], color="tab:red", marker="o", label="Relative")

    ax1.set_title("Frequency Distribution (Binned)")
    ax1.set_xlabel("Class interval")
    ax1.set_ylabel("Absolute frequency")
    ax2.set_ylabel("Relative frequency")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(freq_df["class_interval"], rotation=80)

    ax1.legend([bars], ["Absolute"], loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "frequency_distribution.png", dpi=150)
    plt.close(fig)

    # 2) Histogram + KDE + fitted normal and uniform densities.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=20, density=True, alpha=0.5, color="tab:blue", label="Empirical density")

    kde = stats.gaussian_kde(x)
    xx = np.linspace(x.min(), x.max(), 400)
    ax.plot(xx, kde(xx), color="tab:green", linewidth=2, label="KDE")

    mu, sigma = x.mean(), x.std(ddof=1)
    ax.plot(xx, stats.norm.pdf(xx, loc=mu, scale=sigma), color="tab:orange", linewidth=2, label="Normal fit")

    a, b = x.min(), x.max()
    ax.plot(xx, stats.uniform.pdf(xx, loc=a, scale=b - a), color="tab:red", linewidth=2, linestyle="--", label="Uniform fit")

    ax.set_title("Empirical Density vs Normal/Uniform")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "density_comparison.png", dpi=150)
    plt.close(fig)

    # 3) Discrete empirical distribution (top 30 frequencies to keep readability).
    discrete = x.value_counts(normalize=True).sort_values(ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(discrete.index.astype(str), discrete.values, color="tab:purple", alpha=0.8)
    ax.set_title("Discrete Empirical Distribution (Top 30 values)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Relative frequency")
    ax.tick_params(axis="x", rotation=80)
    fig.tight_layout()
    fig.savefig(out_dir / "discrete_distribution.png", dpi=150)
    plt.close(fig)

    # 4) Boxplot + Q-Q plot for outlier/normality inspection.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot(x, vert=True)
    axes[0].set_title("Boxplot")
    axes[0].set_ylabel("Value")

    stats.probplot(x, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Normal)")

    fig.tight_layout()
    fig.savefig(out_dir / "outliers_qqplot.png", dpi=150)
    plt.close(fig)

    # 4b) Standalone global outlier boxplot (levels) for direct comparison.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(x, vert=True)
    ax.set_title("Global Outliers - Boxplot on Levels")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(out_dir / "global_outliers_boxplot.png", dpi=150)
    plt.close(fig)

    # 5) Trend validation plot: observed series + linear trend fit.
    years = x.index.to_numpy(dtype=float)
    values = x.to_numpy(dtype=float)
    slope, intercept, _, _, _ = stats.linregress(years, values)
    trend_line = intercept + slope * years

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x.index, x.values, color="tab:blue", linewidth=2, label="Observed")
    ax.plot(x.index, trend_line, color="tab:orange", linestyle="--", linewidth=2, label="Linear trend")
    ax.set_title("Trend Validation: Observed vs Linear Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trend_validation.png", dpi=150)
    plt.close(fig)

    # 6) Local anomaly plot on year-over-year changes.
    yoy = x.diff().dropna()
    rolling_median = yoy.rolling(window=11, center=True, min_periods=5).median()
    residual = yoy - rolling_median
    rolling_mad = residual.abs().rolling(window=11, center=True, min_periods=5).median()
    rolling_std = residual.rolling(window=11, center=True, min_periods=5).std(ddof=0)
    modified_z = 0.6745 * residual / (rolling_mad + 1e-9)
    rolling_z = residual / (rolling_std + 1e-9)
    use_std_fallback = rolling_mad.fillna(0.0) < 1e-6
    local_score = modified_z.mask(use_std_fallback, rolling_z)
    local_outlier_mask = local_score.abs() > 3.5

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yoy.index, yoy.values, color="tab:blue", linewidth=1.8, label="YoY change")
    ax.plot(
        rolling_median.index,
        rolling_median.values,
        color="tab:green",
        linestyle="--",
        linewidth=1.8,
        label="Rolling median (local baseline)",
    )
    ax.scatter(
        yoy.index[local_outlier_mask],
        yoy[local_outlier_mask],
        color="tab:red",
        s=45,
        label="Local outliers",
        zorder=3,
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("Local Outliers on Year-over-Year Changes")
    ax.set_xlabel("Year")
    ax.set_ylabel("YoY change")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "local_outliers_yoy.png", dpi=150)
    plt.close(fig)

    # 6b) Local outlier boxplot on YoY changes for same-typology comparison.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(yoy, vert=True)
    ax.set_title("Local Outliers - Boxplot on YoY Changes")
    ax.set_ylabel("YoY change")
    fig.tight_layout()
    fig.savefig(out_dir / "local_outliers_boxplot.png", dpi=150)
    plt.close(fig)


def run_descriptive_analysis(
    paths: DescriptivePaths,
    target: str = "production_total",
) -> dict[str, Path]:
    """Run full descriptive analysis and persist CSV/PNG artifacts.

    Args:
        paths: Input/output path container.
        target: Target key to extract from source CSV.

    Returns:
        Mapping of artifact logical names to output paths.
    """

    paths.results_metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.results_plots_dir.mkdir(parents=True, exist_ok=True)

    series = load_target_series(paths.dataset_path, target=target)

    freq_df = _frequency_distribution(series)
    central_df = _central_tendency(series)
    dispersion_df = _dispersion_measures(series)
    outliers_df, outlier_summary_df = _outlier_table_iqr(series)
    trend_summary_df = _trend_validation(series)
    yoy_details_df, local_outliers_df, local_outliers_summary_df = _local_outliers_on_variation(series)

    # Save tabular outputs.
    output_paths = {
        "series": paths.results_metrics_dir / "series_clean.csv",
        "frequency": paths.results_metrics_dir / "frequency_distribution.csv",
        "central_tendency": paths.results_metrics_dir / "central_tendency.csv",
        "dispersion": paths.results_metrics_dir / "dispersion_measures.csv",
        "outliers": paths.results_metrics_dir / "outliers_iqr.csv",
        "outlier_summary": paths.results_metrics_dir / "outliers_summary.csv",
        "trend_summary": paths.results_metrics_dir / "trend_summary.csv",
        "yoy_variation": paths.results_metrics_dir / "yoy_variation_details.csv",
        "local_outliers": paths.results_metrics_dir / "local_outliers_yoy.csv",
        "local_outlier_summary": paths.results_metrics_dir / "local_outliers_summary.csv",
    }

    series.reset_index().to_csv(output_paths["series"], index=False)
    freq_df.to_csv(output_paths["frequency"], index=False)
    central_df.to_csv(output_paths["central_tendency"], index=False)
    dispersion_df.to_csv(output_paths["dispersion"], index=False)
    outliers_df.to_csv(output_paths["outliers"], index=False)
    outlier_summary_df.to_csv(output_paths["outlier_summary"], index=False)
    trend_summary_df.to_csv(output_paths["trend_summary"], index=False)
    yoy_details_df.to_csv(output_paths["yoy_variation"], index=False)
    local_outliers_df.to_csv(output_paths["local_outliers"], index=False)
    local_outliers_summary_df.to_csv(output_paths["local_outlier_summary"], index=False)

    _save_distribution_plots(series, freq_df, paths.results_plots_dir)

    return output_paths
