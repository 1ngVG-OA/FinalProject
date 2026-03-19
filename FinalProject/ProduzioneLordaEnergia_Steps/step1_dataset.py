from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_istat_series(
    csv_path: str | Path,
    target_col_name: str = "Produzione_lorda_totale",
    skiprows: int = 4,
    year_col: int = 0,
    value_col: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ISTAT-style CSV and return raw preview, clean series, and summary."""
    path = Path(csv_path)

    raw_preview = pd.read_csv(path, sep=";", header=None, dtype=str)

    series_df = pd.read_csv(
        path,
        sep=";",
        skiprows=skiprows,
        header=None,
        usecols=[year_col, value_col],
        names=["Anno", target_col_name],
        dtype=str,
        encoding="utf-8",
    )

    series_df["Anno"] = series_df["Anno"].str.extract(r"(\d{4})").astype(int)
    series_df[target_col_name] = (
        series_df[target_col_name]
        .str.strip()
        .replace({"....": pd.NA, "-": pd.NA, "": pd.NA})
        .str.replace(".", "", regex=False)
        .astype("Float64")
    )

    series_df = series_df.sort_values("Anno").set_index("Anno")
    series_df.index = pd.Index(series_df.index, name="Anno")

    summary = series_df[target_col_name].describe().to_frame(name="value")
    summary.loc["missing_values"] = series_df[target_col_name].isna().sum()
    summary.loc["start_year"] = series_df.index.min()
    summary.loc["end_year"] = series_df.index.max()
    summary.loc["n_observations"] = len(series_df)

    return raw_preview, series_df, summary


def plot_step1_overview(series_df: pd.DataFrame, target_col_name: str) -> plt.Figure:
    """Create the Step 1 overview figure with line plot, histogram, and boxplot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    series_df[target_col_name].plot(
        ax=axes[0],
        color="tab:blue",
        lw=2,
        title="Produzione lorda totale",
    )
    axes[0].set_xlabel("Anno")
    axes[0].set_ylabel("GWh")

    series_df[target_col_name].plot(
        kind="hist",
        bins=20,
        ax=axes[1],
        color="tab:orange",
        title="Distribuzione",
    )
    axes[1].set_xlabel("GWh")

    series_df.boxplot(column=target_col_name, ax=axes[2])
    axes[2].set_title("Boxplot")
    axes[2].set_ylabel("GWh")

    fig.tight_layout()
    return fig
