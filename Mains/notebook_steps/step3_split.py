from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def temporal_split(
    series: pd.Series,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Chronologically split a series into train/validation/test partitions."""
    n_obs = len(series)
    train_end = int(n_obs * train_ratio)
    val_end = int(n_obs * (train_ratio + val_ratio))

    train_series = series.iloc[:train_end].copy()
    val_series = series.iloc[train_end:val_end].copy()
    test_series = series.iloc[val_end:].copy()

    split_summary = pd.DataFrame(
        {
            "subset": ["train", "validation", "test"],
            "n_obs": [len(train_series), len(val_series), len(test_series)],
            "start_year": [
                train_series.index.min(),
                val_series.index.min(),
                test_series.index.min(),
            ],
            "end_year": [
                train_series.index.max(),
                val_series.index.max(),
                test_series.index.max(),
            ],
        }
    )
    return train_series, val_series, test_series, split_summary


def plot_split(train_series: pd.Series, val_series: pd.Series, test_series: pd.Series) -> plt.Figure:
    """Visualize chronological split boundaries."""
    fig, ax = plt.subplots(figsize=(14, 5))

    train_series.plot(ax=ax, label="Train", color="tab:blue", lw=2)
    val_series.plot(ax=ax, label="Validation", color="tab:orange", lw=2)
    test_series.plot(ax=ax, label="Test", color="tab:green", lw=2)

    ax.axvline(val_series.index.min(), color="tab:orange", ls="--", lw=1.5)
    ax.axvline(test_series.index.min(), color="tab:green", ls="--", lw=1.5)
    ax.set_title("Split temporale della serie")
    ax.set_xlabel("Anno")
    ax.set_ylabel("GWh")
    ax.legend(loc="best")

    fig.tight_layout()
    return fig
