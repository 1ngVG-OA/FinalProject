"""Shared low-level helpers reused by series-specific pipeline modules.

Only stateless utilities for data loading and splitting live here.
Series-specific pipelines are in main_demographic.py and main_ogrin.py.
"""

from __future__ import annotations

import pandas as pd


def _load_series(cfg: dict) -> pd.Series:
    """Load and preprocess one series according to its configuration.

    Parameters
    ----------
    cfg : dict
        Series-specific configuration containing:
        - csv_path: path to source CSV file
        - date_col: datetime column used as index
        - value_col: numeric target column
        - freq: target frequency for regularization (for example 'ME' or 'D')

    Returns
    -------
    pd.Series
        Time-indexed numeric series with regular frequency and missing values
        interpolated using time-based interpolation.

    Notes
    -----
    Frequency regularization is required so downstream forecasting models work on
    a consistent temporal grid.
    """
    df = pd.read_csv(cfg["csv_path"], parse_dates=[cfg["date_col"]], index_col=cfg["date_col"])
    s = df[cfg["value_col"]].astype(float).asfreq(cfg["freq"])
    return s.interpolate(method="time")


def _split_series(series: pd.Series, split: tuple[int, int]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split a full series into train, validation and test partitions.

    Parameters
    ----------
    series : pd.Series
        Input full time series.
    split : tuple[int, int]
        Two cut indices `(cut1, cut2)` used as:
        - train: [0, cut1)
        - validation: [cut1, cut2)
        - test: [cut2, end)

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        `(train, val, test)` partitions.
    """
    cut1, cut2 = split
    train = series.iloc[:cut1]
    val = series.iloc[cut1:cut2]
    test = series.iloc[cut2:]
    return train, val, test
