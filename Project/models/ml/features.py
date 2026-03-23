"""Feature engineering and feature-selection utilities for Step 4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


@dataclass(frozen=True)
class LaggedDataset:
    """Container for lagged supervised datasets by split."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]


def build_lagged_dataset(
    train: pd.Series,
    validation: pd.Series,
    test: pd.Series,
    lookback: int,
) -> LaggedDataset:
    """Build lagged supervised matrices for train/validation/test splits.

    Validation and test rows are allowed to use lag history from previous splits,
    which is required for realistic temporal forecasting.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    full = pd.concat([train, validation, test])
    full = full[~full.index.duplicated(keep="first")].sort_index()
    values = full.astype(float).to_numpy()
    idx = full.index

    train_idx = set(train.index.tolist())
    val_idx = set(validation.index.tolist())
    test_idx = set(test.index.tolist())

    feat_cols = [f"lag_{i}" for i in range(1, lookback + 1)]
    x_tr, y_tr, i_tr = [], [], []
    x_va, y_va, i_va = [], [], []
    x_te, y_te, i_te = [], [], []

    for t in range(lookback, len(values)):
        target_index = idx[t]
        lag_window = values[t - lookback:t]
        # lag_1 = most recent observation.
        features = lag_window[::-1]

        if target_index in train_idx:
            x_tr.append(features)
            y_tr.append(values[t])
            i_tr.append(target_index)
        elif target_index in val_idx:
            x_va.append(features)
            y_va.append(values[t])
            i_va.append(target_index)
        elif target_index in test_idx:
            x_te.append(features)
            y_te.append(values[t])
            i_te.append(target_index)

    X_train = pd.DataFrame(np.asarray(x_tr), columns=feat_cols, index=i_tr)
    X_val = pd.DataFrame(np.asarray(x_va), columns=feat_cols, index=i_va)
    X_test = pd.DataFrame(np.asarray(x_te), columns=feat_cols, index=i_te)

    y_train = pd.Series(np.asarray(y_tr), index=i_tr, name="y")
    y_val = pd.Series(np.asarray(y_va), index=i_va, name="y")
    y_test = pd.Series(np.asarray(y_te), index=i_te, name="y")

    if X_train.empty or X_val.empty or X_test.empty:
        raise RuntimeError(
            "Lagged dataset is empty for at least one split. "
            "Increase split sizes or reduce lookback."
        )

    return LaggedDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feat_cols,
    )


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str,
    n_select: int,
    random_state: int,
) -> tuple[list[str], pd.DataFrame]:
    """Select informative lag features using RFE or model importance.

    Returns selected feature names and a ranking/importance table.
    """
    method_norm = method.strip().lower()
    n_total = X_train.shape[1]
    n_select = max(1, min(int(n_select), n_total))

    if method_norm == "none":
        selected = list(X_train.columns)
        report = pd.DataFrame(
            {
                "feature": selected,
                "score": np.nan,
                "rank": 1,
                "selected": True,
                "method": "none",
            }
        )
        return selected, report

    if method_norm == "rfe":
        estimator = DecisionTreeRegressor(max_depth=3, random_state=random_state)
        rfe = RFE(estimator=estimator, n_features_to_select=n_select, step=1)
        rfe.fit(X_train, y_train)
        selected = list(X_train.columns[rfe.support_])
        report = pd.DataFrame(
            {
                "feature": list(X_train.columns),
                "score": np.nan,
                "rank": rfe.ranking_,
                "selected": rfe.support_,
                "method": "rfe",
            }
        ).sort_values(["rank", "feature"], ascending=[True, True])
        return selected, report.reset_index(drop=True)

    if method_norm == "importance":
        estimator = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=1,
        )
        estimator.fit(X_train, y_train)
        importances = estimator.feature_importances_
        report = pd.DataFrame(
            {
                "feature": list(X_train.columns),
                "score": importances,
                "rank": pd.Series(importances).rank(ascending=False, method="dense").astype(int),
                "selected": False,
                "method": "importance",
            }
        ).sort_values(["score", "feature"], ascending=[False, True])
        selected = list(report.head(n_select)["feature"])
        report.loc[report["feature"].isin(selected), "selected"] = True
        return selected, report.reset_index(drop=True)

    raise ValueError("feature_selection must be one of: none, rfe, importance")


def build_model_feature_matrix(
    lagged: LaggedDataset,
    selected_features: list[str],
) -> LaggedDataset:
    """Project lagged matrices onto the selected feature subset."""
    cols = [c for c in selected_features if c in lagged.X_train.columns]
    if not cols:
        raise RuntimeError("No selected features are present in lagged dataset")
    return LaggedDataset(
        X_train=lagged.X_train[cols].copy(),
        y_train=lagged.y_train.copy(),
        X_val=lagged.X_val[cols].copy(),
        y_val=lagged.y_val.copy(),
        X_test=lagged.X_test[cols].copy(),
        y_test=lagged.y_test.copy(),
        feature_names=cols,
    )


def last_window_from_series(series: pd.Series, lookback: int) -> np.ndarray:
    """Return the last lookback values as lag vector [lag_1, lag_2, ...]."""
    tail = series.astype(float).to_numpy()[-lookback:]
    if len(tail) < lookback:
        raise ValueError("Series is shorter than lookback")
    return tail[::-1].copy()
