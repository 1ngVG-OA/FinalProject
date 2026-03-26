"""Pairwise Diebold-Mariano tests for forecast comparison."""

from __future__ import annotations

from itertools import combinations
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano_test(
    actual: pd.Series,
    pred_a: pd.Series,
    pred_b: pd.Series,
    *,
    horizon: int = 1,
    power: int = 2,
) -> dict[str, float | int]:
    """Compute the Harvey-Leybourne-Newbold corrected Diebold-Mariano statistic."""

    aligned = pd.concat(
        [
            pd.to_numeric(actual, errors="coerce").rename("actual"),
            pd.to_numeric(pred_a, errors="coerce").rename("pred_a"),
            pd.to_numeric(pred_b, errors="coerce").rename("pred_b"),
        ],
        axis=1,
    ).dropna()

    n_obs = int(len(aligned))
    if n_obs < max(5, horizon + 2):
        return {
            "n_obs": n_obs,
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "mean_loss_diff": float("nan"),
        }

    err_a = aligned["actual"] - aligned["pred_a"]
    err_b = aligned["actual"] - aligned["pred_b"]
    loss_a = np.abs(err_a) ** power
    loss_b = np.abs(err_b) ** power
    d = (loss_a - loss_b).to_numpy(dtype=float)
    mean_d = float(np.mean(d))

    gamma0 = float(np.var(d, ddof=1))
    long_run_var = gamma0
    for lag in range(1, horizon):
        cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
        long_run_var += 2.0 * float(cov)

    if not np.isfinite(long_run_var) or long_run_var <= 0.0:
        return {
            "n_obs": n_obs,
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "mean_loss_diff": mean_d,
        }

    dm_stat = mean_d / sqrt(long_run_var / n_obs)
    correction = sqrt((n_obs + 1 - 2 * horizon + (horizon * (horizon - 1)) / n_obs) / n_obs)
    dm_hln = dm_stat * correction
    p_value = 2.0 * float(stats.t.sf(abs(dm_hln), df=n_obs - 1))

    return {
        "n_obs": n_obs,
        "dm_stat": float(dm_hln),
        "p_value": p_value,
        "mean_loss_diff": mean_d,
    }


def build_diebold_mariano_table(
    winner_forecasts: pd.DataFrame,
    families: Iterable[str] = ("statistical", "ml", "neural"),
) -> pd.DataFrame:
    """Build pairwise Diebold-Mariano results for family winners on the test set."""

    test_df = winner_forecasts.loc[winner_forecasts["split"] == "test"].copy()
    results: list[dict[str, object]] = []

    for family_a, family_b in combinations(tuple(families), 2):
        dm = diebold_mariano_test(
            actual=test_df["actual"],
            pred_a=test_df[f"{family_a}_pred"],
            pred_b=test_df[f"{family_b}_pred"],
            horizon=1,
            power=2,
        )
        p_value = dm["p_value"]
        if np.isnan(p_value):
            interpretation = "insufficient_data"
        elif p_value < 0.05:
            if dm["mean_loss_diff"] < 0:
                interpretation = f"{family_a}_better"
            else:
                interpretation = f"{family_b}_better"
        else:
            interpretation = "no_significant_difference"

        results.append(
            {
                "family_a": family_a,
                "family_b": family_b,
                "loss_function": "squared_error",
                "horizon": 1,
                "n_obs": dm["n_obs"],
                "mean_loss_diff": dm["mean_loss_diff"],
                "dm_stat": dm["dm_stat"],
                "p_value": dm["p_value"],
                "interpretation": interpretation,
            }
        )

    return pd.DataFrame(results)