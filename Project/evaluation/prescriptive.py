"""Scenario-based prescriptive analytics derived from the global winner forecast."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _signal_from_change(change_pct: float, uncertainty_ratio: float) -> str:
    if uncertainty_ratio >= 0.15:
        prefix = "high_uncertainty_"
    elif uncertainty_ratio >= 0.08:
        prefix = "moderate_uncertainty_"
    else:
        prefix = "low_uncertainty_"

    if change_pct >= 5.0:
        return prefix + "increase"
    if change_pct <= -5.0:
        return prefix + "decrease"
    return prefix + "stable"


def build_prescriptive_table(
    family_winners: pd.DataFrame,
    winner_forecasts: pd.DataFrame,
    original_series: pd.Series,
) -> pd.DataFrame:
    """Build simple scenario-based recommendations from the global winner."""

    global_winner = family_winners.iloc[0]
    family = str(global_winner["family"])
    model = str(global_winner["model"])
    uncertainty_margin = float(global_winner.get("rmse_val_orig", np.nan))
    if not np.isfinite(uncertainty_margin):
        uncertainty_margin = float(global_winner.get("rmse_val", np.nan))

    raw = pd.to_numeric(original_series, errors="coerce").dropna().astype(float)
    rows: list[dict[str, Any]] = []
    test_df = winner_forecasts.loc[winner_forecasts["split"] == "test"].copy()

    for row in test_df.itertuples(index=False):
        timestamp = row.timestamp
        predicted_value = float(getattr(row, f"{family}_pred"))
        prev_candidates = raw[raw.index < timestamp]
        previous_observed = float(prev_candidates.iloc[-1]) if not prev_candidates.empty else float("nan")
        expected_change = predicted_value - previous_observed if np.isfinite(previous_observed) else float("nan")
        change_pct = (expected_change / previous_observed * 100.0) if np.isfinite(previous_observed) and abs(previous_observed) > 1e-9 else float("nan")
        uncertainty_ratio = abs(uncertainty_margin / predicted_value) if np.isfinite(predicted_value) and abs(predicted_value) > 1e-9 else float("nan")
        rows.append(
            {
                "family": family,
                "model": model,
                "timestamp": timestamp,
                "predicted_value": predicted_value,
                "previous_observed": previous_observed,
                "expected_change": expected_change,
                "expected_change_pct": change_pct,
                "uncertainty_margin": uncertainty_margin,
                "scenario_low": predicted_value - uncertainty_margin if np.isfinite(uncertainty_margin) else float("nan"),
                "scenario_high": predicted_value + uncertainty_margin if np.isfinite(uncertainty_margin) else float("nan"),
                "uncertainty_ratio": uncertainty_ratio,
                "recommendation": _signal_from_change(change_pct, uncertainty_ratio if np.isfinite(uncertainty_ratio) else 0.0),
            }
        )

    return pd.DataFrame(rows)