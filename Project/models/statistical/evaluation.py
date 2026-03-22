"""Evaluation utilities for Step 3 statistical models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from .model_config import _aicc, compute_metrics, validation_original_metrics


def build_residuals_table(
    model_name: str,
    residuals: pd.Series,
    ljung_box_lags: int,
) -> pd.DataFrame:
    """Build Ljung-Box diagnostics table for a residual series."""
    lags = min(ljung_box_lags, max(1, len(residuals) // 3))
    lb = acorr_ljungbox(residuals.dropna(), lags=[lags], return_df=True)
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "residual_mean": float(residuals.mean()),
                "residual_std": float(residuals.std(ddof=1)),
                "ljung_box_lag": int(lags),
                "ljung_box_stat": float(lb["lb_stat"].iloc[0]),
                "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
            }
        ]
    )


def build_summary_table(
    validation: pd.Series,
    test: pd.Series,
    sarima_best: dict[str, Any],
    hw_best: dict[str, Any],
    sarima_val_pred: pd.Series,
    hw_val_pred: pd.Series,
    sarima_test_pred: pd.Series,
    hw_test_pred: pd.Series,
    sarima_final: Any,
    hw_final: Any,
    sarima_orig_context: dict[str, Any] | None,
    hw_orig_context: dict[str, Any] | None,
    diff_order: int,
    train_validation_len: int,
) -> pd.DataFrame:
    """Build the summary table used to compare SARIMA and Holt-Winters."""

    sarima_val_orig_metrics = validation_original_metrics(
        sarima_val_pred, sarima_orig_context, diff_order
    )
    hw_val_orig_metrics = validation_original_metrics(
        hw_val_pred, hw_orig_context, diff_order
    )

    def _m(series: pd.Series, pred: pd.Series) -> dict[str, float]:
        return compute_metrics(series, pred)

    return pd.DataFrame(
        [
            {
                "model": "sarima",
                "best_params": str(
                    {
                        "order": sarima_best["cfg"]["order"],
                        "seasonal_order": sarima_best["cfg"]["seasonal_order"],
                    }
                ),
                "rmse_val": _m(validation, sarima_val_pred)["rmse"],
                "mae_val": _m(validation, sarima_val_pred)["mae"],
                "mape_val": _m(validation, sarima_val_pred)["mape"],
                "mbe_val": _m(validation, sarima_val_pred)["mbe"],
                "abs_mbe_val": _m(validation, sarima_val_pred)["abs_mbe"],
                "rmse_val_orig": np.nan if sarima_val_orig_metrics is None else sarima_val_orig_metrics["rmse"],
                "mae_val_orig": np.nan if sarima_val_orig_metrics is None else sarima_val_orig_metrics["mae"],
                "mape_val_orig": np.nan if sarima_val_orig_metrics is None else sarima_val_orig_metrics["mape"],
                "mbe_val_orig": np.nan if sarima_val_orig_metrics is None else sarima_val_orig_metrics["mbe"],
                "abs_mbe_val_orig": np.nan if sarima_val_orig_metrics is None else sarima_val_orig_metrics["abs_mbe"],
                "rmse_test": _m(test, sarima_test_pred)["rmse"],
                "mae_test": _m(test, sarima_test_pred)["mae"],
                "mape_test": _m(test, sarima_test_pred)["mape"],
                "mbe_test": _m(test, sarima_test_pred)["mbe"],
                "abs_mbe_test": _m(test, sarima_test_pred)["abs_mbe"],
                "aic": float(sarima_final.aic),
                "aicc": _aicc(
                    float(sarima_final.aic),
                    train_validation_len,
                    int(sarima_final.params.shape[0]),
                ),
            },
            {
                "model": "holt_winters",
                "best_params": str(hw_best["cfg"]),
                "rmse_val": _m(validation, hw_val_pred)["rmse"],
                "mae_val": _m(validation, hw_val_pred)["mae"],
                "mape_val": _m(validation, hw_val_pred)["mape"],
                "mbe_val": _m(validation, hw_val_pred)["mbe"],
                "abs_mbe_val": _m(validation, hw_val_pred)["abs_mbe"],
                "rmse_val_orig": np.nan if hw_val_orig_metrics is None else hw_val_orig_metrics["rmse"],
                "mae_val_orig": np.nan if hw_val_orig_metrics is None else hw_val_orig_metrics["mae"],
                "mape_val_orig": np.nan if hw_val_orig_metrics is None else hw_val_orig_metrics["mape"],
                "mbe_val_orig": np.nan if hw_val_orig_metrics is None else hw_val_orig_metrics["mbe"],
                "abs_mbe_val_orig": np.nan if hw_val_orig_metrics is None else hw_val_orig_metrics["abs_mbe"],
                "rmse_test": _m(test, hw_test_pred)["rmse"],
                "mae_test": _m(test, hw_test_pred)["mae"],
                "mape_test": _m(test, hw_test_pred)["mape"],
                "mbe_test": _m(test, hw_test_pred)["mbe"],
                "abs_mbe_test": _m(test, hw_test_pred)["abs_mbe"],
                "aic": float(getattr(hw_final, "aic", np.nan)),
                "aicc": float(getattr(hw_final, "aicc", np.nan)),
            },
        ]
    )


def build_forecast_table(
    validation: pd.Series,
    test: pd.Series,
    sarima_val_pred: pd.Series,
    hw_val_pred: pd.Series,
    sarima_test_pred: pd.Series,
    hw_test_pred: pd.Series,
) -> pd.DataFrame:
    """Build the merged forecast table for validation and test splits."""
    return pd.DataFrame(
        {
            "split": ["validation"] * len(validation) + ["test"] * len(test),
            "timestamp": list(validation.index) + list(test.index),
            "actual": list(validation.values) + list(test.values),
            "sarima_pred": list(np.asarray(sarima_val_pred)) + list(np.asarray(sarima_test_pred)),
            "hw_pred": list(np.asarray(hw_val_pred)) + list(np.asarray(hw_test_pred)),
        }
    )


def select_winner(summary: pd.DataFrame) -> tuple[str, pd.Series]:
    """Select winner model by test RMSE then test MAE."""
    best_row = summary.sort_values(["rmse_test", "mae_test"], ascending=[True, True]).iloc[0]
    return str(best_row["model"]), best_row
