"""Holt-Winters grid search runner for Step 3."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .model_config import (
    StatisticalStepConfig,
    build_hw_candidate_configs,
    build_original_scale_context,
    compute_metrics,
    validate_original_series,
    validate_split,
    validation_original_metrics,
)
from .sarima import _update_best


class HoltWintersRunner:
    """Holt-Winters benchmark grid search and refit on a fixed split."""

    def __init__(
        self,
        train: pd.Series,
        validation: pd.Series,
        test: pd.Series,
        config: StatisticalStepConfig | None = None,
        original_series: pd.Series | None = None,
        use_log1p: bool = False,
        diff_order: int = 0,
    ) -> None:
        self.config = config or StatisticalStepConfig()
        self.train = validate_split(train, "train")
        self.validation = validate_split(validation, "validation")
        self.test = validate_split(test, "test")
        self.train_validation = pd.concat([self.train, self.validation])
        self.original_series = validate_original_series(original_series)
        self.use_log1p = bool(use_log1p)
        self.diff_order = int(diff_order)
        self._orig_context = build_original_scale_context(
            self.original_series,
            self.use_log1p,
            self.diff_order,
            self.validation.index.min(),
        )

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    def fit_hw_grid(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run the Holt-Winters benchmark search evaluated on the validation split.

        Returns
        -------
        results : pd.DataFrame
            All candidate rows, sorted by (rank_rmse_val, rank_abs_mbe_val, aicc).
        best : dict
            Dict with keys ``fit``, ``cfg``, ``row``, ``rank_rmse``, ``rank_abs_mbe``
            for the selected best candidate.
        """
        seasonal_period = max(1, int(self.config.seasonal_period))
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for cfg in build_hw_candidate_configs(seasonal_period):
            try:
                model = ExponentialSmoothing(
                    self.train,
                    trend=cfg["trend"],
                    seasonal=cfg["seasonal"],
                    seasonal_periods=(
                        seasonal_period if cfg["seasonal"] is not None else None
                    ),
                    damped_trend=cfg["damped_trend"],
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                pred_val = pd.Series(
                    np.asarray(fit.forecast(len(self.validation))),
                    index=self.validation.index,
                )

                metrics = compute_metrics(self.validation, pred_val)
                metrics_orig = validation_original_metrics(
                    pred_val, self._orig_context, self.diff_order
                )

                row = {
                    "trend": str(cfg["trend"]),
                    "seasonal": str(cfg["seasonal"]),
                    "damped_trend": bool(cfg["damped_trend"]),
                    "rmse_val": metrics["rmse"],
                    "mae_val": metrics["mae"],
                    "mape_val": metrics["mape"],
                    "mbe_val": metrics["mbe"],
                    "abs_mbe_val": metrics["abs_mbe"],
                    "rmse_val_orig": np.nan if metrics_orig is None else metrics_orig["rmse"],
                    "mae_val_orig": np.nan if metrics_orig is None else metrics_orig["mae"],
                    "mape_val_orig": np.nan if metrics_orig is None else metrics_orig["mape"],
                    "mbe_val_orig": np.nan if metrics_orig is None else metrics_orig["mbe"],
                    "abs_mbe_val_orig": np.nan if metrics_orig is None else metrics_orig["abs_mbe"],
                    "aic": float(getattr(fit, "aic", np.nan)),
                    "aicc": float(getattr(fit, "aicc", np.nan)),
                }
                rows.append(row)

                rank_rmse_now = (
                    row["rmse_val_orig"]
                    if not pd.isna(row["rmse_val_orig"])
                    else row["rmse_val"]
                )
                rank_abs_mbe_now = (
                    row["abs_mbe_val_orig"]
                    if not pd.isna(row["abs_mbe_val_orig"])
                    else row["abs_mbe_val"]
                )

                best = _update_best(
                    best, fit, cfg, row, rank_rmse_now, rank_abs_mbe_now
                )

            except Exception:
                continue

        if not rows or best is None:
            raise RuntimeError(
                "Holt-Winters grid search failed for all candidate configurations"
            )

        results = pd.DataFrame(rows)
        results = results.assign(
            rank_rmse_val=results["rmse_val_orig"].fillna(results["rmse_val"])
        )
        results = results.assign(
            rank_abs_mbe_val=results["abs_mbe_val_orig"].fillna(
                results["abs_mbe_val"]
            )
        )
        results = results.sort_values(
            ["rank_rmse_val", "rank_abs_mbe_val", "aicc"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        return results, best

    # ------------------------------------------------------------------
    # Refit on train+validation
    # ------------------------------------------------------------------

    def refit(self, cfg: dict[str, Any]) -> Any:
        """Refit the given HW config on the train+validation set."""
        seasonal_period = max(1, int(self.config.seasonal_period))
        model = ExponentialSmoothing(
            self.train_validation,
            trend=cfg["trend"],
            seasonal=cfg["seasonal"],
            seasonal_periods=(
                seasonal_period if cfg["seasonal"] is not None else None
            ),
            damped_trend=cfg["damped_trend"],
            initialization_method="estimated",
        )
        return model.fit(optimized=True)
