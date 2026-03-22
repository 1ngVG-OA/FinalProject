"""Statistical model selection utilities for Step 3.

Implements:
- SARIMA grid search on train -> validation ranking.
- Holt-Winters benchmark grid search on the same split.
- Refit of best configs on train+validation and test evaluation.
- Residual diagnostics (ACF/PACF + Ljung-Box) and comparison plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE ignoring near-zero denominators."""

    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    ape = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(ape) * 100.0)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Return RMSE/MAE/MAPE on aligned series."""

    pred = pd.Series(np.asarray(y_pred, dtype=float), index=y_true.index)
    yt = y_true.astype(float).to_numpy()
    yp = pred.astype(float).to_numpy()

    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "mape": _safe_mape(yt, yp),
    }


def _aicc(aic: float, n: int, k: int) -> float:
    """Small-sample corrected AIC."""

    if (n - k - 1) <= 0:
        return float("nan")
    return float(aic + (2.0 * k * (k + 1)) / (n - k - 1))


@dataclass(frozen=True)
class StatisticalStepConfig:
    """Configuration for statistical model search."""

    p_values: tuple[int, ...] = (0, 1, 2)
    d_values: tuple[int, ...] = (2,)
    q_values: tuple[int, ...] = (0, 1, 2)
    p_seasonal_values: tuple[int, ...] = (0, 1)
    d_seasonal_values: tuple[int, ...] = (0,)
    q_seasonal_values: tuple[int, ...] = (0, 1)
    seasonal_period: int = 1
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    maxiter: int = 200
    ljung_box_lags: int = 10


class StatisticalModelRunner:
    """Run Step 3 SARIMA and HW benchmark following a shared protocol."""

    def __init__(
        self,
        train: pd.Series,
        validation: pd.Series,
        test: pd.Series,
        config: StatisticalStepConfig | None = None,
    ) -> None:
        self.config = config or StatisticalStepConfig()

        self.train = self._validate_split(train, "train")
        self.validation = self._validate_split(validation, "validation")
        self.test = self._validate_split(test, "test")
        self.train_validation = pd.concat([self.train, self.validation])

    @staticmethod
    def _validate_split(series: pd.Series, name: str) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series")
        s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        if len(s) < 8:
            raise ValueError(f"{name} split is too short")
        if not s.index.is_monotonic_increasing:
            s = s.sort_index()
        s.name = "value"
        return s

    def _sarima_candidates(self) -> list[dict[str, Any]]:
        seasonal_period = max(1, int(self.config.seasonal_period))
        candidates: list[dict[str, Any]] = []

        for p, d, q in product(self.config.p_values, self.config.d_values, self.config.q_values):
            if seasonal_period > 1:
                for ps, ds, qs in product(
                    self.config.p_seasonal_values,
                    self.config.d_seasonal_values,
                    self.config.q_seasonal_values,
                ):
                    candidates.append(
                        {
                            "order": (p, d, q),
                            "seasonal_order": (ps, ds, qs, seasonal_period),
                        }
                    )
            else:
                candidates.append(
                    {
                        "order": (p, d, q),
                        "seasonal_order": (0, 0, 0, 0),
                    }
                )

        return candidates

    def fit_sarima_grid(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run SARIMA grid search evaluated on validation split."""

        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for cfg in self._sarima_candidates():
            try:
                model = SARIMAX(
                    self.train,
                    order=cfg["order"],
                    seasonal_order=cfg["seasonal_order"],
                    enforce_stationarity=self.config.enforce_stationarity,
                    enforce_invertibility=self.config.enforce_invertibility,
                )
                fit = model.fit(disp=False, maxiter=self.config.maxiter)
                pred_val = fit.forecast(steps=len(self.validation))

                metrics = compute_metrics(self.validation, pred_val)
                k = int(fit.params.shape[0])
                aic = float(fit.aic)

                row = {
                    "order": str(cfg["order"]),
                    "seasonal_order": str(cfg["seasonal_order"]),
                    "rmse_val": metrics["rmse"],
                    "mae_val": metrics["mae"],
                    "mape_val": metrics["mape"],
                    "aic": aic,
                    "aicc": _aicc(aic, n=len(self.train), k=k),
                    "k_params": k,
                }
                rows.append(row)

                if best is None:
                    best = {"fit": fit, "cfg": cfg, "row": row}
                else:
                    rmse_now = row["rmse_val"]
                    rmse_best = best["row"]["rmse_val"]
                    if rmse_now < rmse_best - 1e-12:
                        best = {"fit": fit, "cfg": cfg, "row": row}
                    elif abs(rmse_now - rmse_best) <= 1e-12:
                        if row["aicc"] < best["row"]["aicc"]:
                            best = {"fit": fit, "cfg": cfg, "row": row}
            except Exception:
                continue

        if not rows or best is None:
            raise RuntimeError("SARIMA grid search failed for all candidate configurations")

        results = pd.DataFrame(rows).sort_values(["rmse_val", "aicc"], ascending=[True, True]).reset_index(drop=True)
        return results, best

    def fit_hw_grid(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run Holt-Winters benchmark search on validation split."""

        seasonal_period = max(1, int(self.config.seasonal_period))
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        if seasonal_period > 1:
            candidate_cfgs = [
                {"trend": "add", "seasonal": "add", "damped_trend": False},
                {"trend": "add", "seasonal": "add", "damped_trend": True},
                {"trend": "add", "seasonal": "mul", "damped_trend": False},
                {"trend": "add", "seasonal": "mul", "damped_trend": True},
                {"trend": None, "seasonal": "add", "damped_trend": False},
                {"trend": None, "seasonal": "mul", "damped_trend": False},
            ]
        else:
            # No seasonal component for annual-like frequency.
            candidate_cfgs = [
                {"trend": "add", "seasonal": None, "damped_trend": False},
                {"trend": "add", "seasonal": None, "damped_trend": True},
                {"trend": None, "seasonal": None, "damped_trend": False},
            ]

        for cfg in candidate_cfgs:
            try:
                model = ExponentialSmoothing(
                    self.train,
                    trend=cfg["trend"],
                    seasonal=cfg["seasonal"],
                    seasonal_periods=seasonal_period if cfg["seasonal"] is not None else None,
                    damped_trend=cfg["damped_trend"],
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                pred_val = fit.forecast(len(self.validation))
                metrics = compute_metrics(self.validation, pred_val)

                row = {
                    "trend": str(cfg["trend"]),
                    "seasonal": str(cfg["seasonal"]),
                    "damped_trend": bool(cfg["damped_trend"]),
                    "rmse_val": metrics["rmse"],
                    "mae_val": metrics["mae"],
                    "mape_val": metrics["mape"],
                    "aic": float(getattr(fit, "aic", np.nan)),
                    "aicc": float(getattr(fit, "aicc", np.nan)),
                }
                rows.append(row)

                if best is None:
                    best = {"fit": fit, "cfg": cfg, "row": row}
                else:
                    rmse_now = row["rmse_val"]
                    rmse_best = best["row"]["rmse_val"]
                    if rmse_now < rmse_best - 1e-12:
                        best = {"fit": fit, "cfg": cfg, "row": row}
                    elif abs(rmse_now - rmse_best) <= 1e-12:
                        if row["aicc"] < best["row"]["aicc"]:
                            best = {"fit": fit, "cfg": cfg, "row": row}
            except Exception:
                continue

        if not rows or best is None:
            raise RuntimeError("Holt-Winters grid search failed for all candidate configurations")

        results = pd.DataFrame(rows).sort_values(["rmse_val", "aicc"], ascending=[True, True]).reset_index(drop=True)
        return results, best

    def _refit_best_sarima(self, cfg: dict[str, Any]) -> Any:
        model = SARIMAX(
            self.train_validation,
            order=cfg["order"],
            seasonal_order=cfg["seasonal_order"],
            enforce_stationarity=self.config.enforce_stationarity,
            enforce_invertibility=self.config.enforce_invertibility,
        )
        return model.fit(disp=False, maxiter=self.config.maxiter)

    def _refit_best_hw(self, cfg: dict[str, Any]) -> Any:
        seasonal_period = max(1, int(self.config.seasonal_period))
        model = ExponentialSmoothing(
            self.train_validation,
            trend=cfg["trend"],
            seasonal=cfg["seasonal"],
            seasonal_periods=seasonal_period if cfg["seasonal"] is not None else None,
            damped_trend=cfg["damped_trend"],
            initialization_method="estimated",
        )
        return model.fit(optimized=True)

    def _build_residuals_table(self, model_name: str, residuals: pd.Series) -> pd.DataFrame:
        lags = min(self.config.ljung_box_lags, max(1, len(residuals) // 3))
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

    def run(self) -> dict[str, Any]:
        """Run complete Step 3 workflow and return all artifacts."""

        sarima_grid_df, sarima_best = self.fit_sarima_grid()
        hw_grid_df, hw_best = self.fit_hw_grid()

        sarima_val_pred = sarima_best["fit"].forecast(len(self.validation))
        hw_val_pred = hw_best["fit"].forecast(len(self.validation))

        sarima_final = self._refit_best_sarima(sarima_best["cfg"])
        hw_final = self._refit_best_hw(hw_best["cfg"])

        sarima_test_pred = sarima_final.forecast(len(self.test))
        hw_test_pred = hw_final.forecast(len(self.test))

        summary = pd.DataFrame(
            [
                {
                    "model": "sarima",
                    "best_params": str({
                        "order": sarima_best["cfg"]["order"],
                        "seasonal_order": sarima_best["cfg"]["seasonal_order"],
                    }),
                    "rmse_val": compute_metrics(self.validation, sarima_val_pred)["rmse"],
                    "mae_val": compute_metrics(self.validation, sarima_val_pred)["mae"],
                    "mape_val": compute_metrics(self.validation, sarima_val_pred)["mape"],
                    "rmse_test": compute_metrics(self.test, sarima_test_pred)["rmse"],
                    "mae_test": compute_metrics(self.test, sarima_test_pred)["mae"],
                    "mape_test": compute_metrics(self.test, sarima_test_pred)["mape"],
                    "aic": float(sarima_final.aic),
                    "aicc": _aicc(float(sarima_final.aic), len(self.train_validation), int(sarima_final.params.shape[0])),
                },
                {
                    "model": "holt_winters",
                    "best_params": str(hw_best["cfg"]),
                    "rmse_val": compute_metrics(self.validation, hw_val_pred)["rmse"],
                    "mae_val": compute_metrics(self.validation, hw_val_pred)["mae"],
                    "mape_val": compute_metrics(self.validation, hw_val_pred)["mape"],
                    "rmse_test": compute_metrics(self.test, hw_test_pred)["rmse"],
                    "mae_test": compute_metrics(self.test, hw_test_pred)["mae"],
                    "mape_test": compute_metrics(self.test, hw_test_pred)["mape"],
                    "aic": float(getattr(hw_final, "aic", np.nan)),
                    "aicc": float(getattr(hw_final, "aicc", np.nan)),
                },
            ]
        )

        best_row = summary.sort_values(["rmse_test", "mae_test"], ascending=[True, True]).iloc[0]
        winner = str(best_row["model"])

        sarima_resid = (self.validation - pd.Series(np.asarray(sarima_val_pred), index=self.validation.index)).dropna()
        hw_resid = (self.validation - pd.Series(np.asarray(hw_val_pred), index=self.validation.index)).dropna()

        residual_diagnostics = pd.concat(
            [
                self._build_residuals_table("sarima", sarima_resid),
                self._build_residuals_table("holt_winters", hw_resid),
            ],
            ignore_index=True,
        )

        forecast_table = pd.DataFrame(
            {
                "split": ["validation"] * len(self.validation) + ["test"] * len(self.test),
                "timestamp": list(self.validation.index) + list(self.test.index),
                "actual": list(self.validation.values) + list(self.test.values),
                "sarima_pred": list(np.asarray(sarima_val_pred)) + list(np.asarray(sarima_test_pred)),
                "hw_pred": list(np.asarray(hw_val_pred)) + list(np.asarray(hw_test_pred)),
            }
        )

        return {
            "sarima_grid": sarima_grid_df,
            "hw_grid": hw_grid_df,
            "summary": summary,
            "winner": winner,
            "winner_params": dict(best_row),
            "residual_diagnostics": residual_diagnostics,
            "forecast_table": forecast_table,
            "validation_actual": self.validation,
            "test_actual": self.test,
            "sarima_val_pred": pd.Series(np.asarray(sarima_val_pred), index=self.validation.index),
            "hw_val_pred": pd.Series(np.asarray(hw_val_pred), index=self.validation.index),
            "sarima_test_pred": pd.Series(np.asarray(sarima_test_pred), index=self.test.index),
            "hw_test_pred": pd.Series(np.asarray(hw_test_pred), index=self.test.index),
            "sarima_validation_residuals": sarima_resid,
            "hw_validation_residuals": hw_resid,
        }

    @staticmethod
    def save_plots(output: dict[str, Any], out_dir: Path) -> dict[str, Path]:
        """Save comparison and residual diagnostic plots for Step 3."""

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "stat_plot_forecasts": out_dir / "tavola_1_14_stat_forecast_comparison_v1.png",
            "stat_plot_residuals": out_dir / "tavola_1_14_stat_residuals_diagnostics_v1.png",
        }

        val_idx = output["validation_actual"].index
        test_idx = output["test_actual"].index

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(val_idx, output["validation_actual"].values, color="black", linewidth=2, label="actual_val")
        ax.plot(test_idx, output["test_actual"].values, color="dimgray", linewidth=2, label="actual_test")
        ax.plot(val_idx, output["sarima_val_pred"].values, color="tab:blue", linestyle="--", label="sarima_val")
        ax.plot(test_idx, output["sarima_test_pred"].values, color="tab:blue", label="sarima_test")
        ax.plot(val_idx, output["hw_val_pred"].values, color="tab:orange", linestyle="--", label="hw_val")
        ax.plot(test_idx, output["hw_test_pred"].values, color="tab:orange", label="hw_test")
        ax.axvline(val_idx.max(), color="gray", linestyle=":", linewidth=1)
        ax.set_title("Step 3 - SARIMA vs Holt-Winters Forecasts")
        ax.set_xlabel("Time")
        ax.set_ylabel("Transformed value")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(paths["stat_plot_forecasts"], dpi=150)
        plt.close(fig)

        def _acf_lags(residuals: pd.Series) -> int:
            return max(1, min(20, len(residuals) - 1))

        def _pacf_lags(residuals: pd.Series) -> int:
            # statsmodels requires nlags < nobs/2 for PACF.
            return max(1, min(20, (len(residuals) // 2) - 1))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plot_acf(output["sarima_validation_residuals"], lags=_acf_lags(output["sarima_validation_residuals"]), ax=axes[0, 0])
        axes[0, 0].set_title("SARIMA Residuals ACF")
        plot_pacf(
            output["sarima_validation_residuals"],
            lags=_pacf_lags(output["sarima_validation_residuals"]),
            ax=axes[0, 1],
            method="ywm",
        )
        axes[0, 1].set_title("SARIMA Residuals PACF")

        plot_acf(output["hw_validation_residuals"], lags=_acf_lags(output["hw_validation_residuals"]), ax=axes[1, 0])
        axes[1, 0].set_title("HW Residuals ACF")
        plot_pacf(
            output["hw_validation_residuals"],
            lags=_pacf_lags(output["hw_validation_residuals"]),
            ax=axes[1, 1],
            method="ywm",
        )
        axes[1, 1].set_title("HW Residuals PACF")

        fig.tight_layout()
        fig.savefig(paths["stat_plot_residuals"], dpi=150)
        plt.close(fig)

        return paths


def infer_seasonal_period_from_index(index: pd.Index) -> int:
    """Infer seasonal period from datetime index frequency.

    Returns conservative defaults when frequency is unknown:
    - monthly -> 12
    - quarterly -> 4
    - weekly -> 52
    - daily -> 7
    - annual/unknown -> 1 (no seasonal component)
    """

    if not isinstance(index, pd.DatetimeIndex):
        return 1

    freq = pd.infer_freq(index)
    if not freq:
        return 1

    freq = freq.upper()
    if freq.startswith("M"):
        return 12
    if freq.startswith("Q"):
        return 4
    if freq.startswith("W"):
        return 52
    if freq.startswith("D"):
        return 7
    if freq.startswith("A") or freq.startswith("Y"):
        return 1
    return 1
