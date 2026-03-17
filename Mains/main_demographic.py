"""End-to-end forecasting pipeline for the demographic series.

This module is fully self-contained: every modelling choice, hyperparameter
grid, and preprocessing step for the demographic series lives here.
Edit this file to customise the analysis without affecting other series.

Entry point:
    run_pipeline_demographic()

Produced artifacts:
    - results/metrics_summary_demographic.csv
    - results/best_models_params_demographic.json
    - results/plots/demographic_*.png
"""

from __future__ import annotations

import json

import pandas as pd

from config import (
    MLP_EPOCHS,
    MLP_PARAM_GRID,
    PLOTS_DIR,
    RESULTS_DIR,
    SEED,
    SERIES_CONFIG,
    XGB_PARAM_GRID,
)
from Mains.utils import _load_series, _split_series
from Models.neural import tune_and_forecast as run_mlp
from Models.statistical import forecast_statistical
from Models.tree_based import tune_and_forecast as run_xgb
from utils import (
    ensure_directories,
    evaluate_metrics,
    plot_acf_pacf,
    plot_predictions,
    run_adf,
    seed_everything,
    to_series,
)

# ── Series-specific settings ──────────────────────────────────────────────────
# Modify the constants below to customise the demographic analysis without
# touching any other series pipeline.

_SERIES_NAME = "demographic"
_CFG = SERIES_CONFIG[_SERIES_NAME]

# To use a different grid for demographic only, replace the right-hand side.
_MLP_PARAM_GRID = MLP_PARAM_GRID
_XGB_PARAM_GRID = XGB_PARAM_GRID
_MLP_EPOCHS = MLP_EPOCHS


def _run_demographic_pipeline() -> tuple[pd.DataFrame, dict]:
    """Execute the full forecasting workflow for the demographic series.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Metrics DataFrame (one row per model) and best-parameter report dict.

    Workflow
    --------
    1) Load and split the series.
    2) ADF stationarity diagnostics on full series and train.
    3) ACF/PACF diagnostic plot on train.
    4) SARIMA: auto_arima with Box-Cox + seasonal settings for monthly data.
    5) MLP: grid search on val RMSE, recursive forecast, inverse scaling.
    6) XGBoost: grid search on val RMSE, rolling forecast.
    7) Collect metrics (MAE, RMSE, MAPE) and best parameters.
    """
    series = _load_series(_CFG)
    train, val, test = _split_series(series, _CFG["split"])

    adf_report = {
        "series": run_adf(series),
        "train": run_adf(train),
    }

    plot_acf_pacf(
        train,
        f"{_SERIES_NAME} train ACF/PACF",
        PLOTS_DIR / f"{_SERIES_NAME}_acf_pacf.png",
    )

    results: list[dict] = []
    params: dict = {}

    # ── Statistical (SARIMA) ──────────────────────────────────────────────────
    stat = forecast_statistical(
        train, val, test,
        seasonal=_CFG["seasonal"],
        seasonal_period=_CFG["seasonal_period"],
        diff_order=_CFG["diff_order"],
    )
    stat_val_s = to_series(stat["validation_pred"], val.index)
    stat_test_s = to_series(stat["test_pred"], test.index)
    plot_predictions(
        series, stat_val_s, stat_test_s,
        f"{_SERIES_NAME} - {stat['name']}",
        PLOTS_DIR / f"{_SERIES_NAME}_{stat['name'].lower()}_forecast.png",
    )
    stat_val_m = evaluate_metrics(val, stat_val_s)
    stat_test_m = evaluate_metrics(test, stat_test_s)
    results.append({
        "series": _SERIES_NAME, "model": stat["name"],
        "val_mae": stat_val_m["mae"], "val_rmse": stat_val_m["rmse"], "val_mape": stat_val_m["mape"],
        "test_mae": stat_test_m["mae"], "test_rmse": stat_test_m["rmse"], "test_mape": stat_test_m["mape"],
    })
    params[stat["name"]] = stat["best_params"]

    # ── MLP ───────────────────────────────────────────────────────────────────
    mlp = run_mlp(train, val, test, _MLP_PARAM_GRID, _MLP_EPOCHS, SEED)
    mlp_val_s = to_series(mlp["validation_pred"], val.index)
    mlp_test_s = to_series(mlp["test_pred"], test.index)
    plot_predictions(
        series, mlp_val_s, mlp_test_s,
        f"{_SERIES_NAME} - MLP",
        PLOTS_DIR / f"{_SERIES_NAME}_mlp_forecast.png",
    )
    mlp_val_m = evaluate_metrics(val, mlp_val_s)
    mlp_test_m = evaluate_metrics(test, mlp_test_s)
    results.append({
        "series": _SERIES_NAME, "model": "MLP",
        "val_mae": mlp_val_m["mae"], "val_rmse": mlp_val_m["rmse"], "val_mape": mlp_val_m["mape"],
        "test_mae": mlp_test_m["mae"], "test_rmse": mlp_test_m["rmse"], "test_mape": mlp_test_m["mape"],
    })
    params["MLP"] = mlp["best_params"]

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb = run_xgb(train, val, test, _XGB_PARAM_GRID)
    xgb_val_s = to_series(xgb["validation_pred"], val.index)
    xgb_test_s = to_series(xgb["test_pred"], test.index)
    plot_predictions(
        series, xgb_val_s, xgb_test_s,
        f"{_SERIES_NAME} - XGBoost",
        PLOTS_DIR / f"{_SERIES_NAME}_xgboost_forecast.png",
    )
    xgb_val_m = evaluate_metrics(val, xgb_val_s)
    xgb_test_m = evaluate_metrics(test, xgb_test_s)
    results.append({
        "series": _SERIES_NAME, "model": "XGBoost",
        "val_mae": xgb_val_m["mae"], "val_rmse": xgb_val_m["rmse"], "val_mape": xgb_val_m["mape"],
        "test_mae": xgb_test_m["mae"], "test_rmse": xgb_test_m["rmse"], "test_mape": xgb_test_m["mape"],
    })
    params["XGBoost"] = xgb["best_params"]
    params["adf"] = adf_report

    return pd.DataFrame(results), params


def run_pipeline_demographic() -> tuple[pd.DataFrame, dict]:
    """Run and persist the demographic forecasting pipeline.

    Side effects
    ------------
    - Initializes deterministic seeds.
    - Ensures output directories exist.
    - Saves per-series metrics and best parameters.

    Output files
    ------------
    - results/metrics_summary_demographic.csv
    - results/best_models_params_demographic.json

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Metrics DataFrame and parameter report keyed by series name.
    """
    seed_everything(SEED)
    ensure_directories(RESULTS_DIR, PLOTS_DIR)

    metrics_df, params = _run_demographic_pipeline()
    metrics_df = metrics_df.sort_values(["series", "test_rmse"]).reset_index(drop=True)
    metrics_df.to_csv(RESULTS_DIR / "metrics_summary_demographic.csv", index=False)

    report = {_SERIES_NAME: params}
    with open(RESULTS_DIR / "best_models_params_demographic.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Saved: {RESULTS_DIR / 'metrics_summary_demographic.csv'}")
    print(f"Saved: {RESULTS_DIR / 'best_models_params_demographic.json'}")

    return metrics_df, report


if __name__ == "__main__":
    run_pipeline_demographic()
