"""End-to-end forecasting pipeline for Operational Analytics project."""

from __future__ import annotations

import json

import pandas as pd

from config import (
    MLP_EPOCHS,
    MLP_PARAM_GRID,
    PLOTS_DIR,
    RESULTS_DIR,
    SERIES_CONFIG,
    SEED,
    XGB_PARAM_GRID,
)
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


def _load_series(cfg: dict) -> pd.Series:
    df = pd.read_csv(cfg["csv_path"], parse_dates=[cfg["date_col"]], index_col=cfg["date_col"])
    s = df[cfg["value_col"]].astype(float).asfreq(cfg["freq"])
    return s.interpolate(method="time")


def _split_series(series: pd.Series, split: tuple[int, int]) -> tuple[pd.Series, pd.Series, pd.Series]:
    cut1, cut2 = split
    train = series.iloc[:cut1]
    val = series.iloc[cut1:cut2]
    test = series.iloc[cut2:]
    return train, val, test


def _run_one_series(name: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    series = _load_series(cfg)
    train, val, test = _split_series(series, cfg["split"])

    adf_report = {
        "series": run_adf(series),
        "train": run_adf(train),
    }

    plot_acf_pacf(train, f"{name} train ACF/PACF", PLOTS_DIR / f"{name}_acf_pacf.png")

    results = []
    params = {}

    stat = forecast_statistical(
        train,
        val,
        test,
        seasonal=cfg["seasonal"],
        seasonal_period=cfg["seasonal_period"],
        diff_order=cfg["diff_order"],
    )
    stat_val_s = to_series(stat["validation_pred"], val.index)
    stat_test_s = to_series(stat["test_pred"], test.index)
    plot_predictions(series, stat_val_s, stat_test_s, f"{name} - {stat['name']}", PLOTS_DIR / f"{name}_{stat['name'].lower()}_forecast.png")

    stat_val_metrics = evaluate_metrics(val, stat_val_s)
    stat_test_metrics = evaluate_metrics(test, stat_test_s)
    results.append(
        {
            "series": name,
            "model": stat["name"],
            "val_mae": stat_val_metrics["mae"],
            "val_rmse": stat_val_metrics["rmse"],
            "val_mape": stat_val_metrics["mape"],
            "test_mae": stat_test_metrics["mae"],
            "test_rmse": stat_test_metrics["rmse"],
            "test_mape": stat_test_metrics["mape"],
        }
    )
    params[stat["name"]] = stat["best_params"]

    mlp = run_mlp(train, val, test, MLP_PARAM_GRID, MLP_EPOCHS, SEED)
    mlp_val_s = to_series(mlp["validation_pred"], val.index)
    mlp_test_s = to_series(mlp["test_pred"], test.index)
    plot_predictions(series, mlp_val_s, mlp_test_s, f"{name} - MLP", PLOTS_DIR / f"{name}_mlp_forecast.png")

    mlp_val_metrics = evaluate_metrics(val, mlp_val_s)
    mlp_test_metrics = evaluate_metrics(test, mlp_test_s)
    results.append(
        {
            "series": name,
            "model": "MLP",
            "val_mae": mlp_val_metrics["mae"],
            "val_rmse": mlp_val_metrics["rmse"],
            "val_mape": mlp_val_metrics["mape"],
            "test_mae": mlp_test_metrics["mae"],
            "test_rmse": mlp_test_metrics["rmse"],
            "test_mape": mlp_test_metrics["mape"],
        }
    )
    params["MLP"] = mlp["best_params"]

    xgb = run_xgb(train, val, test, XGB_PARAM_GRID)
    xgb_val_s = to_series(xgb["validation_pred"], val.index)
    xgb_test_s = to_series(xgb["test_pred"], test.index)
    plot_predictions(series, xgb_val_s, xgb_test_s, f"{name} - XGBoost", PLOTS_DIR / f"{name}_xgboost_forecast.png")

    xgb_val_metrics = evaluate_metrics(val, xgb_val_s)
    xgb_test_metrics = evaluate_metrics(test, xgb_test_s)
    results.append(
        {
            "series": name,
            "model": "XGBoost",
            "val_mae": xgb_val_metrics["mae"],
            "val_rmse": xgb_val_metrics["rmse"],
            "val_mape": xgb_val_metrics["mape"],
            "test_mae": xgb_test_metrics["mae"],
            "test_rmse": xgb_test_metrics["rmse"],
            "test_mape": xgb_test_metrics["mape"],
        }
    )
    params["XGBoost"] = xgb["best_params"]
    params["adf"] = adf_report

    return pd.DataFrame(results), params


def run_pipeline() -> None:
    seed_everything(SEED)
    ensure_directories(RESULTS_DIR, PLOTS_DIR)

    all_metrics = []
    all_params = {}

    for name, cfg in SERIES_CONFIG.items():
        df_metrics, series_params = _run_one_series(name, cfg)
        all_metrics.append(df_metrics)
        all_params[name] = series_params

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df = metrics_df.sort_values(["series", "test_rmse"], ascending=[True, True]).reset_index(drop=True)
    metrics_df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    with open(RESULTS_DIR / "best_models_params.json", "w", encoding="utf-8") as f:
        json.dump(all_params, f, indent=2, default=str)

    print("Pipeline completed.")
    print(f"Saved: {(RESULTS_DIR / 'metrics_summary.csv')}")
    print(f"Saved: {(RESULTS_DIR / 'best_models_params.json')}")


if __name__ == "__main__":
    run_pipeline()
