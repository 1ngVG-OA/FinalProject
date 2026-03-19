from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA


def regression_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict:
    """Compute MAE, RMSE, and MAPE in original scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err / y_true)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def compare_logd0_vs_leveld2(
    train_series: pd.Series,
    val_series: pd.Series,
    log_orders: list[tuple[int, int, int]] | None = None,
    d2_orders: list[tuple[int, int, int]] | None = None,
) -> pd.DataFrame:
    """Compare two coherent families: log(d=0) vs level(d=2)."""
    if log_orders is None:
        log_orders = [(0, 0, 1), (1, 0, 0), (1, 0, 1), (2, 0, 1)]
    if d2_orders is None:
        d2_orders = [(0, 2, 1), (1, 2, 0), (1, 2, 1), (2, 2, 1)]

    rows: list[dict] = []

    for order in log_orders:
        fitted = ARIMA(np.log(train_series), order=order, trend="c").fit()
        val_pred_log = fitted.forecast(steps=len(val_series))
        val_pred = np.exp(val_pred_log)
        m = regression_metrics(val_series.values, val_pred.values)
        rows.append(
            {
                "family": "log_d0",
                "order": order,
                "aic": fitted.aic,
                "val_mae": m["mae"],
                "val_rmse": m["rmse"],
                "val_mape": m["mape"],
            }
        )

    for order in d2_orders:
        fitted = ARIMA(train_series, order=order, trend="n").fit()
        val_pred = fitted.forecast(steps=len(val_series))
        m = regression_metrics(val_series.values, val_pred.values)
        rows.append(
            {
                "family": "level_d2",
                "order": order,
                "aic": fitted.aic,
                "val_mae": m["mae"],
                "val_rmse": m["rmse"],
                "val_mape": m["mape"],
            }
        )

    return pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)


def select_and_forecast(
    train_series: pd.Series,
    val_series: pd.Series,
    test_series: pd.Series,
    results_df: pd.DataFrame,
) -> dict:
    """Select best model from comparison table, forecast val/test, and score."""
    best_row = results_df.iloc[0]
    best_family = best_row["family"]
    best_order = tuple(best_row["order"])

    if best_family == "log_d0":
        model_train = ARIMA(np.log(train_series), order=best_order, trend="c").fit()
        val_pred = np.exp(model_train.forecast(steps=len(val_series)))

        train_val = pd.concat([train_series, val_series])
        model_refit = ARIMA(np.log(train_val), order=best_order, trend="c").fit()
        test_pred = np.exp(model_refit.forecast(steps=len(test_series)))
    else:
        model_train = ARIMA(train_series, order=best_order, trend="n").fit()
        val_pred = model_train.forecast(steps=len(val_series))

        train_val = pd.concat([train_series, val_series])
        model_refit = ARIMA(train_val, order=best_order, trend="n").fit()
        test_pred = model_refit.forecast(steps=len(test_series))

    val_pred.index = val_series.index
    test_pred.index = test_series.index

    return {
        "best_family": best_family,
        "best_order": best_order,
        "model_train": model_train,
        "model_refit": model_refit,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_metrics": regression_metrics(val_series.values, val_pred.values),
        "test_metrics": regression_metrics(test_series.values, test_pred.values),
        "residuals": model_refit.resid,
    }


def plot_forecast(
    train_series: pd.Series,
    val_series: pd.Series,
    test_series: pd.Series,
    val_pred: pd.Series,
    test_pred: pd.Series,
    best_family: str,
    best_order: tuple[int, int, int],
) -> plt.Figure:
    """Plot train/validation/test plus selected forecast traces."""
    fig, ax = plt.subplots(figsize=(14, 5))

    train_series.plot(ax=ax, label="Train", color="tab:blue", lw=2)
    val_series.plot(ax=ax, label="Validation", color="tab:orange", lw=2)
    test_series.plot(ax=ax, label="Test", color="tab:green", lw=2)

    val_pred.plot(ax=ax, label=f"Forecast val {best_family} {best_order}", color="tab:red", ls="--")
    test_pred.plot(ax=ax, label=f"Forecast test {best_family} {best_order}", color="tab:purple", ls="--")

    ax.set_title("Step 4 - confronto finale (log_d0 vs level_d2)")
    ax.set_xlabel("Anno")
    ax.set_ylabel("GWh")
    ax.legend(loc="best")

    fig.tight_layout()
    return fig


def plot_residual_diagnostics(residuals: pd.Series) -> tuple[plt.Figure, plt.Figure]:
    """Return residual diagnostic figures (time/hist/QQ and ACF)."""
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
    residuals.plot(ax=axes[0], title="Residui nel tempo", color="tab:gray")
    axes[0].axhline(0, color="black", lw=1, ls="--")

    residuals.plot(kind="hist", bins=20, ax=axes[1], title="Distribuzione residui", color="tab:cyan")

    qqplot(residuals.dropna(), line="s", ax=axes[2])
    axes[2].set_title("Q-Q plot residui")
    fig1.tight_layout()

    fig2, ax = plt.subplots(figsize=(7, 3))
    plot_acf(residuals.dropna(), lags=20, ax=ax)
    ax.set_title("ACF residui")
    fig2.tight_layout()

    return fig1, fig2
