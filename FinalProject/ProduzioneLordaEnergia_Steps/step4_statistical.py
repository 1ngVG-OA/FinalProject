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


def aicc_from_fit(fitted_model) -> float:
    """Compute finite-sample corrected AIC from a statsmodels ARIMA fit."""
    n_obs = int(getattr(fitted_model, "nobs", 0) or 0)
    n_params = int(len(getattr(fitted_model, "params", [])))
    aic = float(fitted_model.aic)

    if n_obs - n_params - 1 <= 0:
        return np.inf

    return aic + (2 * n_params * (n_params + 1)) / (n_obs - n_params - 1)


def build_order_grid(
    p_values: range | list[int],
    d_value: int,
    q_values: range | list[int],
    max_order_sum: int | None = None,
) -> list[tuple[int, int, int]]:
    """Build a compact ARIMA order grid with an optional complexity cap."""
    orders: list[tuple[int, int, int]] = []

    for p in p_values:
        for q in q_values:
            if max_order_sum is not None and p + d_value + q > max_order_sum:
                continue
            orders.append((int(p), int(d_value), int(q)))

    return orders


def _fit_family_candidate(
    train_series: pd.Series,
    val_series: pd.Series,
    family: str,
    order: tuple[int, int, int],
    trend: str,
) -> dict | None:
    """Fit one ARIMA candidate and return comparable diagnostics/metrics. Returns None if fit fails."""
    try:
        if family == "log_d0":
            fitted = ARIMA(np.log(train_series), order=order, trend=trend).fit()
            val_pred = np.exp(fitted.forecast(steps=len(val_series)))
        else:
            fitted = ARIMA(train_series, order=order, trend=trend).fit()
            val_pred = fitted.forecast(steps=len(val_series))

        val_pred.index = val_series.index
        metrics = regression_metrics(val_series.values, val_pred.values)
        return {
            "family": family,
            "order": order,
            "trend": trend,
            "aic": float(fitted.aic),
            "aicc": float(aicc_from_fit(fitted)),
            "bic": float(fitted.bic),
            "val_mae": metrics["mae"],
            "val_rmse": metrics["rmse"],
            "val_mape": metrics["mape"],
        }
    except Exception:
        return None


def compare_logd0_vs_leveld2(
    train_series: pd.Series,
    val_series: pd.Series,
    log_orders: list[tuple[int, int, int]] | None = None,
    d2_orders: list[tuple[int, int, int]] | None = None,
    sort_by: list[str] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Compare two coherent families: log(d=0) vs level(d=2).
    
    Returns a tuple of (results_dataframe, metadata_dict) where metadata includes
    convergence statistics.
    """
    if log_orders is None:
        log_orders = [(0, 0, 1), (1, 0, 0), (1, 0, 1), (2, 0, 1)]
    if d2_orders is None:
        d2_orders = [(0, 2, 1), (1, 2, 0), (1, 2, 1), (2, 2, 1)]
    if sort_by is None:
        sort_by = ["aicc", "val_rmse", "val_mae"]

    rows: list[dict] = []
    failed_count = 0
    successful_count = 0

    for order in log_orders:
        result = _fit_family_candidate(train_series, val_series, "log_d0", order, trend="c")
        if result is not None:
            rows.append(result)
            successful_count += 1
        else:
            failed_count += 1
            if verbose:
                print(f"  [skip] log_d0 {order}: fit failed")

    for order in d2_orders:
        result = _fit_family_candidate(train_series, val_series, "level_d2", order, trend="n")
        if result is not None:
            rows.append(result)
            successful_count += 1
        else:
            failed_count += 1
            if verbose:
                print(f"  [skip] level_d2 {order}: fit failed")

    if not rows:
        raise ValueError("No ARIMA candidate was successfully fitted.")

    results_df = pd.DataFrame(rows).sort_values(sort_by).reset_index(drop=True)
    metadata = {
        "successful_fits": successful_count,
        "failed_fits": failed_count,
        "total_candidates": successful_count + failed_count,
    }

    return results_df, metadata


def compare_logd0_vs_leveld2_grid(
    train_series: pd.Series,
    val_series: pd.Series,
    p_values: range | list[int] = range(0, 4),
    q_values: range | list[int] = range(0, 4),
    max_order_sum: int | None = 5,
    sort_by: list[str] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run an automated ARIMA grid search for the coherent log(d=0) and level(d=2) families.
    
    Returns results_dataframe, metadata dict.
    """
    log_orders = build_order_grid(p_values=p_values, d_value=0, q_values=q_values, max_order_sum=max_order_sum)
    d2_orders = build_order_grid(p_values=p_values, d_value=2, q_values=q_values, max_order_sum=max_order_sum)
    return compare_logd0_vs_leveld2(
        train_series=train_series,
        val_series=val_series,
        log_orders=log_orders,
        d2_orders=d2_orders,
        sort_by=sort_by,
        verbose=verbose,
    )


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
    trend = best_row.get("trend", "c" if best_family == "log_d0" else "n")

    if best_family == "log_d0":
        model_train = ARIMA(np.log(train_series), order=best_order, trend=trend).fit()
        val_pred = np.exp(model_train.forecast(steps=len(val_series)))

        train_val = pd.concat([train_series, val_series])
        model_refit = ARIMA(np.log(train_val), order=best_order, trend=trend).fit()
        test_pred = np.exp(model_refit.forecast(steps=len(test_series)))
    else:
        model_train = ARIMA(train_series, order=best_order, trend=trend).fit()
        val_pred = model_train.forecast(steps=len(val_series))

        train_val = pd.concat([train_series, val_series])
        model_refit = ARIMA(train_val, order=best_order, trend=trend).fit()
        test_pred = model_refit.forecast(steps=len(test_series))

    val_pred.index = val_series.index
    test_pred.index = test_series.index

    return {
        "best_family": best_family,
        "best_order": best_order,
        "best_aicc": float(best_row.get("aicc", np.nan)),
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

    ax.set_title("Step 4 - grid search ARIMA (log_d0 vs level_d2)")
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
