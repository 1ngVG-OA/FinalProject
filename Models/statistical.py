"""Statistical forecasting backend (ARMA/SARIMA via pmdarima auto_arima).

This module provides a single function that:
1) fits an ARMA/SARIMA-style model on train,
2) validates on the validation horizon,
3) refits on train+validation,
4) predicts the test horizon,
5) returns forecasts and selected model metadata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pmdarima as pm
from scipy.special import inv_boxcox
from scipy.stats import boxcox


def forecast_statistical(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
    seasonal: bool,
    seasonal_period: int,
    diff_order: int,
) -> dict:
    """Train and evaluate a statistical univariate forecasting model.

    Parameters
    ----------
    train : pd.Series
        Training segment of the time series.
    val : pd.Series
        Validation segment used for the first out-of-sample forecast.
    test : pd.Series
        Test segment used for final out-of-sample evaluation.
    seasonal : bool
        If True, configures a seasonal model and enables Box-Cox transform.
    seasonal_period : int
        Seasonal periodicity `m` passed to auto_arima.
    diff_order : int
        Fixed differencing order `d` passed to auto_arima.

    Returns
    -------
    dict
        Dictionary with:
        - name: model label (`SARIMA` or `ARMA`)
        - best_params: selected model parameters/metadata
        - validation_pred: forecast values for validation horizon
        - test_pred: forecast values for test horizon

    Notes
    -----
    For seasonal series, Box-Cox is fitted on train and train+val separately,
    then predictions are back-transformed with inverse Box-Cox.
    """
    use_boxcox = seasonal

    if use_boxcox:
        train_bc, lmbda = boxcox(train.values)
        train_bc = pd.Series(train_bc, index=train.index)
        train_val = pd.concat([train, val])
        train_val_bc, lmbda_tv = boxcox(train_val.values)
        train_val_bc = pd.Series(train_val_bc, index=train_val.index)
        fit_series = train_bc
    else:
        train_val = pd.concat([train, val])
        lmbda = None
        lmbda_tv = None
        fit_series = train

    model = pm.auto_arima(
        fit_series,
        test="adf",
        d=diff_order,
        D=1 if seasonal else 0,
        seasonal=seasonal,
        m=seasonal_period,
        max_p=5,
        max_q=5,
        start_P=0,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    fitted = model.fit(fit_series)
    val_pred = fitted.predict(n_periods=len(val))

    # Refit on train+val.
    if use_boxcox:
        final_model = model.fit(train_val_bc)
        test_pred = final_model.predict(n_periods=len(test))
        val_pred = inv_boxcox(val_pred, lmbda)
        test_pred = inv_boxcox(test_pred, lmbda_tv)
    else:
        final_model = model.fit(train_val)
        test_pred = final_model.predict(n_periods=len(test))

    params = {
        "order": getattr(final_model, "order", None),
        "seasonal_order": getattr(final_model, "seasonal_order", None),
        "seasonal": seasonal,
        "m": seasonal_period,
        "d": diff_order,
        "boxcox": use_boxcox,
    }

    return {
        "name": "SARIMA" if seasonal else "ARMA",
        "best_params": params,
        "validation_pred": np.asarray(val_pred, dtype=float),
        "test_pred": np.asarray(test_pred, dtype=float),
    }
