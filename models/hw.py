"""
Modulo Holt-Winters (Exponential Smoothing) con grid search.

Collegamento slide:
- Slide 3: modelli statistici classici per time series;
- Slide 8: parameter fitting su griglia iperparametri.

Tecnologia:
- statsmodels.tsa.holtwinters.ExponentialSmoothing
"""

import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def compute_score(y_true, y_pred, scoring_func=None):
    """
    Calcola la metrica di errore per il forecasting.

    Se viene passata una scoring_func esterna (es. RMSE), usa quella;
    altrimenti usa RMSE come default.
    """
    if scoring_func:
        return scoring_func(y_true, y_pred)
    return root_mean_squared_error(y_true, y_pred)


def holt_winters_grid_search(train, validation, seasonal_periods=12,
                             scoring_func=None, return_best_model=True,
                             verbose=False):
    """
    Prova tutte le combinazioni Holt-Winters e seleziona la migliore su validation.

    Parametri esplorati:
    - trend: componente di trend (None/add/mul)
    - seasonal: componente stagionale (add/mul)
    - damped_trend: trend smorzato o meno
    - use_boxcox: trasformazione Box-Cox interna al modello

    Restituisce:
    - DataFrame risultati (ordinati per RMSE)
    - opzionalmente il best fitted model, il forecast e i best params.
    """
    param_grid = {
        'trend': [None, 'add', 'mul'],
        'seasonal': ['add', 'mul'],
        'damped_trend': [True, False],
        'use_boxcox': [True, False]
    }

    results = []
    best_model = {'score': float('inf'), 'model': None, 'forecast': None, 'params': None}

    for params in ParameterGrid(param_grid):
        try:
            model = ExponentialSmoothing(
                train,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
                **params
            )
            fitted_model = model.fit()
            forecast = fitted_model.forecast(len(validation))
            score = compute_score(validation, forecast, scoring_func)

            results.append({**params, 'rmse': score})

            if score < best_model['score']:
                best_model = {
                    'score': score,
                    'model': fitted_model,
                    'forecast': forecast,
                    'params': {
                        **params,
                        'seasonal_periods': seasonal_periods,
                        'initialization_method': 'estimated'
                    }
                }

        except Exception as e:
            if verbose:
                print(f"Failed with params: {params} -> {e}")
            continue

    results_df = pd.DataFrame(results).sort_values('rmse').reset_index(drop=True)

    if results_df.empty:
        return (results_df, *([None] * 3)) if return_best_model else results_df

    if return_best_model:
        return results_df, best_model['model'], best_model['forecast'], best_model['params']
    
    return results_df


def train_final_and_predict(train_validation, n_forecast, best_params):
    """
    Addestra Holt-Winters su train+validation con i migliori parametri
    e genera la previsione finale sul test horizon.
    """
    model = ExponentialSmoothing(train_validation, **best_params)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(n_forecast)
    return fitted_model, forecast
