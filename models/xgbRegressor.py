"""
Modulo XGBoost per forecasting univariato tramite approccio autoregressivo.

Collegamento slide:
- Slide 5: machine learning non neurale (alberi + boosting).
- Slide 8: selezione iperparametri con grid search.

Tecnologia:
- xgboost.XGBRegressor (objective='reg:squarederror').
"""

from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np

def create_dataset(arrdata, look_back=1):
    """
    Converte una serie in coppie supervisionate (X, y) usando finestre lag.

    È la stessa idea usata per MLP: il tempo viene codificato nei lag.
    """
    dataX, dataY = [], []
    for i in range(len(arrdata) - look_back):
        a = arrdata[i:(i + look_back)]
        dataX.append(a)
        dataY.append(arrdata[i + look_back])
    return np.array(dataX), np.array(dataY)


def rolling_forecast(model, x_train, steps_ahead, look_back):
    """
    Genera forecast multi-step in modo ricorsivo (rolling window).

    Si parte dall'ultima finestra osservata e, ad ogni passo:
    - si predice il prossimo valore,
    - si aggiorna la finestra sostituendo il valore più vecchio.
    """
    xinput = x_train[-1]
    yfore = []
    for _ in range(steps_ahead):
        yfore.append(model.predict(xinput.reshape(1, look_back))[0])
        xinput = np.roll(xinput, -1)
        xinput[-1] = yfore[-1]
    return yfore


def train_xgb_model(x_train, y_train, params):
    """
    Addestra un modello XGBRegressor con i parametri forniti.

    Nota:
    - 'look_back' è un parametro del data shaping, non del modello XGBoost,
      quindi viene rimosso da model_params.
    """
    model_params = {k: v for k, v in params.items() if k != 'look_back'}
    model = XGBRegressor(
        objective='reg:squarederror',
        verbosity=0,
        **model_params
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_train, validation, look_back):
    """Valuta il modello calcolando RMSE sul validation set."""
    forecast = rolling_forecast(model, x_train, len(validation), look_back)
    rmse = root_mean_squared_error(validation.values, forecast)
    return rmse, forecast


def grid_search_xgb(train, validation, param_grid):
    """
    Esegue grid search su combinazioni iperparametriche e seleziona il best model.

    Metriche e selezione:
    - criterio di scelta: RMSE minimo su validation;
    - output: modello migliore, forecast associato, best_params, best_rmse.
    """
    best_rmse = float("inf")
    best_model = None
    best_forecast = None
    best_params = None

    for params in ParameterGrid(param_grid):
        look_back = params['look_back']
        x_train, y_train = create_dataset(train, look_back)
        model = train_xgb_model(x_train, y_train, params)
        rmse, forecast = evaluate_model(model, x_train, validation,look_back)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_forecast = forecast
            best_params = params

    return best_model, best_forecast, best_params, best_rmse
