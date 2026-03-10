"""
Progetto Operational Analytics - Caso Ogrin (serie giornaliera)

Obiettivo didattico:
- replicare la pipeline predittiva su una seconda serie (dominio diverso),
  come richiesto dalla consegna del corso;
- confrontare modelli statistici, neurali e tree-based sullo stesso protocollo.

Collegamento alle slide:
- Slide 0-1: processo di predictive analytics operativo.
- Slide 2: preprocessing e diagnostica della serie.
- Slide 3: modello statistico ARMA (via auto_arima senza stagionalità).
- Slide 4: MLP su finestre temporali.
- Slide 5: XGBoost per regressione non lineare.
- Slide 6-7: confronto metriche e test statistici (ADF).
- Slide 8: selezione iperparametri.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
import torch
import random
import utils
import models.xgbRegressor as xgb 
import models.mlp as mlp 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def seed_everything(seed):
    """
    Rende riproducibile l'intera pipeline.
    Nel forecasting comparativo è importante ridurre la variabilità casuale,
    soprattutto nel training neurale.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

SEED = 1234
seed_everything(SEED)

# == Time Series analysis ==
# Lettura serie storica e impostazione frequenza giornaliera.
# Slide 1: base del task predittivo (serie temporale indicizzata nel tempo).
ts = pd.read_csv("./dataset/ogrin.csv", parse_dates=['date'], index_col='date').asfreq('D')
ts['values'] = ts['values'].astype('float32')

# Split temporale per training/tuning/testing.
# Slide 6-7: sperimentazione con holdout cronologico.
train, validation, test = ts[:106], ts[106:136], ts[136:] # 70/20/10
train_validation = ts[:136]

# Controllo visivo di regime, trend e volatilità tra i segmenti.
plt.plot(train, label='Train')
plt.plot(validation, label='Validation')
plt.plot(test, label='Test')
plt.legend()
plt.title('Train, Validation and Test')
plt.show()

# Checking stationarity con ADF.
# Serie più vicine alla stazionarietà sono compatibili con ARMA (d=0).
# Slide 3 e 7.
result = adfuller(train)
print(f'ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4g}')

# Ispezione autocorrelazioni (struttura temporale utile ai modelli AR/MA).
utils.plot_acf_pacf(train,title="Train")

# Detrending diagnostico via differenza prima.
# Non è necessariamente usato nel fit finale, ma serve a capire la dinamica.
# Slide 2: preprocessing/diagnostica.
train_diff = train.diff().dropna()
result_diff = adfuller(train_diff)
print(f'ADF Statistic: {result_diff[0]:.4f}, p-value: {result_diff[1]:.4g}')
utils.plot_acf_pacf(train_diff, title="Train Differenced (1st Order)")  # m = 12

# == ARMA ==
# auto_arima con seasonal=False e stationary=True:
# ricerca di un ARMA(p,q) adatto alla serie senza componente stagionale esplicita.
# Tecnologia: pmdarima.
auto_sarima = pm.auto_arima(train, stationary=True,
                            test='adf', max_p=5, max_q=5,
                            seasonal=False, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

auto_sarima_fit = auto_sarima.fit(train)
auto_sarima_fit.plot_diagnostics(figsize=(12,8))
plt.show()

# Validation forecast (prima stima out-of-sample).
sarima_validation_predictions = auto_sarima_fit.predict(n_periods=len(validation))

# Refit finale e forecast su test.
sarima_final = auto_sarima.fit(train_validation)
sarima_test_predictions = sarima_final.predict(n_periods=len(test))

print("--- ARMA ---")
print("Validation:")
_, _, _ = utils.evaluate_metrics(validation, sarima_validation_predictions)
print("Test:")
_, _, _ = utils.evaluate_metrics(test, sarima_test_predictions)

utils.plot_predictions(ts, sarima_validation_predictions, sarima_test_predictions, title="ARMA predictions")

# == MLP ==
# Modello neurale feed-forward su lag temporali (supervised windowing).
# Implementazione nel modulo models/mlp.py.
# Slide 4 (neural models) + Slide 8 (tuning).
''' Grid Search parameters used
param_grid = {
    'look_back': [7,14,21,28,30],
    'hidden_size': [2, 4, 6, 8],
    'lr': [0.001, 0.0001, 0.01],
    'activation': ['relu', 'tanh'],
    'dropout': [0.0, 0.1],
    'batch_size': [8,16,32]
}
'''
EPOCHS = 500
param_grid = {
    'look_back': [28],
    'hidden_size': [6],
    'lr': [0.001],
    'activation': ['tanh'],
    'dropout': [0.0],
    'batch_size': [32]
}

mlp_grid_results = mlp.train_and_evaluate_grid(train.values, validation.values, param_grid, EPOCHS, seed=SEED, verbose=False)
mlp_validation_predictions = mlp_grid_results.iloc[0]['preds']
mlp_best_params = mlp_grid_results.iloc[0]

mlp_test_predictions = mlp.train_final_and_evaluate(train_validation, test, mlp_best_params, EPOCHS, seed=SEED)

print("--- MLP ---")
print("Validation:")
_, _, _ = utils.evaluate_metrics(validation, mlp_validation_predictions)
print("Test:")
_, _, _ = utils.evaluate_metrics(test, mlp_test_predictions)

mlp_validation_predictions = pd.Series(mlp_validation_predictions, index=validation.index)
mlp_test_predictions = pd.Series(mlp_test_predictions, index=test.index)
utils.plot_predictions(ts, mlp_validation_predictions, mlp_test_predictions, title="MLP predictions")

#  == XGBoost ==
# Regressione tree-based con boosting graduale.
# Converte la serie in problema supervisionato tramite look_back.
# Slide 5 (machine learning models) + Slide 8 (grid search).
''' Grid Search parameters used
param_grid = {
    'look_back': [7,14,21,28,30],
    "n_estimators": [5,10,15,20,30,50,80,100],
    "max_depth": [1, 3, 5, 7, 9],
    "eta" : [0.1,0.2,0.4,0.6,0.8,1],
    "subsample": [0.1, 0.3, 0.5, 0.7, 0.9, 1],
    "colsample_bytree": [0.1 ,0.3, 0.5, 0.7, 0.9, 1],
    'seed': [1234],
}
'''

param_grid = {
    'look_back': [28],
    "n_estimators": [80],
    "max_depth": [5],
    "eta" : [1],
    "subsample": [0.7],
    "colsample_bytree": [1],
    'seed': [1234],
}

xgb_grid_model, xgb_validation_predictions, xgb_best_params, _ = xgb.grid_search_xgb(train.squeeze(), validation.squeeze(), param_grid)

look_back = xgb_best_params['look_back']
x_train, y_train = xgb.create_dataset(train_validation.squeeze(), look_back)
final_model = xgb.train_xgb_model(x_train, y_train, xgb_best_params)
# Previsione multi-step con rolling forecast ricorsivo.
xgb_test_predictions = xgb.rolling_forecast(final_model, x_train, len(test), look_back)

print("--- XGBoost ---")
print("Validation:")
_, _, _ = utils.evaluate_metrics(validation, xgb_validation_predictions)
print("Test:")
_, _, _ = utils.evaluate_metrics(test, xgb_test_predictions)

xgb_validation_predictions = pd.Series(xgb_validation_predictions, index=validation.index)
xgb_test_predictions = pd.Series(xgb_test_predictions, index=test.index)
utils.plot_predictions(ts, xgb_validation_predictions, xgb_test_predictions, title="XGBRegressor predictions")



