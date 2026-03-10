"""
Progetto Operational Analytics - Caso Demographic (M3)

Obiettivo didattico di questo script:
- costruire una pipeline di forecasting time series end-to-end;
- confrontare famiglie modellistiche diverse richieste dal corso:
    1) statistici (SARIMA, Holt-Winters),
    2) neurali (MLP),
    3) machine learning tree-based (XGBoost);
- valutare le prestazioni su validation e test.

Collegamento alle slide:
- Slide 0 "Operational analytics": pipeline orientata al supporto decisionale.
- Slide 1 "Predictive analytics": previsione da dati storici.
- Slide 2 "Predictive data preprocessing": Box-Cox, differencing, ADF, ACF/PACF.
- Slide 3 "Predictive statistical models": SARIMA + Exponential Smoothing.
- Slide 4 "Predictive neural models": MLP per regressione su finestre temporali.
- Slide 5 "Predictive machine learning models": XGBoost regressore su lag.
- Slide 6-7 "Statistics": metriche, test di stazionarietà, confronto modelli.
- Slide 8 "Parameter fitting": ricerca iperparametri (grid/auto search).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import base.utils as utils
import torch
import random
import models.mlp as mlp 
import models.xgbRegressor as xgb
import models.hw as hw

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def seed_everything(seed):
        """Imposta i seed per rendere l'esperimento riproducibile."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        mlp.seed_everything(seed)

SEED = 1234
seed_everything(SEED)

# == Time Series analysis ==
# Fase di ingestion e preparazione iniziale.
# Tecnologia: pandas + matplotlib.
# Slide 0-1: costruzione del flusso analitico predittivo.
ts = pd.read_csv("./dataset/demographic.csv", parse_dates=['date'], index_col='date').asfreq('ME')
ts['values'] = ts['values'].astype('float32')

# Split cronologico (non random): evita data leakage nel forecasting.
# Slide 6-7: corretta impostazione sperimentale (train/validation/test).
train, validation, test = ts[:111], ts[111:123], ts[123:] # 80/10/10
train_validation = ts[:123]

# Visualizzazione base della serie e delle finestre di split.
plt.plot(train, label='Train')
plt.plot(validation, label='Validation')
plt.plot(test, label='Test')
plt.legend()
plt.title('Train, Validation and Test')
plt.show()

# Box-Cox transformation
# Scopo: stabilizzare la varianza e rendere la serie più adatta ai modelli lineari.
# Slide 2: preprocessing transformation.
train_boxcox, lmbda = boxcox(train['values'])
train_boxcox = pd.Series(train_boxcox, index=train.index)
train_validation_boxcox, lmbda_tv = boxcox(train_validation['values'])
train_validation_boxcox = pd.Series(train_validation_boxcox, index=train_validation.index)

# Checking stationarity (ADF)
# ADF: H0 = serie non stazionaria. p-value basso -> si rifiuta H0.
# Slide 7: inferential statistics (hypothesis testing).
result = adfuller(train)
print(f'ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4g}')

# ACF/PACF per ispezione dipendenze seriali e stagionalità.
# Slide 3: strumenti diagnostici pre-modello statistico.
utils.plot_acf_pacf(train,title="Train")

# Detrending via first and second differencing
# Slide 2-3: rimozione trend per avvicinarsi alla stazionarietà.
train_diff = train_boxcox.diff().dropna()
result_diff = adfuller(train_diff)
print(f'ADF Statistic: {result_diff[0]:.4f}, p-value: {result_diff[1]:.4g}')
utils.plot_acf_pacf(train_diff, title="Train Differenced (1st Order)")  # m = 12

train_diff2 = train_diff.diff().dropna()
result_diff2 = adfuller(train_diff2)
print(f'ADF Statistic: {result_diff2[0]:.4f}, p-value: {result_diff2[1]:.4g}')
utils.plot_acf_pacf(train_diff2, title="Train Differenced (2nd Order)")

# Deseasonalization via seasonal differencing (lag 12)
# Con dati mensili, m=12 intercetta la stagionalità annuale.
# Slide 3: SARIMA con componente stagionale.
train_diff2_season = train_diff2.diff(12).dropna()
result_diff2_season = adfuller(train_diff2_season)
print(f'ADF Statistic: {result_diff2_season[0]:.4f}, p-value: {result_diff2_season[1]:.4g}')
utils.plot_acf_pacf(train_diff2_season, title="Train Differenced (2nd Order) + Seasonal Differencing (lag 12)")

# == SARIMA ==
# Modello statistico classico per serie con trend+stagionalità.
# Tecnologia: pmdarima.auto_arima (ricerca guidata ordini p,d,q e P,D,Q).
# Slide 3 + Slide 8: statistical models + parameter fitting.
auto_sarima = pm.auto_arima(train_boxcox,test='adf',
                            d=2, D=1,
                            max_p=5, max_q=5,
                            m=12, start_P=0, seasonal=True,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

auto_sarima_fit = auto_sarima.fit(train_boxcox)
auto_sarima.plot_diagnostics(figsize=(12,8))
plt.show()

# Forecast su validation (out-of-sample rispetto al train).
sarima_validation_predictions = auto_sarima_fit.predict(n_periods=len(validation))

# Refit su train+validation e forecast su test finale.
sarima_final = auto_sarima.fit(train_validation_boxcox)
sarima_test_predictions = sarima_final.predict(n_periods=len(test))

# Inversione Box-Cox per tornare alla scala originale interpretabile.
sarima_validation_predictions = inv_boxcox(sarima_validation_predictions, lmbda)
sarima_test_predictions = inv_boxcox(sarima_test_predictions, lmbda_tv)

print("--- SARIMA ---")
print("Validation:")
_, _, _ =utils.evaluate_metrics(validation, sarima_validation_predictions)
print("Test:")
_, _, _ =utils.evaluate_metrics(test, sarima_test_predictions)

utils.plot_predictions(ts, sarima_validation_predictions, sarima_test_predictions, title="SARIMA predictions")

# == Holt-Winter's ==
# Exponential Smoothing stagionale: alternativa statistica a SARIMA.
# Tecnologia: statsmodels ExponentialSmoothing con grid search custom in models/hw.py.
# Slide 3 (metodi classici) + Slide 8 (ricerca iperparametri).
_, _, hw_validation_predictions, hw_best_params = hw.holt_winters_grid_search(
    train, validation, seasonal_periods=12, scoring_func=root_mean_squared_error, verbose=False
)
hw_final, hw_test_predicions = hw.train_final_and_predict(train_validation, n_forecast=len(test), best_params=hw_best_params)

print("--- Holt-Winter's ---")
print("Validation:")
_, _, _ = utils.evaluate_metrics(validation, hw_validation_predictions)
print("Test:")
_, _, _ = utils.evaluate_metrics(test, hw_test_predicions)

utils.plot_predictions(ts, hw_validation_predictions, hw_test_predicions, title="Holt-Winter's predictions")

# == MLP ==
# Modello neurale feed-forward per forecasting autoregressivo su finestra (look_back).
# Tecnologia: PyTorch.
# Slide 4 (neural predictive models) + Slide 2 (scaling) + Slide 8 (hyperparameter tuning).
''' Grid Search parameters used
param_grid = {
    'look_back': [12,18,24,30,36],
    'hidden_size': [2,4,6,8],
    'lr': [1e-2, 1e-3, 1e-4],
    'activation': ['relu', 'tanh'],
    'dropout': [0, 0.1],
    'batch_size': [8,16,32]
}
'''

EPOCHS = 500
param_grid = {
    'look_back': [30],
    'hidden_size': [2],
    'lr': [1e-2],
    'activation': ['relu'],
    'dropout': [0],
    'batch_size': [16]
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
''' Grid Search parameters used
param_grid = {
    'look_back': [12,18,24,30,36],
    "n_estimators": [5,10,15,20,30,50,80,100],
    "max_depth": [1, 3, 5, 7],
    "eta" : [0.1,0.2,0.4,0.6,0.8,1],
    "subsample": [0.1,0.3,0.7,1],
    "colsample_bytree": [0.1,0.3,0.7,1],
    'seed': [1234],
}
'''
param_grid = {
    'look_back': [24],
    "n_estimators": [15],
    "max_depth": [3],
    "eta": [0.8],
    "subsample": [0.1],
    "colsample_bytree": [0.3],
    'seed': [1234],
}

xgb_grid_model, xgb_validation_predictions, xgb_best_params, _ = xgb.grid_search_xgb(train.squeeze(), validation.squeeze(), param_grid)

look_back = xgb_best_params['look_back']
x_train, y_train = xgb.create_dataset(train_validation.squeeze(), look_back)
final_model = xgb.train_xgb_model(x_train, y_train, xgb_best_params)
# Rolling recursive forecast su test.
# Slide 5: regressore ML usato in setup temporale multi-step.
xgb_test_predictions = xgb.rolling_forecast(final_model, x_train, len(test), look_back)

print("--- XGBoost ---")
print("Validation:")
_, _, _ = utils.evaluate_metrics(validation, xgb_validation_predictions)
print("Test:")
_, _, _ = utils.evaluate_metrics(test, xgb_test_predictions)

xgb_validation_predictions = pd.Series(xgb_validation_predictions, index=validation.index)
xgb_test_predictions = pd.Series(xgb_test_predictions, index=test.index)
utils.plot_predictions(ts, xgb_validation_predictions, xgb_test_predictions, title="XGBRegressor predictions")
