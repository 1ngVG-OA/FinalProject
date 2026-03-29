# File di configurazione globale per il progetto di forecasting. 
# Contiene costanti, percorsi e configurazioni utilizzati in tutto il codice.
from pathlib import Path

# Directory paths per datasets, results, and plots.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datasets"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Seed per la riproducibilità dei risultati, utilizzato in modelli ML e XGBoost.
SEED = 1234

# Blocco di configurazione per serie temporali utilizzato da `Mains/utils.py`.
#
# Significato dei campi:
# - csv_path: percorso del dataset di origine
# - date_col: colonna datetime da analizzare e utilizzare come indice
# - value_col: colonna numerica target
# - freq: frequenza temporale prevista (regolarizzazione)
# - split: tuple (cut1, cut2) per suddividere train/validation/test
# - seasonal: se abilitare le impostazioni del modello statistico stagionale
# - seasonal_period: periodo di stagionalità (ad esempio 12 per mensile)
# - diff_order: ordine di differenziazione utilizzato da auto_arima
SERIES_CONFIG = {
    # Serie definite da ISTAT 
    "Energy production and consumed": {
        "csv_path": DATA_DIR / "Tavola_1.14.csv",
        "date_col": "date",
        "value_col": "values",
        "freq": "ME",
        "split": (111, 123),
        "seasonal": True,
        "seasonal_period": 12,
        "diff_order": 2,
    }
}

# Ricerca degli iperparametri per il modello MLP.
MLP_PARAM_GRID = {
    "look_back": [14, 28],
    "hidden_size": [4, 8],
    "lr": [1e-3, 1e-2],
    "activation": ["relu", "tanh"],
    "dropout": [0.0, 0.1],
    "batch_size": [16, 32],
}

# Ricerca degli iperparametri per il modello autoregressivo XGBoost.
XGB_PARAM_GRID = {
    "look_back": [14, 28],
    "n_estimators": [30, 80],
    "max_depth": [2, 4],
    "eta": [0.1, 0.3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "seed": [SEED],
}

# Numero massimo di epoche per i cicli di fitting dell'MLP (con early stopping interno).
MLP_EPOCHS = 300
