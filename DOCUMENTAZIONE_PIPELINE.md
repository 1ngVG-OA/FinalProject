# Documentazione esplicativa del forecasting pipeline

## 1) Obiettivo del progetto

Questo progetto esegue una pipeline end-to-end di forecasting su due serie temporali:
- `demographic`
- `ogrin`

Per ogni serie vengono confrontate tre famiglie di modelli:
- Modello statistico (`ARMA` o `SARIMA` via `auto_arima`)
- Modello neurale (`MLP` autoregressivo)
- Modello tree-based (`XGBoost` autoregressivo)

La pipeline produce:
- metriche di validazione e test per ogni modello
- parametri migliori trovati in fase di tuning
- grafici diagnostici e grafici di forecast


## 2) Punto di ingresso

Il punto di ingresso principale e:
- `Mains/main.py`

La funzione avviata da script e:
- `run_pipeline()`

Quando esegui:

```bash
python Mains/main.py
```

viene lanciato tutto il flusso per tutte le serie definite in configurazione.


## 3) Configurazione centrale (`config.py`)

In `config.py` trovi la configurazione globale:

- `DATA_DIR`, `RESULTS_DIR`, `PLOTS_DIR`: cartelle dati/output
- `SEED`: seed globale per riproducibilita
- `SERIES_CONFIG`: dizionario con parametri per ciascuna serie
- `MLP_PARAM_GRID`: griglia iperparametri MLP
- `XGB_PARAM_GRID`: griglia iperparametri XGBoost
- `MLP_EPOCHS`: numero massimo epoche MLP

### 3.1) Significato dei campi di `SERIES_CONFIG`

Per ogni serie sono presenti:
- `csv_path`: path al CSV
- `date_col`: nome colonna data
- `value_col`: nome colonna valori
- `freq`: frequenza forzata (`ME`, `D`, ...)
- `split`: tuple `(cut1, cut2)` per train/val/test
- `seasonal`: se usare stagionalita nel modello statistico
- `seasonal_period`: periodo stagionale (es. 12 mensile)
- `diff_order`: ordine differenziazione impostato in `auto_arima`


## 4) Flusso generale (`run_pipeline`)

La funzione `run_pipeline()` fa questi passi:

1. `seed_everything(SEED)`
2. `ensure_directories(RESULTS_DIR, PLOTS_DIR)`
3. ciclo su ogni serie in `SERIES_CONFIG`
4. per ogni serie chiama `_run_one_series(name, cfg)`
5. concatena tutte le metriche in un unico DataFrame
6. ordina per `series` e `test_rmse`
7. salva:
   - `results/metrics_summary.csv`
   - `results/best_models_params.json`
8. stampa i path dei file salvati


## 5) Dettaglio per singola serie (`_run_one_series`)

Per ogni serie temporale la pipeline esegue:

1. caricamento e preprocessing (`_load_series`)
2. split in train/val/test (`_split_series`)
3. test ADF su serie completa e train
4. plot ACF/PACF su train
5. training + forecast modello statistico
6. training + forecast MLP con tuning
7. training + forecast XGBoost con tuning
8. valutazione metrica su validation e test per ogni modello
9. raccolta parametri migliori e report ADF

Output della funzione:
- `DataFrame` con 3 righe (una per modello) e metriche
- `dict` con parametri migliori + report ADF


## 6) Caricamento e preprocess (`_load_series`)

Codice logico:

1. legge CSV con parse date e indice temporale
2. estrae colonna valori e converte a float
3. forza la frequenza con `.asfreq(cfg["freq"])`
4. interpola eventuali buchi con `interpolate(method="time")`

Perche e importante:
- i modelli richiedono una griglia temporale regolare
- l interpolazione evita NaN durante training e forecasting


## 7) Split dati (`_split_series`)

Dato `split = (cut1, cut2)`:
- train = `series[:cut1]`
- val = `series[cut1:cut2]`
- test = `series[cut2:]`

Nota:
- la validazione e usata per selezione iperparametri
- il test e usato solo per valutazione finale (dopo refit su train+val)


## 8) Modello statistico (`Models/statistical.py`)

Funzione: `forecast_statistical(train, val, test, seasonal, seasonal_period, diff_order)`

Logica:

1. Se `seasonal=True`, applica Box-Cox al train
2. Esegue `pm.auto_arima(...)` con:
   - test `adf`
   - differenziazione `d=diff_order`
   - stagionalita secondo config (`D=1`, `m=seasonal_period`)
3. Predice validation partendo dal modello fit su train
4. Refit su `train+val`
5. Predice test
6. Se Box-Cox attivo, applica inversa (`inv_boxcox`) alle predizioni

Output:
- `name`: `SARIMA` o `ARMA`
- `best_params`: ordine e settaggi
- `validation_pred`, `test_pred`


## 9) Modello neurale MLP (`Models/neural.py`)

Funzione principale: `tune_and_forecast(...)`

Pipeline MLP:

1. seed deterministico (`seed_everything`)
2. standardizzazione train (`StandardScaler`)
3. grid search su `MLP_PARAM_GRID`
4. per ogni combinazione:
   - crea dataset autoregressivo con `look_back`
   - allena MLP con Adam + MSE + early stopping (patience 10)
   - forecast ricorsivo su validation
   - calcola RMSE validation
5. sceglie i parametri con RMSE validation minima
6. refit su `train+val`
7. forecast ricorsivo su test
8. inverse scaling delle predizioni

Dettagli architettura MLP:
- `Linear(input -> hidden)`
- attivazione (`relu` o `tanh`)
- `Dropout`
- `Linear(hidden -> 1)`


## 10) Modello XGBoost (`Models/tree_based.py`)

Funzione principale: `tune_and_forecast(...)`

Pipeline XGBoost:

1. grid search su `XGB_PARAM_GRID`
2. per ogni combinazione:
   - crea dataset autoregressivo con `look_back`
   - allena `XGBRegressor`
   - esegue rolling forecast su validation
   - misura RMSE validation
3. sceglie combinazione migliore
4. refit su `train+val`
5. rolling forecast su test

Nota:
- rolling forecast = ogni nuova predizione viene inserita nella finestra come input del passo successivo


## 11) Metriche (`utils.evaluate_metrics`)

Per validation e test vengono calcolate:
- `MAE`
- `RMSE`
- `MAPE` (in percentuale)

Formula sintetica:

- MAE: media di `|y - y_hat|`
- RMSE: `sqrt(media((y - y_hat)^2))`
- MAPE: `media(|(y - y_hat)/y|) * 100`

Attenzione:
- se la serie contiene zeri o valori molto piccoli, la MAPE puo diventare instabile.


## 12) Plot generati

Per ogni serie:

1. diagnostica ACF/PACF su train:
   - `results/plots/<serie>_acf_pacf.png`

2. forecast per modello:
   - `results/plots/<serie>_<modello>_forecast.png`
   - esempi: `demographic_mlp_forecast.png`, `ogrin_xgboost_forecast.png`

I grafici includono:
- serie completa
- forecast su validation
- forecast su test


## 13) File di output finali

1. `results/metrics_summary.csv`
   - una riga per (serie, modello)
   - colonne:
     - `series`, `model`
     - `val_mae`, `val_rmse`, `val_mape`
     - `test_mae`, `test_rmse`, `test_mape`

2. `results/best_models_params.json`
   - parametri migliori per ciascun modello e serie
   - include anche report ADF (`adf_stat`, `pvalue`)


## 14) Struttura logica delle dipendenze

- `Mains/main.py`
  - usa `config.py`
  - usa `utils.py`
  - usa `Models/statistical.py`
  - usa `Models/neural.py`
  - usa `Models/tree_based.py`

Dipendenza dati:
- i CSV in `Datasets/` sono sorgente dei time series


## 15) Esecuzione consigliata

Da root progetto:

```bash
python -m venv .venv
# Attiva l ambiente virtuale (Windows)
.venv\Scripts\activate
pip install -r requirements.txt
python Mains/main.py
```

Al termine controlla:
- `results/metrics_summary.csv`
- `results/best_models_params.json`
- `results/plots/`


## 16) Checklist di validazione rapida

Se vuoi verificare che tutto abbia funzionato:

1. Esiste `results/metrics_summary.csv`
2. Ci sono 6 righe nel CSV (2 serie x 3 modelli)
3. Esiste `results/best_models_params.json`
4. Esistono plot ACF/PACF e 3 forecast per ogni serie
5. Nessun valore NaN nelle colonne metriche


## 17) Limiti attuali e possibili miglioramenti

Limiti:
- split train/val/test fisso (single split)
- confronto basato su una sola finestra temporale
- MAPE sensibile a valori vicini a zero

Miglioramenti possibili:
- backtesting con rolling origin
- intervalli di confidenza delle previsioni
- test statistici di confronto tra modelli
- logging strutturato invece di sole print
- tracciamento esperimenti (es. parametri + metriche per ogni trial)


## 18) In sintesi

La pipeline e gia ben organizzata in moduli separati:
- orchestrazione in `Mains/main.py`
- configurazione in `config.py`
- utility condivise in `utils.py`
- modelli in `Models/`

Il flusso segue una pratica corretta per forecasting supervisionato:
- tuning su validation
- refit su train+validation
- valutazione finale su test
- salvataggio completo di metriche, parametri e grafici
