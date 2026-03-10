# Demographic — Code Walkthrough vs Slides (Operational Analytics)

Questo report collega *ogni parte del codice* del progetto **demographic** con i concetti teorici delle slide (ordine: introduzione → preprocessing → modelli statistici → modelli neurali → modelli ML → statistica/valutazione → parameter fitting).

## Mappa file (cosa guardare)
- Script principale: `demographic.py`
- Utility comuni (plot + metriche): `utils.py`
- Modelli:
  - Holt–Winters: `models/hw.py`
  - MLP: `models/mlp.py`
  - XGBoost: `models/xgbRegressor.py`
- Dataset: `dataset/demographic.csv` (mensile)

---

## 0) Operational Analytics / Predictive Analytics (slide 0–1)
**Idea chiave delle slide:** usare dati storici per produrre previsioni e confrontare metodi diversi con metriche e validazione.

**Nel codice (`demographic.py`):** l’intero script è una pipeline end‑to‑end:
1) carica serie temporale
2) split train/validation/test
3) analisi stazionarietà + trasformazioni (preprocessing)
4) addestra 3 famiglie di algoritmi (statistico, neurale, regression trees)
5) valuta e confronta (RMSE/MAE/MAPE)

---

## 1) Descriptive / Inferential Statistics (slide 6–7)
### 1.1 Serie temporale e campioni
**Slide:** popolazione vs campione; descrizione di dati; inferenza tramite test.

**Codice:**
- Caricamento e casting:
  - `pd.read_csv(..., parse_dates=['date'], index_col='date')`
  - `.asfreq('ME')`: impone frequenza **Month-End** (mensile), utile per modelli stagionali.
  - `ts['values'].astype('float32')`: tipo numerico compatto.

### 1.2 Split Train/Validation/Test
**Slide (valutazione statistica):** separare dati per validare iperparametri senza “guardare” il test.

**Codice:**
- `train, validation, test = ts[:111], ts[111:123], ts[123:]` (80/10/10)
- `train_validation = ts[:123]` per ri‑addestrare il modello finale dopo la scelta iperparametri.

### 1.3 Metriche di errore
**Slide:** misure di errore e confronto modelli.

**Codice (`utils.py`):**
- `mean_absolute_error` → **MAE**
- `root_mean_squared_error` → **RMSE**
- `mean_absolute_percentage_error` → **MAPE** (×100)

Interpretazione rapida:
- MAE: errore medio in unità della serie.
- RMSE: penalizza più gli errori grandi.
- MAPE: errore percentuale (attenzione se valori vicino a 0).

---

## 2) Predictive Data Preprocessing (slide 2)
### 2.1 Motivazione: rendere la serie “adatta” ai modelli
**Slide:** preprocessing per soddisfare assunzioni (es. stazionarietà) o migliorare stabilità/learning.

**Codice (SARIMA):** applica **Box–Cox** solo per SARIMA:
- `train_boxcox, lmbda = boxcox(train['values'])`
- `train_validation_boxcox, lmbda_tv = boxcox(train_validation['values'])`

Box–Cox stabilizza varianza e rende la distribuzione più “gaussiana”, spesso utile per modelli lineari.

### 2.2 Stazionarietà: ADF test
**Slide:** inferential statistics + test per proprietà della serie.

**Codice:**
- `adfuller(train)` (p-value > 0.05 → non stazionaria)
- dopo differenze (vedi sotto) l’ADF viene ripetuto per verificare la stazionarietà.

### 2.3 ACF/PACF
**Slide:** strumenti diagnostici per ordini AR/MA e stagionalità.

**Codice:**
- `utils.plot_acf_pacf(train, ...)`
- `plot_acf` e `plot_pacf` da `statsmodels`

### 2.4 Differencing e seasonal differencing
**Slide:** rendere stazionaria una serie con trend/stagionalità.

**Codice:**
1) differenza prima (d=1): `train_boxcox.diff().dropna()`
2) differenza seconda (d=2): `train_diff.diff().dropna()`
3) differenza stagionale (D=1, m=12): `train_diff2.diff(12).dropna()`

Questo riflette una scelta coerente con una serie mensile M3 con stagionalità annuale.

---

## 3) Predictive Statistical Models (slide 3)
### 3.1 SARIMA con auto_arima (pmdarima)
**Slide:** ARIMA/ARMA, integrazione (I), stagionalità (SARIMA), selezione ordini.

**Codice:**
- `pm.auto_arima(train_boxcox, test='adf', d=2, D=1, m=12, seasonal=True, ...)`
  - `d=2`: differencing non stagionale (già “motivato” dagli ADF post-differenze)
  - `D=1, m=12`: stagionalità annuale
  - `max_p/max_q/max_P/...`: range di ricerca
  - `trace=True`: stampa modelli provati
  - criterio tipico: AIC (interno a auto_arima)

**Fit e forecast:**
- `auto_sarima_fit = auto_sarima.fit(train_boxcox)`
- `predict(n_periods=len(validation))`
- ri‑fit finale su `train_validation_boxcox`, poi forecast su test.

**Postprocessing (inversa Box–Cox):**
- `inv_boxcox(preds, lmbda)` / `lmbda_tv`

### 3.2 Diagnostica residui
**Slide:** validare assunzioni del modello (residui ~ rumore bianco).

**Codice:**
- `auto_sarima.plot_diagnostics(figsize=(12,8))`

---

## 4) Predictive Statistical Models — Holt–Winters (slide 3 + concetti di smoothing)
### 4.1 Perché Holt–Winters
**Slide:** metodi classici di forecasting; componenti livello/trend/stagionalità.

**Codice (`models/hw.py`):**
- usa `statsmodels.tsa.holtwinters.ExponentialSmoothing`
- grid search su:
  - `trend`: None/add/mul
  - `seasonal`: add/mul
  - `damped_trend`: True/False
  - `use_boxcox`: True/False
  - `seasonal_periods=12`

### 4.2 Grid search e scelta del best model
**Slide 8 (parameter fitting):** selezione parametri/iperparametri minimizzando una metrica.

**Codice:**
- `holt_winters_grid_search(... scoring_func=root_mean_squared_error)`
- per ogni combinazione:
  - fit su train
  - forecast su validation
  - calcolo RMSE
  - keep best

### 4.3 Training finale
**Codice:**
- `train_final_and_predict(train_validation, n_forecast=len(test), best_params=...)`

---

## 5) Predictive Neural Models — MLP (slide 4)
### 5.1 Impostazione come problema supervisionato
**Slide:** trasformare una serie in input/output con finestra (look-back).

**Codice (`models/mlp.py`):**
- `create_dataset(arrdata, look_back)` crea:
  - `X`: finestre di lunghezza `look_back`
  - `y`: valore successivo

### 5.2 Scaling (StandardScaler)
**Slide 2:** preprocessing per migliorare training NN.

**Codice:**
- `scale_series(train, val)`
- `inverse_transform(...)` per tornare alla scala originale.

### 5.3 Architettura e training
**Slide 4:** MLP con attivazioni non lineari; ottimizzazione con gradiente.

**Codice:**
- `MLP`: `Linear(look_back→hidden)` + attivazione (`relu` o `tanh`) + `Dropout` + `Linear(hidden→1)`
- loss: `nn.MSELoss()`
- optimizer: `Adam(lr=...)`
- mini-batch: `DataLoader(..., shuffle=True)`
- early stopping “semplice”: stop se loss non migliora per `patience` epoche.

### 5.4 Forecast ricorsivo (multi-step)
**Slide:** previsione multi-step spesso fatta in modo iterativo.

**Codice:**
- `recursive_forecast(model, initial_window, steps)`:
  - predice 1 passo
  - “shift” finestra e inserisce la predizione
  - ripete per `steps`

### 5.5 Grid search iperparametri
**Slide 8:** parameter fitting/selection.

**Codice (usato nello script):**
- `train_and_evaluate_grid(train.values, validation.values, param_grid, EPOCHS, ...)`
- nel progetto è usata una *griglia ridotta* (una sola combinazione), ma lo scheletro supporta molte combinazioni.

---

## 6) Predictive Machine Learning Models — XGBoost (slide 5)
### 6.1 Perché regression trees / boosting
**Slide 5:** alberi + ensemble (boosting) per modellare non linearità senza NN.

**Codice (`models/xgbRegressor.py`):**
- modello: `xgboost.XGBRegressor(objective='reg:squarederror')`
- input: finestre `look_back` (come per MLP)

### 6.2 Rolling forecast
**Slide:** per multi-step, anche con modelli regressivi si usa spesso forecast iterativo.

**Codice:**
- `rolling_forecast(model, x_train, steps_ahead, look_back)`:
  - prende l’ultima finestra disponibile
  - predice next
  - aggiorna la finestra con la predizione

### 6.3 Grid search
**Slide 8:** selezione iperparametri su validation.

**Codice:**
- `grid_search_xgb(train, validation, param_grid)`:
  - per ciascuna combinazione:
    - crea dataset con `create_dataset`
    - allena `train_xgb_model`
    - valuta RMSE su validation con forecast rolling
  - mantiene best

Nello script, anche qui viene usata una griglia “già scelta” (una combinazione) rispetto allo spazio completo commentato.

---

## 7) Reproducibility / Good practice (collegamento trasversale alle slide)
### 7.1 Seeding
**Codice (`demographic.py` + `models/mlp.py`):**
- `seed_everything(SEED)` imposta seed per `random`, `numpy`, `torch` e abilita determinismo.

**Perché:** nelle slide di ML/NN la ripetibilità è importante per confronti corretti (stessa split + stesso training stochastic).

### 7.2 Visualizzazioni
**Codice:**
- plot split train/val/test
- ACF/PACF
- diagnostic ARIMA
- plot previsioni su validation e test con `utils.plot_predictions`

---

## 8) Come il progetto soddisfa i vincoli del PDF del corso
Il PDF dell’esame richiede:
- preprocessing
- almeno 3 algoritmi (statistico, neurale, regression trees)
- confronto statistico della qualità delle previsioni

**Questo script include:**
- preprocessing (Box–Cox, differencing, ADF, ACF/PACF)
- statistico: SARIMA + Holt–Winters
- neurale: MLP
- regression trees: XGBoost
- confronto: metriche su validation e test

Nota: sono usate librerie extra (es. `pmdarima`, `torch`, `xgboost`, `scipy`). Se il corso impone restrizioni “solo librerie viste a lezione”, conviene verificare che siano accettate.
