# Ogrin — Code Walkthrough vs Slides (Operational Analytics)

Questo report collega *ogni parte del codice* del progetto **ogrin** con i concetti teorici delle slide (ordine: introduzione → preprocessing → modelli statistici → modelli neurali → modelli ML → statistica/valutazione → parameter fitting).

## Mappa file (cosa guardare)
- Script principale: `ogrin.py`
- Utility comuni (plot + metriche): `utils.py`
- Modelli:
  - MLP: `models/mlp.py`
  - XGBoost: `models/xgbRegressor.py`
- Dataset: `dataset/ogrin.csv` (giornaliero)

Differenza strutturale rispetto a `demographic.py`:
- qui non c’è stagionalità esplicita (serie giornaliera breve)
- la parte “statistica” usa **ARMA** (auto_arima senza stagionalità e senza differencing)

---

## 0) Operational Analytics / Predictive Analytics (slide 0–1)
**Slide:** forecasting come parte dell’analisi operativa; confronto tra approcci.

**Nel codice (`ogrin.py`):** pipeline completa:
1) carica e prepara serie
2) split train/validation/test
3) diagnosi stazionarietà (ADF + ACF/PACF)
4) modelli: ARMA (statistico), MLP (neurale), XGBoost (tree-based)
5) valutazione con metriche

---

## 1) Descriptive / Inferential Statistics (slide 6–7)
### 1.1 Caricamento e frequenza
**Codice:**
- `pd.read_csv(...).asfreq('D')`: impone frequenza giornaliera.
- cast a `float32`.

### 1.2 Split Train/Validation/Test
**Slide:** separazione corretta per tuning e test.

**Codice:**
- `train, validation, test = ts[:106], ts[106:136], ts[136:]` (70/20/10)
- `train_validation = ts[:136]` per training finale.

### 1.3 Metriche
**Codice (`utils.py`):**
- MAE, RMSE, MAPE come nel progetto demographic.

---

## 2) Predictive Data Preprocessing (slide 2)
### 2.1 Stazionarietà (ADF)
**Slide:** test statistici per capire se serve differencing.

**Codice:**
- `adfuller(train)`
- la relazione con ARMA/ARIMA è diretta: se stazionaria, ARMA può essere sufficiente.

### 2.2 ACF/PACF
**Slide:** pattern ACF/PACF suggeriscono componenti AR/MA.

**Codice:**
- `utils.plot_acf_pacf(train, ...)`

### 2.3 Differencing diagnostico
**Slide:** differencing come trasformazione per rimuovere trend.

**Codice:**
- `train_diff = train.diff().dropna()`
- poi `adfuller(train_diff)`

Qui il differencing viene usato principalmente come *diagnostica*; il modello finale viene impostato come stazionario (vedi ARMA sotto).

---

## 3) Predictive Statistical Models — ARMA (slide 3)
### 3.1 Auto-selezione ARMA(p,q)
**Slide:** ARMA per serie stazionarie, senza integrazione (d=0) e senza stagionalità.

**Codice (`ogrin.py`):**
- `pm.auto_arima(train, stationary=True, test='adf', seasonal=False, ...)`
  - `stationary=True`: forza la scelta su modelli senza differencing
  - `seasonal=False`: no SARIMA
  - ricerca su `max_p/max_q`

**Fit e forecast:**
- fit su train → forecast su validation
- fit su train+validation → forecast su test

### 3.2 Diagnostica residui
**Codice:**
- `auto_sarima_fit.plot_diagnostics(figsize=(12,8))`

---

## 4) Predictive Neural Models — MLP (slide 4)
Il blocco NN è identico come struttura a `demographic`, cambia solo la griglia scelta.

### 4.1 Trasformazione a supervised learning
**Codice (`models/mlp.py`):**
- `create_dataset(series, look_back)` produce finestre e target.

### 4.2 Scaling
**Slide 2:** fondamentale per NN.

**Codice:**
- StandardScaler su train e applicazione su validation/test.

### 4.3 Architettura e training
**Slide 4:** MLP per non linearità.

**Codice (parametri usati nello script):**
- `look_back = 28` (quasi un “mese” di giorni)
- `hidden_size = 6`, `activation='tanh'`, `batch_size=32`, `lr=0.001`
- `epochs = 500` con early stopping (patience=10)

### 4.4 Forecast multi-step ricorsivo
**Codice:**
- `recursive_forecast` come per demographic.

---

## 5) Predictive Machine Learning Models — XGBoost (slide 5)
### 5.1 Regressione su finestre + boosting
**Slide:** tree-based ML + ensemble.

**Codice (`models/xgbRegressor.py` + `ogrin.py`):**
- crea dataset con `look_back = 28`
- `XGBRegressor` con iperparametri scelti via grid search (qui già fissati nello script)

### 5.2 Rolling forecast
**Codice:**
- `rolling_forecast` per previsione iterativa su più step.

### 5.3 Selezione iperparametri (Parameter fitting)
**Slide 8:** valutazione su validation per scegliere settaggi.

**Codice:**
- `grid_search_xgb(train.squeeze(), validation.squeeze(), param_grid)`
- seleziona combinazione con RMSE minimo.

---

## 6) Statistica e confronto modelli (slide 6–7)
**Slide:** confronto quantitativo; evitare overfitting usando validation.

**Codice:**
- per ogni modello:
  - stampa metriche su validation
  - stampa metriche su test
  - plot delle previsioni

Nota: il confronto è “statistico” nel senso di *metriche di errore*; non c’è un test formale di significatività tra forecast (es. Diebold–Mariano). Se il docente intende “statistical comparison” come test, qui non è implementato.

---

## 7) Reproducibility (trasversale)
**Codice:**
- `seed_everything(SEED)` imposta seed per ripetibilità (numpy/torch/random).

---

## 8) Coerenza con i vincoli del PDF dell’esame
Questo script include:
- preprocessing e analisi (ADF, ACF/PACF, differencing diagnostico)
- almeno 3 famiglie:
  - statistico: ARMA
  - neurale: MLP
  - regression trees: XGBoost
- confronto su validation/test con metriche

Nota: come per demographic, sono usate librerie extra (pmdarima, torch, xgboost). Verificare accettazione rispetto alle regole del corso.
