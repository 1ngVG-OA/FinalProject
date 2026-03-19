## Pipeline di analisi completa — Tavola_1.14.csv

### 0 — Inquadramento del problema *(Slide 0 – Operational Analytics)*
- Classificare il tipo di analytics: descriptive → diagnostic → **predictive** → prescriptive
- Definire la variabile target e l'orizzonte di forecast (breve / medio / lungo termine)
- Decidere modello causale vs. time-series puro (si hanno variabili esogene affidabili?)

> PREDICTIVE
> PRODUZIONE LORDA - TOTALE (Unica colonna disponibile dal 1883)
> Ho valori annuali => non ho una stagionalità => ARIMA
> Approccio Time Series UNIVARIATO (Le altre colonne non sono driver indipendenti)
> Previsioni a medio lungo termine
> Le rotture per ora non le gestisco, ma le considero comunque nel tutto.
---

### 1 — Preparazione e comprensione del dataset *(Slide 6 – Descriptive Statistics)*
- Caricamento dati, parsing date, controllo tipi e valori mancanti (`....` nella Tavola_1.14)
- Statistica descrittiva: media, mediana, varianza, deviazione standard, IQR, quantili
- Visualizzazione della distribuzione dei valori (istogramma, box-plot)
- Plot della serie temporale → ispezione visiva di trend, stagionalità, outlier, rotture strutturali
- Scelta della colonna target (es. _Produzione lorda totale_ o _Consumo totale_)

---

### 2 — Preprocessing *(Slide 2 – Predictive Data Preprocessing)*
- **Test di stazionarietà**: ADF test (H0: serie non stazionaria)
- **ACF / PACF**: identificare autocorrelazioni e lag significativi
- **Trasformazioni**:
  - log / Box-Cox → stabilizzazione varianza
  - differencing (I o II ordine) → rimozione trend / stazionarizzazione
- **Decomposizione** (trend + stagionalità + residuo) → verifica componenti
- Eventuale destagionalizzazione prima della modellazione
- Controllo residuo dopo trasformazione (ACF / ADF di verifica)

---

### 3 — Split temporale train / validation / test *(Slide 1 – Predictive Analytics, Slide 2)*
- Split **rigorosamente temporale** (no shuffle): es. 70% train, 15% val, 15% test
- Tutti i parametri delle trasformazioni (media, std, lambda Box-Cox) stimati **solo** sul train set, poi applicati a val e test (no data leakage)
- Definire l'orizzonte di forecast `h` e la metrica di holdout

---

### 4 — Selezione e stima del modello statistico *(Slide 3 – Statistical Models, Slide 8 – Parameter Fitting)*
- **Procedura Box-Jenkins**:
  1. Identificazione ordini `(p,d,q)` da ACF/PACF
  2. Stima parametri (MLE o metodo dei momenti)
  3. Diagnostica residuale: media≈0, varianza stabile, ACF residui non significativa, Q-Q plot, densità
  4. Iterazione se residui hanno ancora struttura
- **Candidati da confrontare**: AR, MA, ARMA, ARIMA, SARIMA `(p,d,q)×(P,D,Q)_m`, SES, Holt-Winters (additivo / moltiplicativo), Theta
- **Selezione automatizzata**: grid search su `(p,d,q)` + `(P,D,Q)` con AIC/AICc come criterio di confronto (AICc su campioni finiti)
- Eventuale LASSO autoregressivo per selezione lag sparsa
- Aggiunta di variabili esogene → SARIMAX (se disponibili nella Tavola_1.14, es. consumi come regressori)

---

### 5 — Modelli Machine Learning *(Slide 5 – ML Models)*
- Conversione serie → dataset supervisionato con **sliding window** (matrice di lag come feature)
- Candidati: **SVR** (con kernel RBF / lineare, tuning `C`, `ε`), **Random Forest Regressor**
- Feature selection: **RFE** (Recursive Feature Elimination) per identificare lag informativi
- Tuning iperparametri con grid/random search su validation set (split temporale)

---

### 6 — Modelli Neurali *(Slide 4 – Neural Models)*
- Sliding window → dataset supervisionato (stesso schema dello step 5)
- Scaling con **MinMaxScaler** (fit su train, transform su val/test)
- Candidati: **MLP feedforward** (baseline neurale), **LSTM** (memoria su sequenze lunghe)
- Training: loss `MSELoss`, ottimizzatore Adam, loop su epoche, early stopping
- Forecast multi-step iterativo (previsione entra nella finestra successiva)
- Inversione scaling per ritornare alla scala originale

---

### 7 — Fitting e ottimizzazione parametri *(Slide 8 – Parameter Fitting)*
- Scelta della **funzione di loss** coerente con l'obiettivo (MAE, MSE, MAPE)
- Ottimizzatori gradient-based (Adam, BFGS) per reti e modelli differenziabili
- Per spazi di parametri complessi/non convessi: **PSO** (Particle Swarm Optimization) come alternativa globale
- Documentare `pbest` / `gbest` e convergenza

---

### 8 — Valutazione delle performance *(Slide 1, Slide 8)*
- Calcolo metriche su test set in scala originale: **BIAS/ME**, **MAE**, **RMSE**, **MAPE**
- Confronto multi-modello in una tabella riassuntiva
- Plot forecast sovrapposto a valori reali (train + test)
- Comunicare **intervalli di previsione** (non solo il punto medio): `μ ± σ` da distribuzione residua

---

### 9 — Confronto statistico tra modelli *(Slide 7 – Inferential Statistics)*
- **Diebold-Mariano test**: verificare se la differenza di errore tra due modelli è statisticamente significativa (non solo numerica)
- Interpretazione statistica DM e p-value (soglia `±1.96` per `α=0.05`)
- Per confronti multipli: correzione per test multipli (Wilcoxon-Holm), critical difference diagrams
- Intervalli di confidenza sui forecast per comunicare incertezza operativa

---

### 10 — Conclusioni e analytics prescrittiva *(Slide 9 – Prescriptive Analytics)* *(opzionale)*
- Usare il forecast migliore come input per un modello decisionale (es. pianificazione energetica)
- Definire la politica operativa ottima: LP/ILP/MILP se il problema ha vincoli lineari
- Metaeuristiche (VNS, ALNS) per problemi combinatori complessi

---

**Nota sulla Tavola_1.14.csv**: la serie copre 1883–2014 con frequenza annuale, valori in GWh. Il dataset ha colonne multiple (produzione + consumo per settore), quindi prima dello step 0 va scelto esplicitamente il target (o si fanno analisi separate / multivariate). I valori `....` sono missing data da gestire nello step 1.
