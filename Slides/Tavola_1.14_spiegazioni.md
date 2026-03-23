# Tavola 1.14 - Spiegazioni complete

## Step 0 - Inquadramento del problema
- Tipo di analisi: predictive analytics.
- Target scelto: Produzione lorda totale, cioe la colonna piu lunga e continua dal 1883.
- Frequenza annuale: non c e stagionalita intra-annuale classica.
- Approccio: time series univariato.
- Obiettivo: previsioni di medio-lungo termine.
- Rotture strutturali: riconosciute ma non modellate esplicitamente nella prima iterazione.

## Step 1 - Preparazione e comprensione del dataset
Obiettivo: costruire una base dati pulita e capire struttura e scala della serie prima del preprocessing.

Attivita:
- caricamento del file grezzo;
- estrazione della variabile target;
- conversione anno e valori in formato numerico;
- controllo missing values, dimensione e statistiche descrittive;
- visualizzazione serie, distribuzione e boxplot.

Interpretazione sintetica:
La serie mostra un trend crescente di lungo periodo con cambi di regime in fasi storiche rilevanti. La distribuzione e asimmetrica a destra: i valori recenti sono molto piu alti di quelli iniziali. Questo giustifica il controllo di stazionarieta e l uso di trasformazioni prima del modello ARIMA.

## Step 2 - Preprocessing
Obiettivo: verificare stazionarieta e selezionare trasformazioni candidate.

Trasformazioni confrontate:
- level;
- log;
- diff_1;
- log_diff_1;
- diff_2.

Diagnostica usata:
- ADF test;
- ACF/PACF su tutte le trasformazioni;
- approfondimento dedicato a diff_2 per valutare anche rischio di over-differencing.

Interpretazione sintetica:
La serie in livello non e stazionaria. Le trasformazioni riducono il problema in misura diversa. diff_2 risulta la candidata piu forte lato stazionarizzazione, ma va sempre verificata insieme a residui e comportamento out-of-sample per evitare differenziazione eccessiva.

## Step 3 - Split temporale train/validation/test
Obiettivo: separare le fasi di stima, selezione e verifica finale senza leakage.

Scelta:
- split cronologico 80/10/10;
- train ampio per stimare ARIMA su serie annuale;
- validation e test separati per tuning e verifica finale.

Interpretazione sintetica:
Lo split temporale preserva la logica del forecasting reale e permette valutazioni fuori campione robuste.

## Step 4 - Modello statistico
### Iterazione iniziale
- mini-grid ARIMA non stagionale con confronto di ordini su validation;
- refit su train+validation;
- forecast su test;
- diagnostica residui (tempo, istogramma, QQ plot, ACF residui).

### Confronto coerente con Step 2
Confronto esplicito tra due famiglie coerenti con stazionarieta osservata:
- log_d0: ARIMA sulla log-serie con d=0 (poi inversione exp);
- level_d2: ARIMA in livello con d=2.

Selezione:
- criterio principale: validation RMSE;
- controllo successivo: metriche su test e residui.

Nota sui warning statsmodels:
Il warning su non-invertible starting MA parameters e in genere un warning di inizializzazione. Non invalida da solo il risultato, ma va considerato insieme a convergenza, residui e performance out-of-sample.

## Step 5 - Modelli Machine Learning non-neurali (tree-based)
Obiettivo: confrontare modelli ML non-neurali su serie trasformata usando lag supervisionati e forecast multi-step ricorsivo.

Pipeline implementata:
- input: stessa serie preprocessata dello Step 2 e stesso split temporale dello Step 3;
- costruzione matrice supervisionata con lookback k (lag features);
- feature selection eseguita solo su train (metodo importance), poi applicata a validation/test;
- tuning su validation set con split temporale;
- refit su train+validation e forecast ricorsivo su test;
- metriche in scala trasformata e in scala originale (quando invertibile).

Modelli confrontati:
- Decision Tree Regressor;
- Random Forest Regressor;
- Gradient Boosting Regressor;
- XGBoost Regressor.

### 5.1 - Run baseline (v1)
File principali:
- Results/metrics/tavola_1_14_ml_summary_v1.csv
- Results/metrics/tavola_1_14_ml_grid_v1.csv
- Results/artifacts/tavola_1_14_ml_winner_params_v1.json

Esito sintetico:
- winner: gradient_boosting;
- lookback vincente: 12;
- test RMSE in scala originale: circa 19894.69.

### 5.2 - Run esteso con XGBoost (v2)
Configurazione estesa usata:
- XGBoost abilitato;
- griglia piu ampia rispetto al main corrente (piu combinazioni su depth, learning rate, ensemble size);
- stesso schema di validazione temporale e stesso preprocessing del baseline.

File principali:
- Results/metrics/tavola_1_14_ml_summary_xgb_v2.csv
- Results/metrics/tavola_1_14_ml_grid_xgb_v2.csv
- Results/artifacts/tavola_1_14_ml_winner_params_xgb_v2.json

Esito sintetico:
- winner: random_forest;
- lookback vincente: 12;
- best_params: n_estimators=200, max_depth=None, min_samples_leaf=1;
- test RMSE in scala originale: circa 25658.15.

### 5.3 - Confronto operativo rapido
Confronto su test RMSE in scala originale (piu basso e migliore):
- Step 3 statistico (SARIMA): circa 6301.98;
- Step 4 ML baseline v1: circa 19894.69;
- Step 4 ML esteso v2 (XGBoost on): circa 25658.15.

Interpretazione:
In questa serie annuale e con questa trasformazione, i modelli statistici restano nettamente piu stabili e accurati in scala originale rispetto ai tree-based non-neurali. L'estensione con XGBoost aumenta la copertura dello spazio iperparametri, ma non migliora il risultato finale sul test rispetto al baseline ML v1.

## Stato operativo del notebook
Il notebook e stato rifattorizzato come raccoglitore operativo.
- Logica esternalizzata nei moduli Python in Mains/notebook_steps.
- Notebook usato come orchestratore step-by-step e visualizzatore risultati.

Moduli usati:
- Mains/notebook_steps/step1_dataset.py
- Mains/notebook_steps/step2_preprocessing.py
- Mains/notebook_steps/step3_split.py
- Mains/notebook_steps/step4_statistical.py
- Mains/run_step4_ml_extended.py
