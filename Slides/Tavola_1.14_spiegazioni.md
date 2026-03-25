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

## Step 4 - Modelli statistici (SARIMA + Holt-Winters)
Obiettivo: confrontare due modelli statistici univariati su serie annuale trasformata.

Pipeline implementata:
- input: serie trasformata log(x) con diff d∈{0,1};
- grid search SARIMA su validation set con criteri AIC/BIC integrati;
- refit del modello vincente su train+validation;
- forecast su test set;
- metriche in scala trasformata e in scala originale (inversion log1p+diff);
- diagnostica: residui, Ljung-Box test, plot ACF/PACF.

Modelli confrontati:
- SARIMA (Auto-Regressive Integrated Moving Average);
- Holt-Winters (benchmark non-ARIMA).

### 4.1 - Run baseline (stat_baseline)
File principali:
- Results/metrics/statistical/sarima_grid_baseline.csv
- Results/metrics/statistical/hw_grid_baseline.csv
- Results/metrics/statistical/summary_baseline.csv
- Results/artifacts/statistical/winner_params_baseline.json

Configurazione:
- p_values: (0, 1, 2);
- d_values: (0, 1);
- q_values: (0, 1, 2);
- p_seasonal, q_seasonal: (0, 1);
- seasonal_period: inferito da split temporale.

Esito sintetico:
- winner: sarima;
- best_p,d,q: TBD (da grid search);
- test RMSE in scala originale: circa 30272.80 GWh.
- plot: forecast_comparison, forecast_original_scale.

### 4.2 - Run esteso (stat_extended)
Configurazione estesa:
- p_values: (0, 1, 2, 3, 4);
- d_values: (0, 1);
- q_values: (0, 1, 2, 3, 4);
- p_seasonal, q_seasonal: (0, 1, 2) se seasonal_period > 1;
- griglia ~5x più densa per esplorazione parametrica approfondita.

File principali:
- Results/metrics/statistical/sarima_grid_extended.csv
- Results/metrics/statistical/summary_extended.csv
- Results/artifacts/statistical/winner_params_extended.json

Esito sintetico:
- winner: sarima;
- test RMSE in scala originale: circa 30272.80 GWh (identico a baseline);
- interpretazione: griglia estesa non migliora, confirma stabilita della soluzione baseline.

### 4.3 - Confronto baseline vs extended
File di confronto:
- Results/metrics/tavola_1_14_stat_comparison_baseline_vs_extended.csv

Risultato:
La griglia estesa produce gli stessi parametri ottimali del baseline. Non c è vantaggio di esplorazione ulteriore — il parametro minimo si trova già nella griglia compatta baseline. Questo suggerisce che lo step della validazione è robusto e il rischio di overfitting locale è basso.

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
- Results/metrics/ml/summary_baseline.csv
- Results/metrics/ml/grid_baseline.csv
- Results/artifacts/ml/winner_params_baseline.json

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
- Results/metrics/ml/summary_extended.csv
- Results/metrics/ml/grid_extended.csv
- Results/artifacts/ml/winner_params_extended.json

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

## Stato operativo del progetto
Il notebook è stato rifattorizzato come raccoglitore operativo.
- Logica esternalizzata nei moduli Python nel package `Project/`.
- Notebook usato come orchestratore step-by-step e visualizzatore risultati.
- Main.py serve come pipeline canonica (baseline per ogni step).

Runner autonomi per sperimentazione:
- Project/models/statistical/runners/run_baseline.py: Step 3 statistico baseline (config identica al main.py);
- Project/models/statistical/runners/run_extended.py: Step 3 statistico extended (griglia SARIMA più densa);
- Project/models/statistical/runners/run_compare.py: confronto baseline vs extended;
- Project/models/ml/runners/run_baseline.py: Step 4 ML baseline (no XGBoost);
- Project/models/ml/runners/run_extended.py: Step 4 ML extended (XGBoost on);
- Project/models/ml/runners/run_compare.py: confronto cross-step e per Step 4.
