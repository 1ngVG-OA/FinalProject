# Tavola 1.14 - Spiegazioni complete (allineate al progetto attuale)

## Step 0 - Inquadramento del problema
- Tipo di analisi: predictive analytics su serie temporale univariata.
- Target: produzione lorda totale.
- Frequenza: annuale (nessuna stagionalita intra-annuale classica).
- Obiettivo: forecasting su validation e test con pipeline leakage-safe.

## Step 1 - Analisi descrittiva
Obiettivo: comprendere struttura, scala e criticita della serie originale prima del preprocessing.

Attivita principali:
- caricamento e pulizia del CSV ISTAT;
- costruzione serie numerica annuale ordinata;
- statistiche descrittive (tendenza centrale e dispersione);
- trend test (lineare + Spearman) e outlier locali sulle variazioni YoY;
- grafici descrittivi.

Evidenze sintetiche:
- trend crescente molto marcato;
- forte eterogeneita di scala tra inizio e fine serie;
- poche anomalie locali, nessun problema dominante di outlier globali.

## Step 2 - Preprocessing operativo
Obiettivo: stabilizzare la serie e costruire split cronologici per il modeling.

Pipeline:
- split temporale senza shuffle;
- trasformazioni candidate (log1p, differencing, scaling opzionale);
- test ADF/KPSS (e Shapiro quando richiesto);
- artifact e plot di diagnostica.

Split effettivo del run corrente:
- train: 90 osservazioni;
- validation: 19 osservazioni;
- test: 20 osservazioni.

## Step 2b - Selezione automatica configurazione
Obiettivo: scegliere la migliore configurazione di preprocessing per famiglia di modelli.

Profili usati:
- statistical;
- ml;
- neural.

Nota importante:
- i runner statistici usano profilo `statistical`;
- i runner ML usano profilo `ml`.

## Step 3 - Modellazione statistica (SARIMA + Holt-Winters)
Obiettivo: confrontare modelli statistici con selezione winner su validation.

Runner:
- `Project/models/statistical/runners/run_baseline.py`
- `Project/models/statistical/runners/run_extended.py`
- `Project/models/statistical/runners/run_compare.py`

Output principali:
- `Results/metrics/statistical/summary_baseline.csv`
- `Results/metrics/statistical/summary_extended.csv`
- `Results/metrics/statistical/comparison.csv`
- `Results/artifacts/statistical/winner_params_baseline.json`
- `Results/artifacts/statistical/winner_params_extended.json`

Esiti sintetici (scala originale, ultimo run):
- winner baseline: `sarima`;
- winner extended: `sarima`;
- SARIMA test RMSE: circa 66871.0 GWh;
- Holt-Winters test RMSE: circa 68999.6 GWh.

Interpretazione:
- la griglia estesa non supera in modo sostanziale il baseline;
- SARIMA resta il riferimento statistico piu robusto.

## Step 4 - Modellazione Machine Learning non-neurale
Obiettivo: confrontare modelli tree-based con feature di lag e forecast ricorsivo multi-step.

Runner:
- `Project/models/ml/runners/run_baseline.py`
- `Project/models/ml/runners/run_extended.py`
- `Project/models/ml/runners/run_compare.py`

Output principali:
- `Results/metrics/ml/summary_baseline.csv`
- `Results/metrics/ml/summary_extended.csv`
- `Results/metrics/ml/comparison.csv`
- `Results/artifacts/ml/winner_params_baseline.json`
- `Results/artifacts/ml/winner_params_extended.json`

Modelli confrontati:
- Decision Tree Regressor;
- Random Forest Regressor;
- Gradient Boosting Regressor;
- XGBoost Regressor (solo run extended).

Esiti sintetici (scala originale, ultimo run):
- baseline winner: `gradient_boosting`, lookback 12, test RMSE circa 64212.2 GWh;
- extended winner: `gradient_boosting`, lookback 12, test RMSE circa 64470.1 GWh;
- migliore modello ML complessivo su test: baseline gradient boosting.

## Confronto operativo rapido
Confronto su test RMSE in scala originale (piu basso e migliore):
- Step 3 SARIMA: circa 66871.0 GWh;
- Step 4 ML baseline (GBR): circa 64212.2 GWh;
- Step 4 ML extended (GBR): circa 64470.1 GWh.

Lettura metodologica:
- nello stato attuale degli esperimenti, il migliore RMSE test e nel blocco ML baseline;
- la differenza rispetto a SARIMA non e ampia e va interpretata insieme alla stabilita delle metriche su validation;
- il run extended con XGBoost amplia la ricerca ma non migliora il risultato finale rispetto al baseline ML.

