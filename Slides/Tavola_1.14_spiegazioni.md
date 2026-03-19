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

## Stato operativo del notebook
Il notebook e stato rifattorizzato come raccoglitore operativo.
- Logica esternalizzata nei moduli Python in Mains/notebook_steps.
- Notebook usato come orchestratore step-by-step e visualizzatore risultati.

Moduli usati:
- Mains/notebook_steps/step1_dataset.py
- Mains/notebook_steps/step2_preprocessing.py
- Mains/notebook_steps/step3_split.py
- Mains/notebook_steps/step4_statistical.py
