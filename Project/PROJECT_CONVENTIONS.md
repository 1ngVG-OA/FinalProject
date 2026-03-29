# Project conventions

Questa guida definisce convenzioni minime per mantenere il progetto ordinato.

## 1) Directory operative

- `Datasets/`: dati sorgente (csv/xls) forniti o raccolti.
- `Datasets/processed/`: versioni preprocessate organizzate per serie (`Datasets/processed/<SerieOutputName>/...`).
- `Project/`: tutta la logica applicativa.
- `Results/`: output persistenti organizzati per serie (`Results/<SerieOutputName>/...`).

## 2) Struttura interna di Project

- `Project/config/`: configurazioni centrali (path, target, split, seed, metriche).
- `Project/preprocessing/`: pulizia, trasformazioni, stazionarieta, feature engineering.
- `Project/models/statistical/`: modelli statistici (ARIMA/SARIMA/ETS).
- `Project/models/ml/`: modelli ML non neurali.
- `Project/models/neural/`: modelli neurali.
- `Project/evaluation/`: metriche e confronto modelli.
- `Project/evaluation/inferential/`: test statistici (es. Diebold-Mariano).
- `Project/pipeline/`: orchestrazione dei passi end-to-end.

## 3) Struttura interna di Results

- `Results/<SerieOutputName>/plots/descriptive/`: grafici EDA e distribuzioni.
- `Results/<SerieOutputName>/plots/preprocessing/`: ACF/PACF, trasformazioni, stazionarieta.
- `Results/<SerieOutputName>/plots/statistical/`: forecast e diagnostica del blocco statistico.
- `Results/<SerieOutputName>/plots/ml/`: forecast e diagnostica del blocco ML.
- `Results/<SerieOutputName>/plots/neural/`: forecast e diagnostica del blocco neurale.
- `Results/<SerieOutputName>/metrics/`: tabelle metriche per tutti gli step della serie.
- `Results/<SerieOutputName>/artifacts/`: file serializzati, configurazioni e winner params della serie.

`<SerieOutputName>` viene derivato automaticamente dalla chiave serie configurata in `config.py`.

## 4) Convenzione nomi file (consigliata)

Formato generale:

`{dataset}_{fase}_{modello}_{versione}.{estensione}`

Esempi:

- `tavola_1_14_descriptive_summary_v1.csv`
- `tavola_1_14_forecast_arima_v1.csv`
- `tavola_1_14_metrics_comparison_v1.csv`
- `tavola_1_14_plot_acf_pacf_v1.png`

Regole:

- usare solo minuscole, underscore e numeri;
- evitare spazi e accenti nei nomi file;
- aggiungere `v1`, `v2`, ... quando cambia il contenuto in modo sostanziale.

## 5) Entry point e scope

- `main.py` in root: esecuzione orchestrata della pipeline, con selezione serie tramite config e parametro `--series`.
- `Tavola_1.14.ipynb` in root: notebook descrittivo/esplicativo del workflow.

Nota: in fase di consegna finale, privilegiare l'esecuzione script-based se richiesto dal corso.

## 6) Aggiunta di una nuova serie

- aggiungere una nuova entry in `config.py` dentro `SERIES_REGISTRY`;
- definire almeno dataset sorgente, indice colonna target e nome della serie;
- eseguire `python main.py --series <series_key>` per lanciare la pipeline sulla nuova serie;
- tutti gli output verranno salvati automaticamente in `Results/<SerieOutputName>/...` e `Datasets/processed/<SerieOutputName>/...`.
