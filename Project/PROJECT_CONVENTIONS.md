# Project conventions

Questa guida definisce convenzioni minime per mantenere il progetto ordinato.

## 1) Directory operative

- `Datasets/`: dati sorgente (csv/xls) forniti o raccolti.
- `Datasets/processed/`: versioni preprocessate pronte per i modelli.
- `Project/`: tutta la logica applicativa.
- `Results/`: output persistenti (plot, metriche, report, artefatti).

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

- `Results/plots/descriptive/`: grafici EDA e distribuzioni.
- `Results/plots/preprocessing/`: ACF/PACF, trasformazioni, stazionarieta.
- `Results/plots/forecasting/`: forecast e residui per modello.
- `Results/plots/comparison/`: confronto tra modelli.
- `Results/metrics/`: tabelle metriche aggregate e per modello.
- `Results/reports/`: report finali e note metodologiche.
- `Results/artifacts/`: file serializzati (parametri migliori, oggetti utili).

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

- `main.py` in root: esecuzione orchestrata della pipeline.
- `Tavola_1.14.ipynb` in root: notebook descrittivo/esplicativo del workflow.

Nota: in fase di consegna finale, privilegiare l'esecuzione script-based se richiesto dal corso.
