# Prompt Copilot - Ripresa progetto su altro PC

## Istruzioni d'uso
- Apri il progetto completo in VS Code.
- Copia e incolla TUTTO il blocco sotto in una nuova chat Copilot.
- Se vuoi massima continuita storica, allega anche il file Trascrizione_chat_completa.md.

---

## TESTO DA INCOLLARE IN COPILOT

Sei il mio assistente tecnico sul progetto FinalProject (forecasting time series annuale su Tavola_1.14).
Voglio riprendere esattamente lo stato del lavoro raggiunto su un altro PC.

### Contesto progetto
- Linguaggio: Python.
- Obiettivo: pipeline end-to-end con Step 1 descrittivo, Step 2 preprocessing, Step 3 modelli statistici, Step 4 modelli ML non-neurali.
- Serie target: Produzione lorda totale (dataset Tavola_1.14.csv).
- Approccio: time series univariato.

### Stato attuale importante
1) Step 3 statistico completato con confronto SARIMA vs Holt-Winters.
2) Step 4 ML non-neurali implementato in package dedicato.
3) Esiste un run baseline v1 (XGBoost disattivato nel main).
4) Esiste un run esteso v2 (XGBoost attivato) tramite script dedicato.
5) Report markdown aggiornato con confronto dei risultati.

### File chiave da leggere subito
- main.py
- Project/models/ml/model_config.py
- Project/models/ml/features.py
- Project/models/ml/runner.py
- Project/models/ml/plotting.py
- Mains/run_step4_ml_extended.py
- Slides/Tavola_1.14_spiegazioni.md
- Results/metrics/tavola_1_14_stat_summary_v1.csv
- Results/metrics/tavola_1_14_ml_summary_v1.csv
- Results/metrics/tavola_1_14_ml_summary_xgb_v2.csv
- Results/artifacts/tavola_1_14_ml_winner_params_v1.json
- Results/artifacts/tavola_1_14_ml_winner_params_xgb_v2.json

### Risultati noti (da verificare ricaricando i CSV)
- Step 3 SARIMA: RMSE test originale circa 6301.98.
- Step 4 ML v1: winner gradient_boosting, RMSE test originale circa 19894.69.
- Step 4 ML v2 (XGBoost on): winner random_forest, RMSE test originale circa 25658.15.

### Vincoli tecnici importanti
- Niente data leakage: feature selection e fit solo su train nelle fasi corrette.
- Split temporale rigoroso (no shuffle).
- Mantenere metriche anche in scala originale (quando invertibile).
- Non sovrascrivere output storici v1/v2 se non richiesto.

### Cosa voglio che tu faccia adesso
1. Verifica coerenza dell'ambiente e dipendenze.
2. Verifica che i file risultati citati esistano e siano leggibili.
3. Fornisci un breve audit: cosa e pronto, cosa manca, rischi principali.
4. Proponi il prossimo step operativo migliore tra:
   - aggiornamento notebook per presentazione,
   - esperimento ML v3 mirato (tuning RF/GBR/XGB),
   - consolidamento report finale.
5. Se trovi incongruenze, correggile direttamente con modifiche minime e spiegami cosa hai cambiato.

### Nota su esecuzione script Step 4 esteso
- Lo script Mains/run_step4_ml_extended.py aggiunge la root del progetto al sys.path per evitare ModuleNotFoundError su import Project quando eseguito standalone.

### Preferenze di output
- Risposte in italiano.
- Prima risultati/finding concreti, poi spiegazioni.
- Includi sempre path dei file toccati e metriche principali aggiornate.

---

## Extra opzionale
Se vuoi fornire tutto il contesto storico completo alla chat nuova, aggiungi anche il contenuto di Trascrizione_chat_completa.md dopo questo prompt.
