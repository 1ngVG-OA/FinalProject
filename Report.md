# Analisi descrittiva

1. Carico la serie temporale adeguandola allo standard utilizzato da pandas. Tolgo il separatore delle migliaia e uso il punto come separatore decimale.

2. Dopo aver ottenuto la serie temporale univariata in formato Pandas, passo al nucleo di questa analisi:
- Distribuzione di frequenza: crea intervalli numerici di valori sui quali basare il calcolo della distribuzione.
- Media, Moda e Mediana
- Misure di dispersione: Range, varianza, deviazione standard, coefficiente di variazione e IQR (Range interquartile)
- Outlier utilizzando il metodo di intervallo interquartile
- Validazione del trend (da approfondire)
- Outlier locali
- Grafici dei risultati

3. Funzione che esegue tutto quello elencato sopra.

# Pre-elaborazione

Qui, per eseguire la pre-elaborazione, sono state create 2 classi, con l'obiettivo di...

## auto_config
Serve a confrontare i risultati della pre-elaborazione e selezionare automaticamente la trasformazione più adeguata.

1. Definiamo varie trasformazioni (l'idea è che ogni tipologia di analisi possa avere la trasformazione più adeguata, questo implica che saranno diverse tra modello statistico, ML e Reti neurali).

2. Backtest (profilo `statistical`): per ogni trasformazione candidata viene eseguito un mini backtest SARIMA su train/validation. Le previsioni vengono riportate anche in scala originale per misurare RMSE e bias (MBE), così da penalizzare configurazioni che introducono drift dopo l'inversione delle trasformazioni.

3. Recupera i candidati in base al profilo (`statistical`, `ml`, `neural`) e li valuta con `TimeSeriesPreprocessor.evaluate_candidates`.

4. Arricchisce la valutazione con metriche aggiuntive di robustezza:
- `abs_mbe_zero_val` (bias assoluto su validation con baseline zero-forecast nella scala trasformata)
- `abs_mbe_zero_val_orig` (stessa logica ma in scala originale, quando applicabile)
- per `statistical`, ranking da backtest e controllo `drift_guard` per escludere candidati troppo instabili.


---

**Problema che lo score vuole risolvere**

Ogni candidato (es. log1p+diff1) trasforma la serie. Le previsioni vengono fatte nello spazio trasformato, poi **invertite** per tornare in scala originale (GWh). Il rischio è che la trasformazione intro duca **bias sistematico** durante l'inversione: la serie predetta si sposta stabilmente sopra o sotto quella reale.

---

**I due componenti**

**RMSE_orig** (errore quadratico medio in scala originale) — misura l'accuratezza della previsione: penalizza tanto gli errori grandi. È la metrica principale di qualità.

**|MBE_orig|** (valore assoluto del Mean Bias Error in scala originale) — misura il bias sistematico:
$$\text{MBE} = \frac{1}{n}\sum_{t}(\hat{y}_t - y_t)$$
Se è positivo, il modello sovrastima sempre. Se è negativo, sottostima sempre. Il valore assoluto cattura entrambi i casi.

---

**Score composito**

$$\text{score} = \text{RMSE}_{\text{orig}} + 0.5 \times |\text{MBE}_{\text{orig}}|$$

Il λ = 0.5 (da `BACKTEST_COMPOSITE_LAMBDA`) è un peso di penalizzazione: il bias vale metà rispetto all'errore puro. Nei tuoi dati:

| Config | RMSE_orig | \|MBE_orig\| | Score |
|---|---|---|---|
| log1p + diff(1) | 39.699 | 32.813 | **56.105** ← vincitore |
| log1p + diff(2) | 46.766 | 39.204 | 66.368 |

---

**Drift Guard** — il veto prima dello score

Prima ancora di calcolare lo score, viene applicato un filtro duro: se `|MBE_orig| > 2000 GWh` la configurazione viene **esclusa direttamente**, indipendentemente dall'RMSE. Questo è il motivo per cui log1p+diff(2) — pur avendo entrambi i test di stazionarietà ✓ — viene scartato prima del ranking.
---
5. Esegue il ranking finale con regole diverse per profilo:
- `statistical`: priorità a stabilità/backtest, stazionarietà e parsimonia della differenziazione
- `ml`: priorità a stazionarietà, bassa differenziazione e uso sensato dello scaling
- `neural`: priorità a scaling + stazionarietà, mantenendo trasformazioni non eccessive.

6. Seleziona la trasformazione vincente, costruisce la `PreprocessingConfig` finale, esegue `preprocess()` e salva la configurazione scelta per garantire la riproducibilità dell'esecuzione.

## Time_series_preprocessor

Questa classe esegue la pre-elaborazione vera e propria sulla serie temporale, in modo configurabile e senza leakage.

1. Valida la serie in input: converte in numerico, rimuove valori non validi, ordina temporalmente e garantisce una base pulita per le trasformazioni.

2. Applica trasformazioni deterministiche nell'ordine:
- `log1p` (se attivo)
- trasformazione di potenza con segno (se configurata)
- differenziazione (`diff_order`) per rendere la serie più stazionaria.

3. Esegue la suddivisione in `train/val/test` secondo le proporzioni configurate (`SplitConfig`).

4. Fitta lo scaler solo sul train (`none`, `standard`, `minmax`) e applica la stessa trasformazione a validation e test, evitando leakage informativo.

5. Esegue i test di stazionarietà su ogni split (se sufficientemente lungo):
- ADF (stazionaria se `p < 0.05`)
- KPSS (stazionaria se `p >= 0.05`)
- opzionalmente Shapiro-Wilk per un controllo di normalità.

6. Calcola gli outlier locali sulle variazioni YoY con approccio robusto (rolling median + MAD, con fallback su deviazione standard) e marca i punti anomali con una soglia configurabile.
Importante: non vengono rimossi o corretti valori della serie. Sono usati per analisi/visualizzazione.
In pratica servono a evidenziare shock locali (picchi/crolli) mantenendo intatti i dati per il training.

7. Restituisce un output strutturato con:
- configurazione usata
- serie trasformata
- split scalati
- tabella riassuntiva degli split
- risultati dei test
- tabella outlier locali.

8. Genera anche i grafici di pre-elaborazione per analisi e reportistica:
- serie grezza vs serie trasformata
- vista train/val/test
- ACF/PACF sul train

---

**ACF — Autocorrelation Function**

Misura la correlazione tra un valore e i suoi lag passati, inclusi gli effetti indiretti. Nel tuo grafico:
- Lag 0 = 1.0 sempre (una serie è perfettamente correlata con se stessa)
- Lag 1 ≈ +0.30 significativo (ancora l'ombra del lag precedente)
- Lag 2 ≈ −0.25 significativo
- Dal lag 3 in poi: tutto dentro la banda di confidenza (zona azzurra) → rumore

Banda azzurra = intervallo di confidenza al 95% (≈ ±2/√n). Se una barra esce dalla banda, la correlazione è statisticamente significativa.

---

**PACF — Partial Autocorrelation Function**

Misura la correlazione "netta" tra il valore al tempo t e il lag k, **rimuovendo l'effetto di tutti i lag intermedi**. Nel tuo grafico ha lo stesso pattern: significativo a lag 1 e 2, poi nulla.

---

**Come si usano per scegliere p e q in ARIMA:**

| Comportamento | Interpretazione |
|---|---|
| ACF scende lentamente (esponenziale o oscillante) | Serie non stazionaria — serve differenziare |
| ACF taglia brusco dopo lag q | suggerisce MA(q) |
| PACF taglia brusco dopo lag p | suggerisce AR(p) |
| Entrambi decadono lentamente | suggerisce ARMA(p,q) |

---

**Nel tuo caso specifico:** dopo log1p + diff(1) l'ACF taglia dopo lag 2 e anche la PACF, con entrambi quasi identici → compatibile con un **MA(1) o AR(1) semplice**, che è esattamente quello che il backtest SARIMA ha selezionato come ordine migliore `(1,0,0)`.
- outlier locali YoY.

# Modelli statistici

## SARIMA

**1) MAPE: cos'è**
MAPE = Mean Absolute Percentage Error, cioè errore medio percentuale assoluto.

Formula:
$$
\text{MAPE} = \frac{100}{n}\sum_{t=1}^{n}\left|\frac{y_t-\hat{y}_t}{y_t}\right|
$$

Interpretazione:
- 5% significa: in media sbagli del 5% rispetto al valore reale.
- Più è basso, meglio è.

Limite importante:
- Se alcuni $y_t$ sono molto piccoli (o vicini a 0), il MAPE può esplodere.
- Per questo puoi vedere valori molto alti (come 921%) anche quando visivamente il fit non sembra “catastrofico” in tutti i punti.

---

**2) Tie-break: cosa intendo**
Nel ranking dei modelli prima ordini per una metrica principale (es. RMSE).
Se due modelli sono praticamente uguali su quella metrica, serve una regola secondaria per “rompere il pareggio”: questo è il tie-break.

Nel tuo caso, semplificando:
1. criterio principale: RMSE (o RMSE in scala originale)
2. tie-break 1: $|MBE|$ (bias assoluto)
3. tie-break 2: AICc (parsimony/qualità statistica)

Quindi “tie-break” = criterio usato solo quando i modelli sono quasi pari sul criterio principale.

---

**3) Residual std: cos’è**
Residual std = deviazione standard dei residui.

Residuo:
$$
e_t = y_t - \hat{y}_t
$$

Residual std:
$$
\sigma_e = \text{std}(e_t)
$$

Interpretazione:
- Misura quanto “si sparpagliano” gli errori attorno a 0.
- Bassa residual std = errori più concentrati = modello più stabile.
- Alta residual std = errori molto variabili.

Differenza da residual mean:
- residual mean ti dice se c’è bias medio (sovra/sotto stima sistematica).
- residual std ti dice la variabilità degli errori.

---


Questa parte implementa la ricerca del miglior modello SARIMA su suddivisioni fisse train/validation/test.

1. Costruisce i candidati combinando i parametri non stagionali (`p,d,q`) e, se presente stagionalità, anche quelli stagionali (`P,D,Q,s`).

2. Per ogni candidato:
- fitta il modello su train
- produce forecast su validation
- calcola metriche (`RMSE`, `MAE`, `MAPE`, `MBE`, `ABS_MBE`)
- calcola anche metriche in scala originale quando è disponibile il contesto di inversione (log1p + differencing).

3. Applica il ranking dei candidati con priorità:
- `rank_rmse_val`
- `rank_abs_mbe_val`
- `AICc` come tie-break finale.

4. Seleziona il best candidate e poi esegue il refit su `train + validation` per ottenere il modello finale usato nella previsione test.

## Configurazione modello

Questo modulo centralizza configurazioni e utility comuni dello step statistico.

1. Definisce `StatisticalStepConfig` con:
- griglia parametri SARIMA
- opzioni di stazionarietà/invertibilità
- `maxiter`
- lag per test Ljung-Box.

2. Fornisce funzioni di validazione degli split e della serie originale, in modo da avere input consistenti e ordinati temporalmente.

3. Implementa le metriche comuni:
- `RMSE`, `MAE`, `MAPE`
- `MBE` e `ABS_MBE`
- `AICc` per campioni piccoli.

4. Gestisce il contesto per riportare le previsioni in scala originale (`build_original_scale_context`) e il calcolo delle metriche original-scale (`validation_original_metrics`).

5. Include helper per inferire il periodo stagionale dall'indice temporale (`infer_seasonal_period_from_index`).

## Runner statistico

Questa classe orchestra l'intero step statistico e restituisce un output unico pronto per salvataggio e confronto.

1. Inizializza runner e split validati (`train`, `validation`, `test`) mantenendo anche il blocco `train_validation`.

2. Lancia la grid search SARIMA attraverso `SarimaRunner.fit_sarima_grid()`.

3. Genera:
- forecast validation dal best model
- refit finale su train+validation
- forecast test dal modello rifittato.

4. Costruisce le tabelle finali:
- riepilogo metriche
- winner e parametri del winner
- diagnostica residui
- tabella forecast completa (validation + test).

5. Espone anche `save_plots(...)` per delegare la produzione dei grafici statistici.

## Valutazione

Qui sono definite le funzioni di valutazione e selezione finale del modello.

1. `build_summary_table(...)` crea la tabella centrale con metriche validation/test e indicatori informativi (`AIC`, `AICc`).

2. `build_forecast_table(...)` unifica in un unico DataFrame actual e previsioni su validation e test.

3. `build_residuals_table(...)` produce la diagnostica dei residui con test Ljung-Box (media, deviazione, p-value).

4. `select_winner(...)` seleziona il vincitore usando prima metriche validation in scala originale (se disponibili), altrimenti fallback su scala trasformata.

## Grafici

Il modulo plotting salva i grafici principali dello step statistico.

1. Confronto forecast su serie trasformata:
- actual validation/test
- previsione SARIMA su validation e test.

2. Diagnostica residui SARIMA:
- ACF residui
- PACF residui.

3. Forecast in scala originale (quando inversione disponibile):
- ricostruzione con log1p/differencing
- confronto visivo con serie originale.

4. Tutti i grafici vengono salvati con naming consistente e suffisso opzionale (es. baseline).

## Esecuzione baseline

Script operativo per eseguire end-to-end la baseline statistica.

1. Carica la serie target dal `config.py` e dal dataset associato.

2. Esegue il preprocessing automatico con profilo `statistical`.

3. Inferisce la stagionalità dall'indice, costruisce `StatisticalStepConfig` e lancia `StatisticalModelRunner`.

4. Salva in `Results/<Serie>/...`:
- metriche preprocessing e SARIMA
- split preprocessati
- summary, residual diagnostics, forecast table
- winner params
- plot finali.

5. Stampa a terminale il completamento dell'esecuzione e i path degli artifact generati.

# ML

## Configurazione modello

Questo modulo centralizza configurazione e metriche comuni per la Fase 4 (ML non neurale).

1. Definisce `MLStepConfig`, con:
- griglie dei modelli (`DecisionTree`, `RandomForest`, `GradientBoosting`, `XGBoost` opzionale)
- lookback candidati
- metodo di feature selection (`none`, `rfe`, `importance`)
- numero di feature selezionate
- parametri opzionali per robustezza su serie corte (`cv_folds`, `overfitting_lambda`).

2. Implementa metriche condivise:
- `RMSE`, `MAE`, `MAPE`
- `MBE` e `ABS_MBE`
- versioni allineate su indice e con gestione NaN.

3. Gestisce il calcolo delle metriche in scala originale quando il preprocessing include `log1p` e differenziazione (`diff_order` 0/1/2), con inversione coerente delle previsioni.

4. Fornisce utility di validazione input:
- split train/validation/test numerici e ordinati temporalmente
- serie originale opzionale per confronto tra scala trasformata e scala originale.

## features

Qui viene costruito il dataset supervisionato basato sui lag e viene fatta la selezione delle feature.

1. `build_lagged_dataset(...)` crea matrici `X/y` per train, validation e test usando una finestra `lookback`:
- `lag_1` è il valore più recente
- validation/test possono usare la storia precedente (impostazione realistica per il forecasting).

2. `select_features(...)` supporta tre strategie:
- `none`: usa tutti i lag
- `rfe`: Recursive Feature Elimination con albero decisionale
- `importance`: ranking da feature importance con RandomForest.

3. `build_model_feature_matrix(...)` proietta i tre split solo sulle feature selezionate, mantenendo coerenza tra train e inferenza.

4. `last_window_from_series(...)` estrae l'ultima finestra laggata pronta per forecast ricorsivo.

## Grafici

Questo modulo salva i grafici finali della Fase 4.

1. Grafico in scala trasformata:
- valori osservati su validation e test
- forecast di tutti i modelli selezionati
- separazione visiva validation/test.

2. Grafico in scala originale (quando inversione disponibile):
- ricostruzione da output trasformato con `log1p` + differencing
- confronto visuale tra serie originale e forecast val/test per ogni modello.

3. Salva i file in `Results/<Serie>/plots/ml/` con naming consistente e suffisso opzionale (es. `baseline`).

## Runner

`MLModelRunner` e l'orchestratore centrale dello step ML.

1. Genera la griglia dei candidati per famiglia modello e iperparametri:
- `decision_tree`
- `random_forest`
- `gradient_boosting`
- `xgboost` (solo se disponibile e abilitato).

2. Per ogni `lookback` prepara una sola volta:
- dataset laggato
- feature selection
- matrici ridotte sulle feature selezionate.

3. Esegue il fit del candidato su train e forecast ricorsivo su validation.

4. Calcola metriche validation in scala trasformata e, se possibile, anche in scala originale.

5. Ranking candidati:
- primario su `RMSE` (in scala originale se disponibile)
- tie-break su `ABS_MBE`
- opzionalmente usa walk-forward CV sul train (`cv_folds`) per serie corte.

6. Seleziona il best per ogni famiglia modello, poi rifitta su `train + validation` e produce forecast su test.

7. Costruisce gli artifact finali:
- `grid` completo
- `summary` per modello
- `forecast_table` (validation + test)
- `feature_selection_report`
- `winner` e `winner_params`.

8. Se `overfitting_lambda > 0`, applica una penalizzazione al gap validation-test nel `composite_score`, rendendo la scelta finale più robusta.

## Esecuzione baseline

Script operativo end-to-end per la baseline ML non neurale.

1. Carica la serie target da `config.py`.

2. Esegue la pre-elaborazione automatica con profilo `ml`.

3. Costruisce una `MLStepConfig` baseline (griglia controllata) e lancia `MLModelRunner`.

4. Salva gli output principali in `Results/<Serie>/...`:
- metriche preprocessing e ML (`grid`, `summary`, `forecasts`, `feature_selection`)
- artifact (`winner_params`, `config`, preprocessing config selezionata)
- split preprocessati per tracciabilità.

5. Salva i grafici ML tramite `save_ml_plots(...)` e stampa a terminale winner e path degli artifact prodotti.

# Modelli neurali

## Configurazione modello

Questo modulo centralizza configurazione, metriche e inversione delle trasformazioni per la Fase 5 neurale.

1. Definisce `NeuralStepConfig`, con:
- modelli candidati (`mlp`, `lstm`)
- lookback candidati
- iperparametri di training (batch size, learning rate, weight decay)
- parametri specifici di architettura (MLP: hidden/activation/dropout, LSTM: hidden/layers/dropout)
- controlli training (`max_epochs`, `patience`, `seed`, `device`).

2. Fornisce utility di riproducibilità e gestione del device:
- `seed_everything(...)` per Python/NumPy/Torch
- `resolve_torch_device(...)` con fallback esplicito a CPU.

3. Implementa metriche condivise:
- `RMSE`, `MAE`, `MAPE`
- `MBE` e `ABS_MBE`
- versioni allineate su indice e filtraggio NaN.

4. Gestisce l'inversione della pre-elaborazione per metriche in scala originale:
- inversione eventuale scaling (`none`, `standard`, `minmax`) rifittando lo scaler solo sul train
- inversione `log1p` e differenziazione (`diff_order` 0/1/2)
- calcolo metriche in scala originale quando il contesto è disponibile.

5. Espone configurazioni pronte all'uso:
- `build_compact_neural_config()` per baseline
- `build_extended_neural_config()` per ricerca più ampia.

## Feature

Qui viene creato il dataset supervisionato a finestre mobili per il training neurale.

1. `build_training_windows(...)` costruisce coppie `X/y` one-step su un singolo segmento.

2. `build_segment_windows(...)` crea finestre per target in segmenti successivi (validation/test), mantenendo la cronologia reale.

3. `build_windowed_splits(...)` prepara in modo coerente:
- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`
- seed ricorsivi (`train_seed`, `train_val_seed`) usati nella previsione multi-step.

## Modelli

Questo modulo definisce le architetture torch utilizzate nello step neurale.

1. `MLPForecaster`:
- rete feedforward many-to-one
- input: vettore laggato
- blocco lineare + attivazione (`ReLU`/`Tanh`) + dropout + output lineare.

2. `LSTMForecaster`:
- LSTM many-to-one su sequenza temporale
- supporto a più layer
- dropout effettivo solo quando `num_layers > 1`
- testa lineare finale sullo stato nascosto ultimo timestep.

## Grafici

Il modulo plotting salva i grafici finali del confronto neurale.

1. Grafico in scala preprocessata:
- valori osservati su validation e test
- previsioni val/test per ogni modello neurale
- separatore visivo tra blocchi temporali.

2. Grafico in scala originale (se inversione disponibile):
- ricostruzione forecast tramite inversione preprocessing
- confronto diretto con la serie originale.

3. Salva i file in `Results/<Serie>/plots/neural/` con naming consistente e suffisso opzionale (es. `baseline`).

## Runner

`NeuralModelRunner` è l'orchestratore della Fase 5.

1. Valida split e serie originale, risolve device torch e prepara il blocco `train + validation`.

2. Genera la griglia candidati combinando:
- tipo modello (`mlp` o `lstm`)
- lookback
- iperparametri di architettura e training.

3. Per ogni candidato:
- costruisce finestre supervisionate
- allena il modello con `Adam` e loss MSE
- usa early stopping su validation (`patience`) salvando `best_epoch` e `best_val_loss`.

4. Genera forecast ricorsivo su validation e calcola metriche transformed + original scale.

5. Seleziona il best candidate per famiglia modello con ranking:
- `rank_rmse_val` (prioritario)
- `rank_abs_mbe_val` (tie-break).

6. Refit finale:
- ricostruisce dataset su `train + validation`
- riallena per `best_epoch`
- produce forecast su test.

7. Costruisce output finali:
- `grid`
- `summary`
- `forecast_table`
- `winner` e `winner_params`
- serie predette per validation/test per ogni modello.

## Esecuzione baseline

Script operativo end-to-end per la baseline neurale.

1. Carica la serie target da `config.py`.

2. Esegue la pre-elaborazione automatica con profilo `neural`.

3. Costruisce configurazione compatta (`build_compact_neural_config`) e lancia `NeuralModelRunner`.

4. Salva output in `Results/<Serie>/...`:
- metriche preprocessing e neural (`grid`, `summary`, `forecasts`)
- artifact (`winner_params`, `config`, selected preprocessing config)
- split preprocessati per tracciabilità.

5. Salva i plot neurali con `save_neural_plots(...)` e stampa winner e path artifact a terminale.

# Valutazione finale

## Confronto

Questo modulo confronta in modo unificato i risultati delle tre famiglie (statistical, ml, neural).

1. Aggrega le summary dei tre step in una tabella unica `all_models`.

2. Costruisce una tabella `family_winners` contenente il miglior modello per ciascuna famiglia.

3. Esegue il ranking globale con priorità:
- `rank_rmse_val_global` (usa metriche in scala originale quando disponibili)
- `rank_abs_mbe_val_global` come tie-break.

4. Ricostruisce una tabella `winner_forecasts` in scala originale con:
- valori reali
- predizione del vincitore statistical
- predizione del vincitore ml
- predizione del vincitore neural
- errori per famiglia.

5. Espone il `global_winner` finale (famiglia, modello e rank).

Risultati correnti salvati:
- ConsumptionTotal: global winner = neural::mlp
- ProductionTotal: global winner = statistical::sarima.

## Analisi inferenziale

La valutazione inferenziale usa il test Diebold-Mariano pairwise sui family winners.

1. Confronti effettuati:
- statistical vs ml
- statistical vs neural
- ml vs neural.

2. Funzione di perdita usata: errore quadratico (`power=2`), orizzonte `h=1`.

3. Applica correzione Harvey-Leybourne-Newbold alla statistica DM e restituisce:
- `dm_stat`
- `p_value`
- `mean_loss_diff`
- interpretazione testuale (`*_better`, `no_significant_difference`, `insufficient_data`).

Esito sintetico corrente:
- ConsumptionTotal: ML e Neural migliori di Statistical; ML vs Neural non significativo.
- ProductionTotal: differenza significativa a favore ML su Statistical; gli altri confronti non significativi.

## Analisi prescrittiva

Questo modulo trasforma le forecast del global winner in indicazioni operative basate su scenari.

1. Lavora sul solo split test.

2. Per ogni timestamp calcola:
- valore previsto
- osservazione precedente
- variazione attesa assoluta e percentuale
- margine di incertezza (da RMSE validation del winner)
- banda di scenario (`scenario_low`, `scenario_high`)
- `uncertainty_ratio`.

3. Genera una raccomandazione combinando variazione e incertezza:
- `increase`, `decrease`, `stable`
- prefisso di confidenza (`low`, `moderate`, `high uncertainty`).

Esito sintetico corrente:
- ConsumptionTotal: prevalenza segnali `low_uncertainty_increase` sul finale dell'orizzonte test.
- ProductionTotal: prevalenza segnali `low_uncertainty_decrease` nella parte centrale del test.

## Orchestrazione pipeline

L'orchestrazione end-to-end è gestita da due entrypoint.

1. `main.py`:
- pipeline completa sulla serie di default
- esegue descrittiva, preprocessing, statistical, ml, neural, comparison, inferential, prescriptive
- salva artifact strutturati in `Results/<Serie>/metrics|plots|artifacts/...`.

2. `main_consumption.py`:
- pipeline dedicata a `consumption_total`
- mantiene lo stesso formato output ma usa configurazioni più conservative per serie corta (soprattutto in ML e Neural)
- include gli stessi step finali di confronto e valutazione.

# Flusso di dati 

L'analisi descrittiva viene eseguita in modo indipendente e non genera output necessari allo sviluppo dei modelli, ma serve per capire il comportamento della serie.

## Analisi Statistica

```mermaid
flowchart TD
	A[Dataset CSV grezzo] --> B[Caricamento serie target]
	B --> C[Pulizia e casting numerico]
	C --> D[Analisi descrittiva]
	D --> E[Pre-elaborazione automatica - profilo statistical]

	E --> E1[Generazione candidati trasformazione]
	E1 --> E2[Valutazione candidati + ranking]
	E2 --> E3[Selezione configurazione vincente]
	E3 --> E4[Trasformazioni deterministiche: log1p/power/diff]
	E4 --> E5[Split train/validation/test]
	E5 --> E6[Fit scaler solo su train + applicazione su val/test]
	E6 --> E7[Test statistici + outlier locali + plot]

	E7 --> S0[Branch Statistical]
	E7 --> M0[Branch ML]
	C --> N0[Branch Neural: pre-elaborazione profilo neural]

	S0 --> S1[Grid SARIMA su train]
	S1 --> S2[Forecast validation]
	S2 --> S3[Metriche val transformed + original]
	S3 --> S4[Ranking candidati]
	S4 --> S5[Refit su train + validation]
	S5 --> S6[Forecast test]
	S6 --> S7[Salvataggio: summary/forecasts/residuals/winner]

	M0 --> M1[Costruzione dataset laggato]
	M1 --> M2[Feature selection]
	M2 --> M3[Grid ML: DT/RF/GBR/XGB opzionale]
	M3 --> M4[Fit su train + forecast ricorsivo validation]
	M4 --> M5[Metriche val transformed + original]
	M5 --> M6[Ranking (RMSE + ABS_MBE, CV opzionale)]
	M6 --> M7[Refit su train + validation]
	M7 --> M8[Forecast test]
	M8 --> M9[Salvataggio: grid/summary/forecasts/features/winner]

	N0 --> N1[Pre-elaborazione profilo neural]
	N1 --> N2[Costruzione finestre mobili]
	N2 --> N3[Grid neurale: MLP/LSTM]
	N3 --> N4[Training con early stopping]
	N4 --> N5[Forecast ricorsivo validation]
	N5 --> N6[Metriche val transformed + original]
	N6 --> N7[Ranking candidati per famiglia]
	N7 --> N8[Refit su train + validation]
	N8 --> N9[Forecast test]
	N9 --> N10[Salvataggio: grid/summary/forecasts/winner]

	S7 --> C0[Confronto cross-family]
	M9 --> C0
	N10 --> C0

	C0 --> C1[all_models]
	C0 --> C2[family_winners]
	C0 --> C3[winner_forecasts in scala originale]
	C0 --> C4[global_winner]

	C4 --> I0[Analisi inferenziale]
	C3 --> I0
	I0 --> I1[Test Diebold-Mariano pairwise]
	I1 --> I2[diebold_mariano.csv]

	C2 --> P0[Analisi prescrittiva]
	C3 --> P0
	P0 --> P1[Scenari sullo split test]
	P1 --> P2[Segnali: increase/decrease/stable]
	P2 --> P3[scenarios.csv]
```

## Sequenza operativa (in breve)

1. I dati grezzi vengono caricati e convertiti in una serie temporale numerica ordinata.
2. Si esegue analisi descrittiva per comprendere distribuzione, dispersione, outlier e trend.
3. La serie passa nella pre-elaborazione: trasformazioni candidate, ranking automatico e selezione configurazione.
4. Si applicano trasformazioni deterministiche, split train/validation/test, scaling leakage-safe, test statistici e diagnostica outlier.
5. Dallo stesso input preprocessato partono i tre rami modellistici:
- Statistical: grid SARIMA, ranking, refit, forecast test.
- ML: lag features, feature selection, grid modelli, ranking, refit, forecast test.
- Neural: finestre mobili, training MLP/LSTM con early stopping, ranking, refit, forecast test.
6. I winner di famiglia vengono confrontati nel layer cross-family e si ottiene il global winner.
7. Sui forecast dei winner vengono eseguite:
- analisi inferenziale (Diebold-Mariano)
- analisi prescrittiva (scenari e raccomandazioni operative).
8. Tutti gli artifact finali vengono salvati in `Results/<Serie>/metrics|plots|artifacts/...`.
