# Descriptive

1. Carico la serie temporale adeguandola allo standard utilizzato da pandas. Tolgo il separatore delle migliaia e uso il punto come separatore decimale.

2. Dopo aver ottenuto la Serie temporale univariata in formato Pandas, passo al core di questa analisi:
- Freq Distribution: crea range numerici di valori sui quali basare il calcolo della distribuzione.
- Media, Moda e Mediana
- Misure di dispersione: Range, varianza, deviazione standard, coefficiente di variazione e IQR (Range interquartile)
- Outlier utilizzando il metodo di intervallo interquartile
- Trend validation (da approfondire)
- Local Outliers
- Plot dei risultati

3. Funzione che esegue tutto quello elencato sopra.

# Preprocessing

Qui per eseguire il preprocessing sono state create 2 classi, con l'obiettivo di...

## auto_config
Serve a confrontare i risultati del preprocessing e selezionare automaticamente la trasformazione più adeguata.

1. Definiamo varie trasformazioni (l'idea è che ogni tipologia di analisi possa avere la trasformazione più adeguata, questo implica che saranno diverse tra modello statistico, ML e Reti neurali).

2. Backtest (profilo statistical): per ogni trasformazione candidata viene eseguito un mini backtest SARIMA su train/validation. Le previsioni vengono riportate anche in scala originale per misurare RMSE e bias (MBE), così da penalizzare configurazioni che introducono drift dopo l'inversione delle trasformazioni.

3. Recupera i candidati in base al profilo (`statistical`, `ml`, `neural`) e li valuta con `TimeSeriesPreprocessor.evaluate_candidates`.

4. Arricchisce la valutazione con metriche aggiuntive di robustezza:
- `abs_mbe_zero_val` (bias assoluto su validation con baseline zero-forecast nella scala trasformata)
- `abs_mbe_zero_val_orig` (stessa logica ma in scala originale, quando applicabile)
- per `statistical`, ranking da backtest e controllo `drift_guard` per escludere candidati troppo instabili.

5. Esegue il ranking finale con regole diverse per profilo:
- `statistical`: priorità a stabilità/backtest, stazionarietà e parsimonia della differenziazione
- `ml`: priorità a stazionarietà, bassa differenziazione e uso sensato dello scaling
- `neural`: priorità a scaling + stazionarietà, mantenendo trasformazioni non eccessive.

6. Seleziona la trasformazione vincente, costruisce la `PreprocessingConfig` finale, esegue `preprocess()` e salva la configurazione scelta per garantire riproducibilità del run.

## Time_series_preprocessor

Questa classe esegue il preprocessing vero e proprio sulla serie temporale, in modo configurabile e leakage-safe.

1. Valida la serie in input: converte in numerico, rimuove valori non validi, ordina temporalmente e garantisce una base pulita per le trasformazioni.

2. Applica trasformazioni deterministiche nell'ordine:
- `log1p` (se attivo)
- trasformazione di potenza con segno (se configurata)
- differenziazione (`diff_order`) per rendere la serie più stazionaria.

3. Esegue lo split in `train/val/test` secondo le proporzioni configurate (`SplitConfig`).

4. Fitta lo scaler solo sul train (`none`, `standard`, `minmax`) e applica la stessa trasformazione a validation e test, evitando data leakage.

5. Esegue i test di stazionarietà su ogni split (se sufficientemente lungo):
- ADF (stazionaria se `p < 0.05`)
- KPSS (stazionaria se `p >= 0.05`)
- opzionalmente Shapiro-Wilk per un controllo di normalità.

6. Calcola gli outlier locali sulle variazioni YoY con approccio robusto (rolling median + MAD, con fallback su deviazione standard) e marca i punti anomali con una soglia configurabile.

7. Restituisce un output strutturato con:
- configurazione usata
- serie trasformata
- split scalati
- tabella riassuntiva degli split
- risultati dei test
- tabella outlier locali.

8. Genera anche i plot di preprocessing per analisi e reportistica:
- raw vs transformed
- vista train/val/test
- ACF/PACF sul train
- outlier locali YoY.

# Statistical

## SARIMA

Questa parte implementa la ricerca del miglior modello SARIMA su split fissi train/validation/test.

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

## Model_config

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

## statistical_runner

Questa classe orchestra l'intero step statistico e restituisce un output unico pronto per salvataggio e confronto.

1. Inizializza runner e split validati (`train`, `validation`, `test`) mantenendo anche il blocco `train_validation`.

2. Lancia la grid search SARIMA attraverso `SarimaRunner.fit_sarima_grid()`.

3. Genera:
- forecast validation dal best model
- refit finale su train+validation
- forecast test dal modello rifittato.

4. Costruisce le tabelle finali:
- summary metriche
- winner e parametri del winner
- diagnostica residui
- forecast table completa (validation + test).

5. Espone anche `save_plots(...)` per delegare la produzione dei grafici statistici.

## evaluation

Qui sono definite le funzioni di valutazione e selezione finale del modello.

1. `build_summary_table(...)` crea la tabella centrale con metriche validation/test e indicatori informativi (`AIC`, `AICc`).

2. `build_forecast_table(...)` unifica in un unico DataFrame actual e previsioni su validation e test.

3. `build_residuals_table(...)` produce la diagnostica dei residui con test Ljung-Box (media, deviazione, p-value).

4. `select_winner(...)` seleziona il vincitore usando prima metriche validation in scala originale (se disponibili), altrimenti fallback su scala trasformata.

## plotting

Il modulo plotting salva i grafici principali dello step statistico.

1. Forecast comparison su serie trasformata:
- actual validation/test
- previsione SARIMA su validation e test.

2. Diagnostica residui SARIMA:
- ACF residui
- PACF residui.

3. Forecast in scala originale (quando inversione disponibile):
- ricostruzione con log1p/differencing
- confronto visivo con serie originale.

4. Tutti i plot vengono salvati con naming consistente e suffisso opzionale (es. baseline).

## run_baseline

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

5. Stampa a terminale il completamento run e i path degli artifact generati.

# ML

## model_config

Questo modulo centralizza configurazione e metriche comuni per lo Step 4 (ML non neurale).

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
- serie originale opzionale per confronto transformed vs original scale.

## features

Qui viene costruito il dataset supervisionato lag-based e viene fatta la selezione delle feature.

1. `build_lagged_dataset(...)` crea matrici `X/y` per train, validation e test usando una finestra `lookback`:
- `lag_1` e il valore piu recente
- validation/test possono usare la storia precedente (setup realistico per forecasting).

2. `select_features(...)` supporta tre strategie:
- `none`: usa tutti i lag
- `rfe`: Recursive Feature Elimination con albero decisionale
- `importance`: ranking da feature importance con RandomForest.

3. `build_model_feature_matrix(...)` proietta i tre split solo sulle feature selezionate, mantenendo consistenza tra train e inferenza.

4. `last_window_from_series(...)` estrae l'ultima finestra laggata pronta per forecast ricorsivo.

## plotting

Questo modulo salva i plot finali dello Step 4.

1. Grafico in scala trasformata:
- actual su validation e test
- forecast di tutti i modelli selezionati
- separazione visiva validation/test.

2. Grafico in scala originale (quando inversione disponibile):
- ricostruzione da output trasformato con `log1p` + differencing
- confronto visuale tra serie originale e forecast val/test per ogni modello.

3. Salva i file in `Results/<Serie>/plots/ml/` con naming consistente e suffisso opzionale (es. `baseline`).

## runner

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

3. Esegue fit del candidato su train e forecast ricorsivo su validation.

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

8. Se `overfitting_lambda > 0`, applica una penalizzazione al gap validation-test nel `composite_score`, rendendo la scelta finale piu robusta.

## run_baseline

Script operativo end-to-end per la baseline ML non neurale.

1. Carica la serie target da `config.py`.

2. Esegue preprocessing automatico con profilo `ml`.

3. Costruisce una `MLStepConfig` baseline (griglia controllata) e lancia `MLModelRunner`.

4. Salva output principali in `Results/<Serie>/...`:
- metriche preprocessing e ML (`grid`, `summary`, `forecasts`, `feature_selection`)
- artifact (`winner_params`, `config`, preprocessing config selezionata)
- split preprocessati per tracciabilita.

5. Salva i grafici ML tramite `save_ml_plots(...)` e stampa a terminale winner e path degli artifact prodotti.

# Neural

## model_config

Questo modulo centralizza configurazione, metriche e inversione delle trasformazioni per lo Step 5 neurale.

1. Definisce `NeuralStepConfig`, con:
- modelli candidati (`mlp`, `lstm`)
- lookback candidati
- iperparametri di training (batch size, learning rate, weight decay)
- parametri specifici di architettura (MLP: hidden/activation/dropout, LSTM: hidden/layers/dropout)
- controlli training (`max_epochs`, `patience`, `seed`, `device`).

2. Fornisce utility di riproducibilita e device management:
- `seed_everything(...)` per Python/NumPy/Torch
- `resolve_torch_device(...)` con fallback esplicito a CPU.

3. Implementa metriche condivise:
- `RMSE`, `MAE`, `MAPE`
- `MBE` e `ABS_MBE`
- versioni allineate su indice e filtraggio NaN.

4. Gestisce inversione del preprocessing per metriche in scala originale:
- inversione eventuale scaling (`none`, `standard`, `minmax`) rifittando lo scaler solo sul train
- inversione `log1p` e differenziazione (`diff_order` 0/1/2)
- calcolo metriche original-scale quando il contesto e disponibile.

5. Espone configurazioni pronte all'uso:
- `build_compact_neural_config()` per baseline
- `build_extended_neural_config()` per ricerca piu ampia.

## features

Qui viene creato il dataset supervisionato a finestre mobili per il training neurale.

1. `build_training_windows(...)` costruisce coppie `X/y` one-step su un singolo segmento.

2. `build_segment_windows(...)` crea finestre per target in segmenti successivi (validation/test), mantenendo la cronologia reale.

3. `build_windowed_splits(...)` prepara in modo coerente:
- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`
- seed ricorsivi (`train_seed`, `train_val_seed`) usati nella previsione multi-step.

## models

Questo modulo definisce le architetture torch utilizzate nello step neurale.

1. `MLPForecaster`:
- rete feedforward many-to-one
- input: vettore laggato
- blocco lineare + attivazione (`ReLU`/`Tanh`) + dropout + output lineare.

2. `LSTMForecaster`:
- LSTM many-to-one su sequenza temporale
- supporto a piu layer
- dropout effettivo solo quando `num_layers > 1`
- testa lineare finale sullo stato nascosto ultimo timestep.

## plotting

Il modulo plotting salva i grafici finali del confronto neurale.

1. Grafico in scala preprocessata:
- actual su validation e test
- previsioni val/test per ogni modello neurale
- separatore visivo tra blocchi temporali.

2. Grafico in scala originale (se inversione disponibile):
- ricostruzione forecast tramite inversione preprocessing
- confronto diretto con la serie originale.

3. Salva i file in `Results/<Serie>/plots/neural/` con naming consistente e suffisso opzionale (es. `baseline`).

## runner

`NeuralModelRunner` e l'orchestratore dello Step 5.

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

## run_baseline

Script operativo end-to-end per la baseline neurale.

1. Carica la serie target da `config.py`.

2. Esegue preprocessing automatico con profilo `neural`.

3. Costruisce configurazione compatta (`build_compact_neural_config`) e lancia `NeuralModelRunner`.

4. Salva output in `Results/<Serie>/...`:
- metriche preprocessing e neural (`grid`, `summary`, `forecasts`)
- artifact (`winner_params`, `config`, selected preprocessing config)
- split preprocessati per tracciabilita.

5. Salva i plot neurali con `save_neural_plots(...)` e stampa winner e path artifact a terminale.

# Evaluation

## comparison

Questo modulo confronta in modo unificato i risultati delle tre famiglie (statistical, ml, neural).

1. Aggrega le summary dei tre step in una tabella unica `all_models`.

2. Costruisce una tabella `family_winners` contenente il miglior modello per ciascuna famiglia.

3. Esegue il ranking globale con priorita:
- `rank_rmse_val_global` (usa metriche in scala originale quando disponibili)
- `rank_abs_mbe_val_global` come tie-break.

4. Ricostruisce una tabella `winner_forecasts` in scala originale con:
- actual
- predizione statistical winner
- predizione ml winner
- predizione neural winner
- errori per famiglia.

5. Espone il `global_winner` finale (famiglia, modello e rank).

Risultati correnti salvati:
- ConsumptionTotal: global winner = neural::mlp
- ProductionTotal: global winner = statistical::sarima.

## inferential

La valutazione inferenziale usa il test Diebold-Mariano pairwise sui family winners.

1. Confronti effettuati:
- statistical vs ml
- statistical vs neural
- ml vs neural.

2. Loss function usata: errore quadratico (`power=2`), orizzonte `h=1`.

3. Applica correzione Harvey-Leybourne-Newbold alla statistica DM e restituisce:
- `dm_stat`
- `p_value`
- `mean_loss_diff`
- interpretazione testuale (`*_better`, `no_significant_difference`, `insufficient_data`).

Esito sintetico corrente:
- ConsumptionTotal: ML e Neural migliori di Statistical; ML vs Neural non significativo.
- ProductionTotal: differenza significativa a favore ML su Statistical; gli altri confronti non significativi.

## prescriptive

Questo modulo trasforma le forecast del global winner in indicazioni operative scenario-based.

1. Lavora sul solo split test.

2. Per ogni timestamp calcola:
- valore previsto
- osservazione precedente
- variazione attesa assoluta e percentuale
- margine di incertezza (da RMSE validation del winner)
- banda scenario (`scenario_low`, `scenario_high`)
- `uncertainty_ratio`.

3. Genera una raccomandazione combinando variazione e incertezza:
- `increase`, `decrease`, `stable`
- prefisso di confidenza (`low`, `moderate`, `high uncertainty`).

Esito sintetico corrente:
- ConsumptionTotal: prevalenza segnali `low_uncertainty_increase` sul finale dell'orizzonte test.
- ProductionTotal: prevalenza segnali `low_uncertainty_decrease` nella parte centrale del test.

## pipeline_orchestration

L'orchestrazione end-to-end e gestita da due entrypoint.

1. `main.py`:
- pipeline completa sulla serie di default
- esegue descrittiva, preprocessing, statistical, ml, neural, comparison, inferential, prescriptive
- salva artifact strutturati in `Results/<Serie>/metrics|plots|artifacts/...`.

2. `main_consumption.py`:
- pipeline dedicata a `consumption_total`
- mantiene lo stesso formato output ma usa configurazioni piu conservative per serie corta (soprattutto in ML e Neural)
- include gli stessi step finali di confronto e valutazione.
