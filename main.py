# Main del progetto: esegue l'intera pipeline end-to-end, orchestrando i vari step e salvando i risultati in modo organizzato.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
	DEFAULT_SERIES_KEY,
	get_processed_root,
	get_results_root,
	get_results_subdir,
	get_series_config,
	get_series_output_name,
)

#Import del modulo di valutazione, che contiene funzioni per confrontare le performance dei modelli e generare tabelle di confronto e scenari prescriptivi.
from Project.evaluation import (
	build_cross_family_comparison, # Funzione per costruire una tabella di confronto tra le performance dei modelli statistici, ML e neurali.
	build_diebold_mariano_table,   # Funzione per costruire una tabella con i risultati del test di Diebold-Mariano, che confronta le previsioni dei modelli e valuta se le differenze sono statisticamente significative.
	build_prescriptive_table,      # Funzione per costruire una tabella con scenari prescriptivi basati sulle previsioni dei modelli.
)

#Import dei moduli di preprocessing e modellazione, che contengono classi e funzioni per eseguire il preprocessing dei dati, addestrare i modelli statistici, ML e neurali,
# e salvare i risultati.
from Project.preprocessing import (
	PreprocessingConfig, 					# Classe di configurazione per il preprocessing, che definisce le trasformazioni da applicare ai dati e i test da eseguire.
	TimeSeriesPreprocessor,					# Classe principale per eseguire il preprocessing dei dati, che applica le trasformazioni e i test definiti nella configurazione.
	prepare_preprocessing_for_profile, 		# Funzione per preparare il preprocessing in base al profilo del modello (statistico, ML o neurale). Esegue e restituisce i risultati e la configurazione selezionata.
	save_selected_preprocessing_config,		# Funzione per salvare la configurazione di preprocessing selezionata in un file JSON.
)

#Import dei moduli di modellazione, che contengono classi e funzioni per addestrare i modelli statistici, ML e neurali, eseguire le previsioni e salvare i risultati.
from Project.models.statistical import (
	StatisticalModelRunner,  			# Classe per eseguire l'addestramento e la previsione dei modelli statistici.
	StatisticalStepConfig,   			# Classe di configurazione per i modelli statistici, che definisce i parametri e le impostazioni del modello.
	infer_seasonal_period_from_index,   # Funzione per inferire il periodo stagionale dai dati temporali.
)
from Project.models.ml import (
	MLModelRunner,  # Classe per eseguire l'addestramento e la previsione dei modelli di machine learning.
	MLStepConfig,   # Classe di configurazione per i modelli di machine learning, che definisce i parametri e le impostazioni del modello.
	save_ml_plots,  # Funzione per salvare i grafici dei modelli di machine learning.

)
from Project.models.neural import (
	NeuralModelRunner,  			# Classe per eseguire l'addestramento e la previsione dei modelli neurali.
	build_compact_neural_config,    # Funzione per costruire una configurazione compatta per i modelli neurali.
	save_neural_plots,  			# Funzione per salvare i grafici dei modelli neurali.
)

#Import del modulo di analisi descrittiva, che contiene funzioni per eseguire l'analisi esplorativa dei dati, generare statistiche descrittive e visualizzazioni.
from Project.preprocessing.descriptive_analysis import (
	DescriptivePaths,			# Classe per definire i percorsi dei file di input e output dell'analisi descrittiva.
	load_target_series,			# Funzione per caricare le serie temporali target.
	run_descriptive_analysis,	# Funzione per eseguire l'analisi descrittiva sui dati.
)

# Definizione della chiave di default della serie temporale target.
TARGET_SERIES_KEY = DEFAULT_SERIES_KEY

# Funzione per salvare gli output del preprocessing. I risultati vengono salvati in modo organizzato nelle cartelle di metrics, plots e artifacts.
def _save_preprocessing_outputs(
	series_key: str,
	preproc: TimeSeriesPreprocessor,
	preproc_output: dict,
	candidate_df: pd.DataFrame,
	selected_cfg: PreprocessingConfig,
) -> dict[str, Path]:

	metrics_dir = get_results_subdir(series_key, "metrics", "preprocessing")
	processed_dir = get_processed_root(series_key)
	artifacts_dir = get_results_subdir(series_key, "artifacts", "preprocessing")
	preproc_plots_dir = get_results_subdir(series_key, "plots", "preprocessing")

	metrics_dir.mkdir(parents=True, exist_ok=True)
	processed_dir.mkdir(parents=True, exist_ok=True)
	artifacts_dir.mkdir(parents=True, exist_ok=True)
	preproc_plots_dir.mkdir(parents=True, exist_ok=True)

	# Costruzione del DataFrame dei test statistici a partire dal dizionario di output del preprocessing.
	tests_rows = []
	# Il dizionario preproc_output["tests"] contiene i risultati dei test statistici eseguiti sui dati (ADF, KPSS, Shapiro-Wilk). 
  	# Per ogni split (train, val, test), estraiamo i risultati e li organizziamo in un formato tabellare.
	for split_name, test_dict in preproc_output["tests"].items():
		row = {
			"split": split_name, 						# Indica a quale split dei dati si riferiscono i risultati del test (train, val o test).
			"n": test_dict.get("n"), 					# Numero di osservazioni nello split, importante per interpretare i risultati dei test statistici.
			"adf_stat": test_dict.get("adf_stat"), 		# Statistica del test ADF (Augmented Dickey-Fuller) per verificare la stazionarietà della serie temporale.
			"adf_pvalue": test_dict.get("adf_pvalue"), 	# Valore p del test ADF.
			"adf_stationary_at_05": test_dict.get("adf_stationary_at_05"), # Indica se la serie è stazionaria al livello di significatività 0.05 secondo il test ADF.
			"kpss_stat": test_dict.get("kpss_stat"), 	# Statistica del test KPSS (Kwiatkowski-Phillips-Schmidt-Shin) per verificare la stazionarietà della serie temporale.
			"kpss_pvalue": test_dict.get("kpss_pvalue"),# Valore p del test KPSS.
			"kpss_stationary_at_05": test_dict.get("kpss_stationary_at_05"), # Indica se la serie è stazionaria al livello di significatività 0.05 secondo il test KPSS.
			"shapiro_stat": test_dict.get("shapiro_stat"), # Statistica del test Shapiro-Wilk per verificare la normalità della serie temporale.
			"shapiro_pvalue": test_dict.get("shapiro_pvalue"), # Valore p del test Shapiro-Wilk.
			"shapiro_normal_at_05": test_dict.get("shapiro_normal_at_05"), # Indica se la serie è normale al livello di significatività 0.05 secondo il test Shapiro-Wilk.
			"kpss_note": test_dict.get("kpss_note", ""), # Note aggiuntive sul test KPSS, se presenti.
		}
		tests_rows.append(row)
	tests_df = pd.DataFrame(tests_rows) # Creazione di un DataFrame a partire dalla lista di righe, che contiene i risultati dei test statistici per ogni split dei dati.

	output_paths = {
		"preproc_split_summary": metrics_dir / "split_summary.csv",
		"preproc_tests": metrics_dir / "tests.csv",
		"preproc_local_outliers": metrics_dir / "local_outliers.csv",
		"preproc_candidate_tests": metrics_dir / "candidate_tests.csv",
		"preproc_candidate_backtest": metrics_dir / "candidate_backtest.csv",
		"preproc_selected_config": artifacts_dir / "selected_config.json",
		"preproc_train": processed_dir / "preprocessed_train_v1.csv",
		"preproc_val": processed_dir / "preprocessed_val_v1.csv",
		"preproc_test": processed_dir / "preprocessed_test_v1.csv",
	}

	preproc_output["split_summary"].to_csv(output_paths["preproc_split_summary"], index=False)
	tests_df.to_csv(output_paths["preproc_tests"], index=False)
	preproc_output["local_outliers"].to_csv(output_paths["preproc_local_outliers"], index=False)
	candidate_df.to_csv(output_paths["preproc_candidate_tests"], index=False)

	# La tabella di backtest dei candidati potrebbe non essere sempre disponibile (dipende dalle configurazioni e dai test eseguiti), quindi selezioniamo solo le colonne effettivamente presenti.
	backtest_cols = [
		"use_log1p",
		"power_exponent",
		"diff_order",
		"scale_method",
		"best_order",
		"best_seasonal_order",
		"aicc_best",
		"rmse_val_backtest",
		"mae_val_backtest",
		"mbe_val_backtest",
		"abs_mbe_val_backtest",
		"rmse_val_orig_backtest",
		"mae_val_orig_backtest",
		"mbe_val_orig_backtest",
		"abs_mbe_val_orig_backtest",
		"has_orig_backtest",
		"rank_rmse_backtest",
		"rank_abs_mbe_backtest",
		"score_backtest",
		"rank_backtest",
		"backtest_lambda",
		"drift_guard_threshold",
		"drift_guard_excluded",
	]
	available_backtest_cols = [c for c in backtest_cols if c in candidate_df.columns]
	if available_backtest_cols:
		candidate_df[available_backtest_cols].to_csv(output_paths["preproc_candidate_backtest"], index=False)
	else:
		pd.DataFrame().to_csv(output_paths["preproc_candidate_backtest"], index=False)

	# Salvataggio della configurazione di preprocessing selezionata in un file JSON, per documentare le trasformazioni e i test applicati ai dati prima di addestrare i modelli.
	save_selected_preprocessing_config(selected_cfg, output_paths["preproc_selected_config"])

    # Salvataggio dei dati preprocessed per train, validation e test.
	for split_name in ("train", "val", "test"):
		split_series = preproc_output["splits"][split_name]
		split_series.rename("value").reset_index().to_csv(output_paths[f"preproc_{split_name}"], index=False)
	# Salvataggio dei grafici di preprocessing, che possono includere visualizzazioni delle serie temporali, autocorrelazioni, distribuzioni e altri grafici utili per comprendere le trasformazioni applicate ai dati.
	plot_paths = preproc.save_preprocessing_plots(preproc_output, preproc_plots_dir)
	output_paths.update(plot_paths)

	return output_paths

# Le funzioni per salvare gli output dei vari step (statistico, ML, neurale, confronto e valutazione inferenziale/prescrittiva) seguono una struttura simile: creano le cartelle necessarie, definiscono i percorsi di output, salvano i DataFrame e i file JSON nei percorsi definiti e restituiscono un dizionario con i nomi degli output e i rispettivi percorsi.
def _save_statistical_outputs(series_key: str, stat_output: dict) -> dict[str, Path]:
	"""Persist Step 3 statistical artifacts to metrics/plots/artifacts folders."""

	metrics_dir = get_results_subdir(series_key, "metrics", "statistical")
	plots_dir = get_results_subdir(series_key, "plots", "statistical")
	artifacts_dir = get_results_subdir(series_key, "artifacts", "statistical")

	metrics_dir.mkdir(parents=True, exist_ok=True)
	plots_dir.mkdir(parents=True, exist_ok=True)
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	output_paths = {
		"stat_sarima_grid": metrics_dir / "sarima_grid.csv",
		"stat_summary": metrics_dir / "summary.csv",
		"stat_residual_diagnostics": metrics_dir / "residual_diagnostics.csv",
		"stat_forecasts": metrics_dir / "forecasts.csv",
		"stat_winner_params": artifacts_dir / "winner_params.json",
	}

	stat_output["sarima_grid"].to_csv(output_paths["stat_sarima_grid"], index=False)
	stat_output["summary"].to_csv(output_paths["stat_summary"], index=False)
	stat_output["residual_diagnostics"].to_csv(output_paths["stat_residual_diagnostics"], index=False)
	stat_output["forecast_table"].to_csv(output_paths["stat_forecasts"], index=False)

	pd.Series(stat_output["winner_params"]).to_json(output_paths["stat_winner_params"], indent=2)

	plot_paths = StatisticalModelRunner.save_plots(stat_output, plots_dir)
	output_paths.update(plot_paths)

	return output_paths


def _save_ml_outputs(series_key: str, ml_output: dict) -> dict[str, Path]:
	"""Persist Step 4 ML artifacts to metrics/plots/artifacts folders."""

	metrics_dir = get_results_subdir(series_key, "metrics", "ml")
	plots_dir = get_results_subdir(series_key, "plots", "ml")
	artifacts_dir = get_results_subdir(series_key, "artifacts", "ml")

	metrics_dir.mkdir(parents=True, exist_ok=True)
	plots_dir.mkdir(parents=True, exist_ok=True)
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	output_paths = {
		"ml_grid": metrics_dir / "grid.csv",
		"ml_summary": metrics_dir / "summary.csv",
		"ml_forecasts": metrics_dir / "forecasts.csv",
		"ml_feature_selection": metrics_dir / "feature_selection.csv",
		"ml_winner_params": artifacts_dir / "winner_params.json",
		"ml_config": artifacts_dir / "config.json",
	}

	ml_output["grid"].to_csv(output_paths["ml_grid"], index=False)
	ml_output["summary"].to_csv(output_paths["ml_summary"], index=False)
	ml_output["forecast_table"].to_csv(output_paths["ml_forecasts"], index=False)
	ml_output["feature_selection_report"].to_csv(output_paths["ml_feature_selection"], index=False)

	pd.Series(ml_output["winner_params"]).to_json(output_paths["ml_winner_params"], indent=2)
	pd.Series(ml_output["config"]).to_json(output_paths["ml_config"], indent=2)

	plot_paths = save_ml_plots(ml_output, plots_dir)
	output_paths.update(plot_paths)

	return output_paths


def _save_neural_outputs(series_key: str, neural_output: dict) -> dict[str, Path]:
	"""Persist Step 5 neural artifacts to metrics/plots/artifacts folders."""

	metrics_dir = get_results_subdir(series_key, "metrics", "neural")
	plots_dir = get_results_subdir(series_key, "plots", "neural")
	artifacts_dir = get_results_subdir(series_key, "artifacts", "neural")

	metrics_dir.mkdir(parents=True, exist_ok=True)
	plots_dir.mkdir(parents=True, exist_ok=True)
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	output_paths = {
		"neural_grid": metrics_dir / "grid.csv",
		"neural_summary": metrics_dir / "summary.csv",
		"neural_forecasts": metrics_dir / "forecasts.csv",
		"neural_winner_params": artifacts_dir / "winner_params.json",
		"neural_config": artifacts_dir / "config.json",
	}

	neural_output["grid"].to_csv(output_paths["neural_grid"], index=False)
	neural_output["summary"].to_csv(output_paths["neural_summary"], index=False)
	neural_output["forecast_table"].to_csv(output_paths["neural_forecasts"], index=False)

	pd.Series(neural_output["winner_params"]).to_json(output_paths["neural_winner_params"], indent=2)
	pd.Series(neural_output["config"]).to_json(output_paths["neural_config"], indent=2)

	plot_paths = save_neural_plots(neural_output, plots_dir)
	output_paths.update(plot_paths)

	return output_paths

# La funzione per salvare i risultati del confronto tra le famiglie di modelli (statistico, ML, neurale) organizza i risultati in cartelle separate per metrics e artifacts, salva i DataFrame e i file JSON nei percorsi definiti e restituisce un dizionario con i nomi degli output e i rispettivi percorsi.
def _save_comparison_outputs(series_key: str, comparison_output: dict) -> dict[str, Path]:
	"""Persist cross-family comparison artifacts."""

	metrics_dir = get_results_subdir(series_key, "metrics", "comparison")
	artifacts_dir = get_results_subdir(series_key, "artifacts", "comparison")

	metrics_dir.mkdir(parents=True, exist_ok=True)
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	output_paths = {
		"comparison_all_models": metrics_dir / "all_models.csv",
		"comparison_family_winners": metrics_dir / "family_winners.csv",
		"comparison_winner_forecasts": metrics_dir / "winner_forecasts.csv",
		"comparison_global_winner": artifacts_dir / "global_winner.json",
	}

	comparison_output["all_models"].to_csv(output_paths["comparison_all_models"], index=False)
	comparison_output["family_winners"].to_csv(output_paths["comparison_family_winners"], index=False)
	comparison_output["winner_forecasts"].to_csv(output_paths["comparison_winner_forecasts"], index=False)
	pd.Series(comparison_output["global_winner"]).to_json(output_paths["comparison_global_winner"], indent=2)

	return output_paths

# La funzione per salvare i risultati della valutazione inferenziale (test di Diebold-Mariano) organizza i risultati in una cartella metrics/inferential, salva il DataFrame con i risultati del test e restituisce un dizionario con i nomi degli output e i rispettivi percorsi.
def _save_inferential_outputs(series_key: str, inferential_df: pd.DataFrame) -> dict[str, Path]:
	"""Persist inferential comparison artifacts."""

	metrics_dir = get_results_subdir(series_key, "metrics", "inferential")
	metrics_dir.mkdir(parents=True, exist_ok=True)

	output_paths = {
		"inferential_diebold_mariano": metrics_dir / "diebold_mariano.csv",
	}

	inferential_df.to_csv(output_paths["inferential_diebold_mariano"], index=False)
	return output_paths

# La funzione per salvare i risultati della valutazione prescriptive organizza i risultati in una cartella metrics/prescriptive, salva il DataFrame con gli scenari prescriptivi e restituisce un dizionario con i nomi degli output e i rispettivi percorsi.
def _save_prescriptive_outputs(series_key: str, prescriptive_df: pd.DataFrame) -> dict[str, Path]:
	"""Persist prescriptive analytics artifacts."""

	metrics_dir = get_results_subdir(series_key, "metrics", "prescriptive")
	metrics_dir.mkdir(parents=True, exist_ok=True)

	output_paths = {
		"prescriptive_scenarios": metrics_dir / "scenarios.csv",
	}

	prescriptive_df.to_csv(output_paths["prescriptive_scenarios"], index=False)
	return output_paths

# La funzione main() esegue l'intera pipeline end-to-end, orchestrando i vari step (analisi descrittiva, preprocessing, modellazione statistica, ML e neurale, confronto e valutazione)
# e salvando i risultati in modo organizzato.
def main(series_key: str = TARGET_SERIES_KEY) -> None:
	"""Run the currently implemented pipeline steps end-to-end."""

	# Definizione della serie attiva e dei percorsi dedicati ai suoi output.
	series_cfg = get_series_config(series_key)
	series_output_name = get_series_output_name(series_key)
	dataset_path = series_cfg.dataset_path
	results_root = get_results_root(series_key)

	print(f"Running pipeline for series: {series_key}")
	print(f"Series output name: {series_output_name}")
	print(f"Series output root: {results_root}")

	# Step 1 - descriptive analysis.
	desc_paths = DescriptivePaths(
		dataset_path=dataset_path,
		results_metrics_dir=get_results_subdir(series_key, "metrics", "descriptive"),
		results_plots_dir=get_results_subdir(series_key, "plots", "descriptive"),
	)
 	# La funzione run_descriptive_analysis esegue l'analisi descrittiva sui dati, 
  	# generando statistiche descrittive, visualizzazioni e identificando eventuali pattern o anomalie nella serie temporale target. 
   	# I risultati vengono salvati nei percorsi definiti in desc_paths e restituiti in un dizionario con i nomi degli output e i rispettivi percorsi.
	descriptive_outputs = run_descriptive_analysis(desc_paths, target=series_key)

	print("Descriptive analysis completed.")
 	#Stampa i percorsi dei risultati dell'analisi descrittiva.
	for name, file_path in descriptive_outputs.items():
		print(f"- {name}: {file_path}")

	# Step 2 - preprocessing based on descriptive conclusions.
	series = load_target_series(dataset_path, target=series_key)

	# La funzione prepare_preprocessing_for_profile esegue il preprocessing dei dati in base al profilo del modello (statistico, ML o neurale).
	preproc, preproc_output, candidate_df, selected_cfg = prepare_preprocessing_for_profile(
		series=series,
		profile="statistical",
		base_config=PreprocessingConfig(run_shapiro=True),
	)
	# Il preprocessing include l'applicazione di trasformazioni (log, differenziazione, scaling) e l'esecuzione di test statistici (ADF, KPSS, Shapiro-Wilk) per valutare la stazionarietà e la normalità dei dati.
	preproc_outputs = _save_preprocessing_outputs(
		series_key,
		preproc,
		preproc_output,
		candidate_df,
		selected_cfg,
	)

	print("Preprocessing completed.")
	for name, file_path in preproc_outputs.items():
		print(f"- {name}: {file_path}")

	# Step 3 - canonical statistical baseline (SARIMA + Holt-Winters).
	# Standalone Step 3 runners are available in Project/models/statistical/runners/.
	seasonal_period = infer_seasonal_period_from_index(preproc_output["splits"]["train"].index)# Analizza l'indice temporale dei dati di training per identificare il periodo stagionale più probabile.
	stat_cfg = StatisticalStepConfig(
		d_values=(0, 1),
		seasonal_period=seasonal_period,
	)
	# La classe StatisticalModelRunner esegue l'addestramento e la previsione dei modelli statistici utilizzando i dati preprocessed, la configurazione definita e le serie originali.
 	# Restituisce un dizionario con i risultati dell'addestramento, delle previsioni e delle metriche di valutazione.
	stat_runner = StatisticalModelRunner(
		train=preproc_output["splits"]["train"],
		validation=preproc_output["splits"]["val"],
		test=preproc_output["splits"]["test"],
		config=stat_cfg,
		original_series=series,
		use_log1p=selected_cfg.transform.use_log1p,
		diff_order=selected_cfg.transform.diff_order,
	)
	stat_output = stat_runner.run()
	stat_paths = _save_statistical_outputs(series_key, stat_output)

	print(f"Statistical Step completed. Winner: {stat_output['winner']}")
	for name, file_path in stat_paths.items():
		print(f"- {name}: {file_path}")

	# Step 4 - canonical non-neural ML baseline.
	# Standalone Step 4 runners are available in Project/models/ml/runners/.
	ml_cfg = MLStepConfig(
		lookback_values=(6, 12),
		feature_selection="importance",
		selected_feature_count=6,
		use_xgboost=False,
		dt_max_depth=(3, None),
		dt_min_samples_leaf=(1, 2),
		rf_n_estimators=(200,),
		rf_max_depth=(6, None),
		rf_min_samples_leaf=(1,),
		gbr_n_estimators=(300,),
		gbr_learning_rate=(0.05,),
		gbr_max_depth=(2, 3),
	)
	ml_runner = MLModelRunner(
		train=preproc_output["splits"]["train"],
		validation=preproc_output["splits"]["val"],
		test=preproc_output["splits"]["test"],
		config=ml_cfg,
		original_series=series,
		use_log1p=selected_cfg.transform.use_log1p,
		diff_order=selected_cfg.transform.diff_order,
	)
	ml_output = ml_runner.run()
	ml_paths = _save_ml_outputs(series_key, ml_output)

	print(f"Step 4 ML completed. Winner: {ml_output['winner']}")
	for name, file_path in ml_paths.items():
		print(f"- {name}: {file_path}")

	# Step 5 - canonical torch-based neural baseline.
	# Standalone Step 5 runners are available in Project/models/neural/runners/.
	_, neural_preproc_output, _, neural_selected_cfg = prepare_preprocessing_for_profile(
		series=series,
		profile="neural",
		base_config=PreprocessingConfig(run_shapiro=True),
	)

	neural_runner = NeuralModelRunner(
		train=neural_preproc_output["splits"]["train"],
		validation=neural_preproc_output["splits"]["val"],
		test=neural_preproc_output["splits"]["test"],
		config=build_compact_neural_config(),
		original_series=series,
		preprocessing_config=neural_selected_cfg,
	)
	neural_output = neural_runner.run()
	neural_paths = _save_neural_outputs(series_key, neural_output)

	print(f"Step 5 Neural completed. Winner: {neural_output['winner']}")
	for name, file_path in neural_paths.items():
		print(f"- {name}: {file_path}")

	# Step 6 - cross-family comparison and evaluation.
	# La funzione build_cross_family_comparison confronta le performance dei modelli statistici, ML e neurali, identificando i vincitori di ogni famiglia e il vincitore globale. 
 	# Restituisce un dizionario con i risultati del confronto.
	comparison_output = build_cross_family_comparison(stat_output, ml_output, neural_output)
	comparison_paths = _save_comparison_outputs(series_key, comparison_output)

	print(f"Cross-family comparison completed. Global winner: {comparison_output['global_winner']['family']}::{comparison_output['global_winner']['model']}")
	for name, file_path in comparison_paths.items():
		print(f"- {name}: {file_path}")

	# Step 7 - inferential evaluation and prescriptive analytics.
	# La funzione build_diebold_mariano_table costruisce una tabella con i risultati del test di Diebold-Mariano, che confronta le previsioni dei modelli e valuta se le differenze sono statisticamente significative.
	inferential_df = build_diebold_mariano_table(comparison_output["winner_forecasts"])
	inferential_paths = _save_inferential_outputs(series_key, inferential_df)

	print("Inferential statistics completed.")
	for name, file_path in inferential_paths.items():
		print(f"- {name}: {file_path}")

	# La funzione build_prescriptive_table costruisce una tabella con scenari prescriptivi basati sulle previsioni dei modelli, che possono essere utilizzati per prendere decisioni informate sulla base delle performance dei modelli.
	prescriptive_df = build_prescriptive_table(
		comparison_output["family_winners"],
		comparison_output["winner_forecasts"],
		series,
	)
	prescriptive_paths = _save_prescriptive_outputs(series_key, prescriptive_df)

	print("Prescriptive analytics completed.")
	for name, file_path in prescriptive_paths.items():
		print(f"- {name}: {file_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run the forecasting pipeline for a configured series.")
	parser.add_argument("--series", default=TARGET_SERIES_KEY, help="Series key configured in config.py")
	args = parser.parse_args()
	main(series_key=args.series)

