"""Project entry point.

Implemented steps:
1) Descriptive analysis for Tavola_1.14.
2) Time-series preprocessing (split, transforms, stationarity tests, local outliers).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from Project.evaluation import (
	build_cross_family_comparison,
	build_diebold_mariano_table,
	build_prescriptive_table,
)
from Project.preprocessing import (
	PreprocessingConfig,
	TimeSeriesPreprocessor,
	prepare_preprocessing_for_profile,
	save_selected_preprocessing_config,
)
from Project.models.statistical import (
	StatisticalModelRunner,
	StatisticalStepConfig,
	infer_seasonal_period_from_index,
)
from Project.models.ml import (
	MLModelRunner,
	MLStepConfig,
	save_ml_plots,
)
from Project.models.neural import (
	NeuralModelRunner,
	build_compact_neural_config,
	save_neural_plots,
)
from Project.preprocessing.descriptive_analysis import (
	DescriptivePaths,
	load_target_series,
	run_descriptive_analysis,
)


TARGET_SERIES_KEY = "production_total"


def _save_preprocessing_outputs(
	root: Path,
	preproc: TimeSeriesPreprocessor,
	preproc_output: dict,
	candidate_df: pd.DataFrame,
	selected_cfg: PreprocessingConfig,
) -> dict[str, Path]:
	"""Persist preprocessing artifacts to metrics and processed data folders."""

	metrics_dir = root / "Results" / "metrics" / "preprocessing"
	processed_dir = root / "Datasets" / "processed"
	artifacts_dir = root / "Results" / "artifacts" / "preprocessing"
	preproc_plots_dir = root / "Results" / "plots" / "preprocessing"

	metrics_dir.mkdir(parents=True, exist_ok=True)
	processed_dir.mkdir(parents=True, exist_ok=True)
	artifacts_dir.mkdir(parents=True, exist_ok=True)
	preproc_plots_dir.mkdir(parents=True, exist_ok=True)

	tests_rows = []
	for split_name, test_dict in preproc_output["tests"].items():
		row = {
			"split": split_name,
			"n": test_dict.get("n"),
			"adf_stat": test_dict.get("adf_stat"),
			"adf_pvalue": test_dict.get("adf_pvalue"),
			"adf_stationary_at_05": test_dict.get("adf_stationary_at_05"),
			"kpss_stat": test_dict.get("kpss_stat"),
			"kpss_pvalue": test_dict.get("kpss_pvalue"),
			"kpss_stationary_at_05": test_dict.get("kpss_stationary_at_05"),
			"shapiro_stat": test_dict.get("shapiro_stat"),
			"shapiro_pvalue": test_dict.get("shapiro_pvalue"),
			"shapiro_normal_at_05": test_dict.get("shapiro_normal_at_05"),
			"kpss_note": test_dict.get("kpss_note", ""),
		}
		tests_rows.append(row)
	tests_df = pd.DataFrame(tests_rows)

	output_paths = {
		"preproc_split_summary": metrics_dir / "split_summary.csv",
		"preproc_tests": metrics_dir / "tests.csv",
		"preproc_local_outliers": metrics_dir / "local_outliers.csv",
		"preproc_candidate_tests": metrics_dir / "candidate_tests.csv",
		"preproc_selected_config": artifacts_dir / "selected_config.json",
		"preproc_train": processed_dir / "tavola_1_14_preprocessed_train_v1.csv",
		"preproc_val": processed_dir / "tavola_1_14_preprocessed_val_v1.csv",
		"preproc_test": processed_dir / "tavola_1_14_preprocessed_test_v1.csv",
	}

	preproc_output["split_summary"].to_csv(output_paths["preproc_split_summary"], index=False)
	tests_df.to_csv(output_paths["preproc_tests"], index=False)
	preproc_output["local_outliers"].to_csv(output_paths["preproc_local_outliers"], index=False)
	candidate_df.to_csv(output_paths["preproc_candidate_tests"], index=False)
	save_selected_preprocessing_config(selected_cfg, output_paths["preproc_selected_config"])

	for split_name in ("train", "val", "test"):
		split_series = preproc_output["splits"][split_name]
		split_series.rename("value").reset_index().to_csv(output_paths[f"preproc_{split_name}"], index=False)

	plot_paths = preproc.save_preprocessing_plots(preproc_output, preproc_plots_dir)
	output_paths.update(plot_paths)

	return output_paths


def _save_statistical_outputs(root: Path, stat_output: dict) -> dict[str, Path]:
	"""Persist Step 3 statistical artifacts to metrics/plots/artifacts folders."""

	metrics_dir = root / "Results" / "metrics" / "statistical"
	plots_dir = root / "Results" / "plots" / "statistical"
	artifacts_dir = root / "Results" / "artifacts" / "statistical"

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


def _save_ml_outputs(root: Path, ml_output: dict) -> dict[str, Path]:
	"""Persist Step 4 ML artifacts to metrics/plots/artifacts folders."""

	metrics_dir = root / "Results" / "metrics" / "ml"
	plots_dir = root / "Results" / "plots" / "ml"
	artifacts_dir = root / "Results" / "artifacts" / "ml"

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


def _save_neural_outputs(root: Path, neural_output: dict) -> dict[str, Path]:
	"""Persist Step 5 neural artifacts to metrics/plots/artifacts folders."""

	metrics_dir = root / "Results" / "metrics" / "neural"
	plots_dir = root / "Results" / "plots" / "neural"
	artifacts_dir = root / "Results" / "artifacts" / "neural"

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


def _save_comparison_outputs(root: Path, comparison_output: dict) -> dict[str, Path]:
	"""Persist cross-family comparison artifacts."""

	metrics_dir = root / "Results" / "metrics" / "comparison"
	artifacts_dir = root / "Results" / "artifacts" / "comparison"

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


def _save_inferential_outputs(root: Path, inferential_df: pd.DataFrame) -> dict[str, Path]:
	"""Persist inferential comparison artifacts."""

	metrics_dir = root / "Results" / "metrics" / "inferential"
	metrics_dir.mkdir(parents=True, exist_ok=True)

	output_paths = {
		"inferential_diebold_mariano": metrics_dir / "diebold_mariano.csv",
	}

	inferential_df.to_csv(output_paths["inferential_diebold_mariano"], index=False)
	return output_paths


def _save_prescriptive_outputs(root: Path, prescriptive_df: pd.DataFrame) -> dict[str, Path]:
	"""Persist prescriptive analytics artifacts."""

	metrics_dir = root / "Results" / "metrics" / "prescriptive"
	metrics_dir.mkdir(parents=True, exist_ok=True)

	output_paths = {
		"prescriptive_scenarios": metrics_dir / "scenarios.csv",
	}

	prescriptive_df.to_csv(output_paths["prescriptive_scenarios"], index=False)
	return output_paths


def main() -> None:
	"""Run the currently implemented pipeline steps end-to-end."""

	root = Path(__file__).resolve().parent
	dataset_path = root / "Datasets" / "Tavola_1.14.csv"

	# Step 1 - descriptive analysis.
	desc_paths = DescriptivePaths(
		dataset_path=dataset_path,
		results_metrics_dir=root / "Results" / "metrics" / "descriptive",
		results_plots_dir=root / "Results" / "plots" / "descriptive",
	)
	descriptive_outputs = run_descriptive_analysis(desc_paths, target=TARGET_SERIES_KEY)

	print("Descriptive analysis completed.")
	for name, file_path in descriptive_outputs.items():
		print(f"- {name}: {file_path}")

	# Step 2 - preprocessing based on descriptive conclusions.
	series = load_target_series(dataset_path, target=TARGET_SERIES_KEY)

	preproc, preproc_output, candidate_df, selected_cfg = prepare_preprocessing_for_profile(
		series=series,
		profile="statistical",
		base_config=PreprocessingConfig(run_shapiro=True),
	)

	preproc_outputs = _save_preprocessing_outputs(
		root,
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
	seasonal_period = infer_seasonal_period_from_index(preproc_output["splits"]["train"].index)
	stat_cfg = StatisticalStepConfig(
		d_values=(0, 1),
		seasonal_period=seasonal_period,
	)
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
	stat_paths = _save_statistical_outputs(root, stat_output)

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
	ml_paths = _save_ml_outputs(root, ml_output)

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
	neural_paths = _save_neural_outputs(root, neural_output)

	print(f"Step 5 Neural completed. Winner: {neural_output['winner']}")
	for name, file_path in neural_paths.items():
		print(f"- {name}: {file_path}")

	comparison_output = build_cross_family_comparison(stat_output, ml_output, neural_output)
	comparison_paths = _save_comparison_outputs(root, comparison_output)

	print(f"Cross-family comparison completed. Global winner: {comparison_output['global_winner']['family']}::{comparison_output['global_winner']['model']}")
	for name, file_path in comparison_paths.items():
		print(f"- {name}: {file_path}")

	inferential_df = build_diebold_mariano_table(comparison_output["winner_forecasts"])
	inferential_paths = _save_inferential_outputs(root, inferential_df)

	print("Inferential statistics completed.")
	for name, file_path in inferential_paths.items():
		print(f"- {name}: {file_path}")

	prescriptive_df = build_prescriptive_table(
		comparison_output["family_winners"],
		comparison_output["winner_forecasts"],
		series,
	)
	prescriptive_paths = _save_prescriptive_outputs(root, prescriptive_df)

	print("Prescriptive analytics completed.")
	for name, file_path in prescriptive_paths.items():
		print(f"- {name}: {file_path}")


if __name__ == "__main__":
	main()

