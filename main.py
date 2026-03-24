"""Project entry point.

Implemented steps:
1) Descriptive analysis for Tavola_1.14.
2) Time-series preprocessing (split, transforms, stationarity tests, local outliers).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from Project.preprocessing import (
	PreprocessingConfig,
	TimeSeriesPreprocessor,
	TransformConfig,
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
from Project.preprocessing.descriptive_analysis import (
	DescriptivePaths,
	load_target_series,
	run_descriptive_analysis,
)


def _save_preprocessing_outputs(
	root: Path,
	preproc: TimeSeriesPreprocessor,
	preproc_output: dict,
	candidate_df: pd.DataFrame,
) -> dict[str, Path]:
	"""Persist preprocessing artifacts to metrics and processed data folders."""

	metrics_dir = root / "Results" / "metrics"
	processed_dir = root / "Datasets" / "processed"
	preproc_plots_dir = root / "Results" / "plots" / "preprocessing"

	metrics_dir.mkdir(parents=True, exist_ok=True)
	processed_dir.mkdir(parents=True, exist_ok=True)
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
		"preproc_split_summary": metrics_dir / "tavola_1_14_preproc_split_summary_v1.csv",
		"preproc_tests": metrics_dir / "tavola_1_14_preproc_tests_v1.csv",
		"preproc_local_outliers": metrics_dir / "tavola_1_14_preproc_local_outliers_v1.csv",
		"preproc_candidate_tests": metrics_dir / "tavola_1_14_preproc_candidate_tests_v1.csv",
		"preproc_train": processed_dir / "tavola_1_14_preprocessed_train_v1.csv",
		"preproc_val": processed_dir / "tavola_1_14_preprocessed_val_v1.csv",
		"preproc_test": processed_dir / "tavola_1_14_preprocessed_test_v1.csv",
	}

	preproc_output["split_summary"].to_csv(output_paths["preproc_split_summary"], index=False)
	tests_df.to_csv(output_paths["preproc_tests"], index=False)
	preproc_output["local_outliers"].to_csv(output_paths["preproc_local_outliers"], index=False)
	candidate_df.to_csv(output_paths["preproc_candidate_tests"], index=False)

	for split_name in ("train", "val", "test"):
		split_series = preproc_output["splits"][split_name]
		split_series.rename("value").reset_index().to_csv(output_paths[f"preproc_{split_name}"], index=False)

	plot_paths = preproc.save_preprocessing_plots(preproc_output, preproc_plots_dir)
	output_paths.update(plot_paths)

	return output_paths


def _save_statistical_outputs(root: Path, stat_output: dict) -> dict[str, Path]:
	"""Persist Step 3 statistical artifacts to metrics/plots/artifacts folders."""

	metrics_dir = root / "Results" / "metrics"
	plots_dir = root / "Results" / "plots" / "forecasting"
	artifacts_dir = root / "Results" / "artifacts"

	metrics_dir.mkdir(parents=True, exist_ok=True)
	plots_dir.mkdir(parents=True, exist_ok=True)
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	output_paths = {
		"stat_sarima_grid": metrics_dir / "tavola_1_14_stat_sarima_grid_v1.csv",
		"stat_hw_grid": metrics_dir / "tavola_1_14_stat_hw_grid_v1.csv",
		"stat_summary": metrics_dir / "tavola_1_14_stat_summary_v1.csv",
		"stat_residual_diagnostics": metrics_dir / "tavola_1_14_stat_residual_diagnostics_v1.csv",
		"stat_forecasts": metrics_dir / "tavola_1_14_stat_forecasts_v1.csv",
		"stat_winner_params": artifacts_dir / "tavola_1_14_stat_winner_params_v1.json",
	}

	stat_output["sarima_grid"].to_csv(output_paths["stat_sarima_grid"], index=False)
	stat_output["hw_grid"].to_csv(output_paths["stat_hw_grid"], index=False)
	stat_output["summary"].to_csv(output_paths["stat_summary"], index=False)
	stat_output["residual_diagnostics"].to_csv(output_paths["stat_residual_diagnostics"], index=False)
	stat_output["forecast_table"].to_csv(output_paths["stat_forecasts"], index=False)

	pd.Series(stat_output["winner_params"]).to_json(output_paths["stat_winner_params"], indent=2)

	plot_paths = StatisticalModelRunner.save_plots(stat_output, plots_dir)
	output_paths.update(plot_paths)

	return output_paths


def _save_ml_outputs(root: Path, ml_output: dict) -> dict[str, Path]:
	"""Persist Step 4 ML artifacts to metrics/plots/artifacts folders."""

	metrics_dir = root / "Results" / "metrics"
	plots_dir = root / "Results" / "plots" / "forecasting"
	artifacts_dir = root / "Results" / "artifacts"

	metrics_dir.mkdir(parents=True, exist_ok=True)
	plots_dir.mkdir(parents=True, exist_ok=True)
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	output_paths = {
		"ml_grid": metrics_dir / "tavola_1_14_ml_grid_v1.csv",
		"ml_summary": metrics_dir / "tavola_1_14_ml_summary_v1.csv",
		"ml_forecasts": metrics_dir / "tavola_1_14_ml_forecasts_v1.csv",
		"ml_feature_selection": metrics_dir / "tavola_1_14_ml_feature_selection_v1.csv",
		"ml_winner_params": artifacts_dir / "tavola_1_14_ml_winner_params_v1.json",
		"ml_config": artifacts_dir / "tavola_1_14_ml_config_v1.json",
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


def main() -> None:
	"""Run the currently implemented pipeline steps end-to-end."""

	root = Path(__file__).resolve().parent
	dataset_path = root / "Datasets" / "Tavola_1.14.csv"

	# Step 1 - descriptive analysis.
	desc_paths = DescriptivePaths(
		dataset_path=dataset_path,
		results_metrics_dir=root / "Results" / "metrics",
		results_plots_dir=root / "Results" / "plots" / "descriptive",
	)
	descriptive_outputs = run_descriptive_analysis(desc_paths)

	print("Descriptive analysis completed.")
	for name, file_path in descriptive_outputs.items():
		print(f"- {name}: {file_path}")

	# Step 2 - preprocessing based on descriptive conclusions.
	series = load_target_series(dataset_path)

	chosen_cfg = PreprocessingConfig(
		transform=TransformConfig(
			use_log1p=True,
			power_exponent=None,
			diff_order=1,
			scale_method="none",
		),
		run_shapiro=True,
	)
	preproc = TimeSeriesPreprocessor(series, chosen_cfg)
	preproc_output = preproc.preprocess()

	candidate_cfgs = [
		TransformConfig(use_log1p=False, diff_order=0, scale_method="none"),
		TransformConfig(use_log1p=False, diff_order=1, scale_method="none"),
		TransformConfig(use_log1p=True, diff_order=1, scale_method="none"),
		TransformConfig(use_log1p=True, diff_order=2, scale_method="none"),
	]
	candidate_df = preproc.evaluate_candidates(candidate_cfgs)

	preproc_outputs = _save_preprocessing_outputs(root, preproc, preproc_output, candidate_df)

	print("Preprocessing completed.")
	for name, file_path in preproc_outputs.items():
		print(f"- {name}: {file_path}")

	# Step 3 - canonical statistical baseline (SARIMA + Holt-Winters).
	# Standalone Step 3 runners are available in Mains/ for baseline-only and extended experiments.
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
		use_log1p=chosen_cfg.transform.use_log1p,
		diff_order=chosen_cfg.transform.diff_order,
	)
	stat_output = stat_runner.run()
	stat_paths = _save_statistical_outputs(root, stat_output)

	print(f"Statistical Step completed. Winner: {stat_output['winner']}")
	for name, file_path in stat_paths.items():
		print(f"- {name}: {file_path}")

	# Step 4 - canonical non-neural ML baseline.
	# Standalone Step 4 runners are available in Mains/ for baseline-only and extended experiments.
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
		use_log1p=chosen_cfg.transform.use_log1p,
		diff_order=chosen_cfg.transform.diff_order,
	)
	ml_output = ml_runner.run()
	ml_paths = _save_ml_outputs(root, ml_output)

	print(f"Step 4 ML completed. Winner: {ml_output['winner']}")
	for name, file_path in ml_paths.items():
		print(f"- {name}: {file_path}")


if __name__ == "__main__":
	main()

