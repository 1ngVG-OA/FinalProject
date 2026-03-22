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
			diff_order=2,
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


if __name__ == "__main__":
	main()

