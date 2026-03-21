"""Project entry point.

Current implemented step:
1) Descriptive analysis for Tavola_1.14.

The script reads the dataset from `Datasets/`, computes descriptive statistics,
and saves metrics/plots to `Results/`.
"""

from pathlib import Path

from Project.preprocessing.descriptive_analysis import (
	DescriptivePaths,
	run_descriptive_analysis,
)


def main() -> None:
	"""Run currently implemented pipeline steps."""

	root = Path(__file__).resolve().parent

	paths = DescriptivePaths(
		dataset_path=root / "Datasets" / "Tavola_1.14.csv",
		results_metrics_dir=root / "Results" / "metrics",
		results_plots_dir=root / "Results" / "plots" / "descriptive",
	)

	outputs = run_descriptive_analysis(paths)

	print("Descriptive analysis completed.")
	for name, file_path in outputs.items():
		print(f"- {name}: {file_path}")


if __name__ == "__main__":
	main()

