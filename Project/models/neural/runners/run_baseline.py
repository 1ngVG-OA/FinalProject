"""Run Step 5 neural baseline experiment with MLP and LSTM."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET_SERIES_KEY = "production_total"


def run_baseline() -> None:
    from Project.models.neural import (
        NeuralModelRunner,
        build_compact_neural_config,
        save_neural_plots,
    )
    from Project.preprocessing import (
        PreprocessingConfig,
        prepare_preprocessing_for_profile,
        save_selected_preprocessing_config,
    )
    from Project.preprocessing.descriptive_analysis import load_target_series

    dataset_path = ROOT / "Datasets" / "Tavola_1.14.csv"
    series = load_target_series(dataset_path, target=TARGET_SERIES_KEY)

    _, preproc_output, candidate_df, selected_cfg = prepare_preprocessing_for_profile(
        series=series,
        profile="neural",
        base_config=PreprocessingConfig(run_shapiro=True),
    )

    neural_cfg = build_compact_neural_config()

    runner = NeuralModelRunner(
        train=preproc_output["splits"]["train"],
        validation=preproc_output["splits"]["val"],
        test=preproc_output["splits"]["test"],
        config=neural_cfg,
        original_series=series,
        preprocessing_config=selected_cfg,
    )
    output = runner.run()

    metrics_dir = ROOT / "Results" / "metrics"
    artifacts_dir = ROOT / "Results" / "artifacts"
    preproc_metrics_dir = metrics_dir / "preprocessing"
    preproc_artifacts_dir = artifacts_dir / "preprocessing"
    neural_metrics_dir = metrics_dir / "neural"
    neural_artifacts_dir = artifacts_dir / "neural"
    plots_dir = ROOT / "Results" / "plots" / "neural"

    preproc_metrics_dir.mkdir(parents=True, exist_ok=True)
    preproc_artifacts_dir.mkdir(parents=True, exist_ok=True)
    neural_metrics_dir.mkdir(parents=True, exist_ok=True)
    neural_artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "preproc_candidates": preproc_metrics_dir / "candidate_tests_neural_baseline.csv",
        "preproc_selected_config": preproc_artifacts_dir / "selected_config_neural_baseline.json",
        "grid": neural_metrics_dir / "grid_baseline.csv",
        "summary": neural_metrics_dir / "summary_baseline.csv",
        "forecasts": neural_metrics_dir / "forecasts_baseline.csv",
        "winner": neural_artifacts_dir / "winner_params_baseline.json",
        "config": neural_artifacts_dir / "config_baseline.json",
    }

    candidate_df.to_csv(out_paths["preproc_candidates"], index=False)
    save_selected_preprocessing_config(selected_cfg, out_paths["preproc_selected_config"])

    output["grid"].to_csv(out_paths["grid"], index=False)
    output["summary"].to_csv(out_paths["summary"], index=False)
    output["forecast_table"].to_csv(out_paths["forecasts"], index=False)

    pd.Series(output["winner_params"]).to_json(out_paths["winner"], indent=2)
    pd.Series(output["config"]).to_json(out_paths["config"], indent=2)

    plot_paths = save_neural_plots(output, plots_dir, suffix="baseline")

    print(f"Step 5 neural baseline completed. Winner: {output['winner']}")
    print("Saved outputs:")
    for name, path in out_paths.items():
        print(f"- {name}: {path}")
    for name, path in plot_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    run_baseline()