"""Run Step 3 statistical extended experiment."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET_SERIES_KEY = "production_total"


def run_extended() -> None:
    from Project.models.statistical import StatisticalModelRunner, StatisticalStepConfig
    from Project.models.statistical.model_config import infer_seasonal_period_from_index
    from Project.preprocessing import PreprocessingConfig, prepare_preprocessing_for_profile, save_selected_preprocessing_config
    from Project.preprocessing.descriptive_analysis import load_target_series

    dataset_path = ROOT / "Datasets" / "Tavola_1.14.csv"
    series = load_target_series(dataset_path, target=TARGET_SERIES_KEY)

    _, preproc_output, candidate_df, selected_cfg = prepare_preprocessing_for_profile(
        series=series,
        profile="statistical",
        base_config=PreprocessingConfig(run_shapiro=True),
    )

    seasonal_period = infer_seasonal_period_from_index(preproc_output["splits"]["train"].index)
    stat_cfg = StatisticalStepConfig(
        p_values=(0, 1, 2, 3, 4),
        d_values=(0, 1),
        q_values=(0, 1, 2, 3, 4),
        p_seasonal_values=(0, 1, 2) if seasonal_period > 1 else (0, 1),
        d_seasonal_values=(0,),
        q_seasonal_values=(0, 1, 2) if seasonal_period > 1 else (0, 1),
        seasonal_period=seasonal_period,
    )

    runner = StatisticalModelRunner(
        train=preproc_output["splits"]["train"],
        validation=preproc_output["splits"]["val"],
        test=preproc_output["splits"]["test"],
        config=stat_cfg,
        original_series=series,
        use_log1p=selected_cfg.transform.use_log1p,
        diff_order=selected_cfg.transform.diff_order,
    )
    output = runner.run()

    metrics_dir = ROOT / "Results" / "metrics"
    artifacts_dir = ROOT / "Results" / "artifacts"
    preproc_metrics_dir = metrics_dir / "preprocessing"
    preproc_artifacts_dir = artifacts_dir / "preprocessing"
    stat_metrics_dir = metrics_dir / "statistical"
    stat_artifacts_dir = artifacts_dir / "statistical"
    plots_dir = ROOT / "Results" / "plots" / "statistical"
    preproc_metrics_dir.mkdir(parents=True, exist_ok=True)
    preproc_artifacts_dir.mkdir(parents=True, exist_ok=True)
    stat_metrics_dir.mkdir(parents=True, exist_ok=True)
    stat_artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "preproc_candidates": preproc_metrics_dir / "candidate_tests_extended.csv",
        "preproc_selected_config": preproc_artifacts_dir / "selected_config_extended.json",
        "sarima_grid": stat_metrics_dir / "sarima_grid_extended.csv",
        "hw_grid": stat_metrics_dir / "hw_grid_extended.csv",
        "summary": stat_metrics_dir / "summary_extended.csv",
        "residual_diagnostics": stat_metrics_dir / "residual_diagnostics_extended.csv",
        "forecasts": stat_metrics_dir / "forecasts_extended.csv",
        "winner": stat_artifacts_dir / "winner_params_extended.json",
    }

    candidate_df.to_csv(out_paths["preproc_candidates"], index=False)
    save_selected_preprocessing_config(selected_cfg, out_paths["preproc_selected_config"])
    output["sarima_grid"].to_csv(out_paths["sarima_grid"], index=False)
    output["hw_grid"].to_csv(out_paths["hw_grid"], index=False)
    output["summary"].to_csv(out_paths["summary"], index=False)
    output["residual_diagnostics"].to_csv(out_paths["residual_diagnostics"], index=False)
    output["forecast_table"].to_csv(out_paths["forecasts"], index=False)
    pd.Series(output["winner_params"]).to_json(out_paths["winner"], indent=2)

    plot_paths = StatisticalModelRunner.save_plots(output, plots_dir, suffix="extended")

    print(f"Step 3 extended completed. Winner: {output['winner']}")
    print("Saved outputs:")
    for name, path in out_paths.items():
        print(f"- {name}: {path}")
    for name, path in plot_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    run_extended()