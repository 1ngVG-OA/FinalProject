"""Run Step 3 statistical extended experiment with expanded SARIMA hyperparameter grid.

Deeper grid search across p, d, q ranges to explore wider parameter space.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TARGET_SERIES_KEY = "consumption_total"


def run_extended_step3() -> None:
    """Run Step 3 statistical extended (deeper SARIMA grid search)."""

    from Project.models.statistical import StatisticalModelRunner, StatisticalStepConfig
    from Project.models.statistical.model_config import infer_seasonal_period_from_index
    from Project.preprocessing import (
        DEFAULT_PREPROCESSING_CANDIDATES,
        PreprocessingConfig,
        prepare_preprocessing_from_candidates,
        save_selected_preprocessing_config,
    )
    from Project.preprocessing.descriptive_analysis import load_target_series

    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "Datasets" / "Tavola_1.14.csv"

    series = load_target_series(dataset_path, target=TARGET_SERIES_KEY)

    _, preproc_output, candidate_df, selected_cfg = prepare_preprocessing_from_candidates(
        series=series,
        base_config=PreprocessingConfig(run_shapiro=True),
        candidate_cfgs=DEFAULT_PREPROCESSING_CANDIDATES,
    )

    # Extended config: wider grid for p, d, q
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

    metrics_dir = root / "Results" / "metrics"
    artifacts_dir = root / "Results" / "artifacts"
    plots_dir = root / "Results" / "plots" / "forecasting"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "preproc_candidates": metrics_dir / "tavola_1_14_preproc_candidate_tests_stat_extended.csv",
        "preproc_selected_config": artifacts_dir / "tavola_1_14_preproc_selected_config_stat_extended.json",
        "sarima_grid": metrics_dir / "tavola_1_14_stat_sarima_grid_stat_extended.csv",
        "hw_grid": metrics_dir / "tavola_1_14_stat_hw_grid_stat_extended.csv",
        "summary": metrics_dir / "tavola_1_14_stat_summary_stat_extended.csv",
        "residual_diagnostics": metrics_dir / "tavola_1_14_stat_residual_diagnostics_stat_extended.csv",
        "forecasts": metrics_dir / "tavola_1_14_stat_forecasts_stat_extended.csv",
        "winner": artifacts_dir / "tavola_1_14_stat_winner_params_stat_extended.json",
    }

    candidate_df.to_csv(out_paths["preproc_candidates"], index=False)
    save_selected_preprocessing_config(selected_cfg, out_paths["preproc_selected_config"])

    output["sarima_grid"].to_csv(out_paths["sarima_grid"], index=False)
    output["hw_grid"].to_csv(out_paths["hw_grid"], index=False)
    output["summary"].to_csv(out_paths["summary"], index=False)
    output["residual_diagnostics"].to_csv(out_paths["residual_diagnostics"], index=False)
    output["forecast_table"].to_csv(out_paths["forecasts"], index=False)

    pd.Series(output["winner_params"]).to_json(out_paths["winner"], indent=2)

    plot_paths = StatisticalModelRunner.save_plots(output, plots_dir)

    print(f"Step 3 extended completed. Winner: {output['winner']}")
    print("Saved outputs:")
    for name, path in out_paths.items():
        print(f"- {name}: {path}")
    for name, path in plot_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    run_extended_step3()
