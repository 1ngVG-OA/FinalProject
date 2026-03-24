"""Run Step 3 statistical baseline experiment with the same configuration used by main.py.

SARIMA grid search and Holt-Winters benchmark using baseline hyperparameters.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Project.models.statistical import StatisticalModelRunner, StatisticalStepConfig
from Project.models.statistical.model_config import infer_seasonal_period_from_index
from Project.preprocessing import PreprocessingConfig, TimeSeriesPreprocessor, TransformConfig
from Project.preprocessing.descriptive_analysis import load_target_series


def run_baseline_step3() -> None:
    """Run Step 3 statistical baseline (same config as main.py)."""

    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "Datasets" / "Tavola_1.14.csv"

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

    # Baseline config: same as main.py Step 3
    seasonal_period = infer_seasonal_period_from_index(preproc_output["splits"]["train"].index)
    stat_cfg = StatisticalStepConfig(
        d_values=(0, 1),
        seasonal_period=seasonal_period,
    )

    runner = StatisticalModelRunner(
        train=preproc_output["splits"]["train"],
        validation=preproc_output["splits"]["val"],
        test=preproc_output["splits"]["test"],
        config=stat_cfg,
        original_series=series,
        use_log1p=chosen_cfg.transform.use_log1p,
        diff_order=chosen_cfg.transform.diff_order,
    )
    output = runner.run()

    metrics_dir = root / "Results" / "metrics"
    artifacts_dir = root / "Results" / "artifacts"
    plots_dir = root / "Results" / "plots" / "forecasting"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "sarima_grid": metrics_dir / "tavola_1_14_stat_sarima_grid_stat_baseline.csv",
        "hw_grid": metrics_dir / "tavola_1_14_stat_hw_grid_stat_baseline.csv",
        "summary": metrics_dir / "tavola_1_14_stat_summary_stat_baseline.csv",
        "residual_diagnostics": metrics_dir / "tavola_1_14_stat_residual_diagnostics_stat_baseline.csv",
        "forecasts": metrics_dir / "tavola_1_14_stat_forecasts_stat_baseline.csv",
        "winner": artifacts_dir / "tavola_1_14_stat_winner_params_stat_baseline.json",
    }

    output["sarima_grid"].to_csv(out_paths["sarima_grid"], index=False)
    output["hw_grid"].to_csv(out_paths["hw_grid"], index=False)
    output["summary"].to_csv(out_paths["summary"], index=False)
    output["residual_diagnostics"].to_csv(out_paths["residual_diagnostics"], index=False)
    output["forecast_table"].to_csv(out_paths["forecasts"], index=False)

    pd.Series(output["winner_params"]).to_json(out_paths["winner"], indent=2)

    plot_paths = StatisticalModelRunner.save_plots(output, plots_dir)

    print(f"Step 3 baseline completed. Winner: {output['winner']}")
    print("Saved outputs:")
    for name, path in out_paths.items():
        print(f"- {name}: {path}")
    for name, path in plot_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    run_baseline_step3()
