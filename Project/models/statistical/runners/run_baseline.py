"""Esecuzione baseline dello Step 3 statistico.

Script operativo per lanciare preprocessing profile=statistical,
allenamento SARIMA e salvataggio completo di metriche/artifact/plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from config import (
    DEFAULT_SERIES_KEY,
    get_processed_root,
    get_results_subdir,
    get_series_config,
)

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET_SERIES_KEY = DEFAULT_SERIES_KEY


def run_baseline(series_key: str = TARGET_SERIES_KEY) -> None:
    # ------------------------------------------------------------------
    # Import locali e caricamento serie target
    # ------------------------------------------------------------------

    from Project.models.statistical import StatisticalModelRunner, StatisticalStepConfig
    from Project.models.statistical.model_config import infer_seasonal_period_from_index
    from Project.preprocessing import PreprocessingConfig, prepare_preprocessing_for_profile, save_selected_preprocessing_config
    from Project.preprocessing.descriptive_analysis import load_target_series

    series_cfg = get_series_config(series_key)
    dataset_path = series_cfg.dataset_path
    series = load_target_series(dataset_path, target=series_key)

    _, preproc_output, candidate_df, selected_cfg = prepare_preprocessing_for_profile(
        series=series,
        profile="statistical",
        base_config=PreprocessingConfig(run_shapiro=True),
    )

    # ------------------------------------------------------------------
    # Configurazione statistica, run modello e raccolta output
    # ------------------------------------------------------------------

    seasonal_period = infer_seasonal_period_from_index(preproc_output["splits"]["train"].index)
    stat_cfg = StatisticalStepConfig(d_values=(0, 1), seasonal_period=seasonal_period)

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

    # ------------------------------------------------------------------
    # Persistenza risultati nella cartella serie-aware sotto Results/<SerieOutputName>/...
    # ------------------------------------------------------------------

    preproc_metrics_dir = get_results_subdir(series_key, "metrics", "preprocessing")
    preproc_artifacts_dir = get_results_subdir(series_key, "artifacts", "preprocessing")
    stat_metrics_dir = get_results_subdir(series_key, "metrics", "statistical")
    stat_artifacts_dir = get_results_subdir(series_key, "artifacts", "statistical")
    plots_dir = get_results_subdir(series_key, "plots", "statistical")
    processed_dir = get_processed_root(series_key)
    preproc_metrics_dir.mkdir(parents=True, exist_ok=True)
    preproc_artifacts_dir.mkdir(parents=True, exist_ok=True)
    stat_metrics_dir.mkdir(parents=True, exist_ok=True)
    stat_artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "preproc_candidates": preproc_metrics_dir / "candidate_tests_baseline.csv",
        "preproc_selected_config": preproc_artifacts_dir / "selected_config_baseline.json",
        "preproc_train": processed_dir / "preprocessed_train_v1.csv",
        "preproc_val": processed_dir / "preprocessed_val_v1.csv",
        "preproc_test": processed_dir / "preprocessed_test_v1.csv",
        "sarima_grid": stat_metrics_dir / "sarima_grid_baseline.csv",
        "summary": stat_metrics_dir / "summary_baseline.csv",
        "residual_diagnostics": stat_metrics_dir / "residual_diagnostics_baseline.csv",
        "forecasts": stat_metrics_dir / "forecasts_baseline.csv",
        "winner": stat_artifacts_dir / "winner_params_baseline.json",
    }

    candidate_df.to_csv(out_paths["preproc_candidates"], index=False)
    save_selected_preprocessing_config(selected_cfg, out_paths["preproc_selected_config"])
    for split_name in ("train", "val", "test"):
        preproc_output["splits"][split_name].rename("value").reset_index().to_csv(out_paths[f"preproc_{split_name}"], index=False)
    output["sarima_grid"].to_csv(out_paths["sarima_grid"], index=False)
    output["summary"].to_csv(out_paths["summary"], index=False)
    output["residual_diagnostics"].to_csv(out_paths["residual_diagnostics"], index=False)
    output["forecast_table"].to_csv(out_paths["forecasts"], index=False)
    pd.Series(output["winner_params"]).to_json(out_paths["winner"], indent=2)

    plot_paths = StatisticalModelRunner.save_plots(output, plots_dir, suffix="baseline")

    print(f"Step 3 baseline completed. Winner: {output['winner']}")
    print("Saved outputs:")
    for name, path in out_paths.items():
        print(f"- {name}: {path}")
    for name, path in plot_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the statistical baseline for a configured series.")
    parser.add_argument("--series", default=TARGET_SERIES_KEY, help="Series key configured in config.py")
    args = parser.parse_args()
    run_baseline(series_key=args.series)