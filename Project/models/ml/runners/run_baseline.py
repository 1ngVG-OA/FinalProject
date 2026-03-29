"""Esecuzione baseline dello Step 4 ML non neurale.

Script operativo per lanciare preprocessing profile=ml, training dei modelli
e salvataggio completo di metriche/artifact/plot.
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

    from Project.models.ml import MLModelRunner, MLStepConfig, save_ml_plots
    from Project.preprocessing import (
        PreprocessingConfig,
        prepare_preprocessing_for_profile,
        save_selected_preprocessing_config,
    )
    from Project.preprocessing.descriptive_analysis import load_target_series

    series_cfg = get_series_config(series_key)
    dataset_path = series_cfg.dataset_path
    series = load_target_series(dataset_path, target=series_key)

    _, preproc_output, candidate_df, selected_cfg = prepare_preprocessing_for_profile(
        series=series,
        profile="ml",
        base_config=PreprocessingConfig(run_shapiro=True),
    )

    # ------------------------------------------------------------------
    # Configurazione modello, run e raccolta output
    # ------------------------------------------------------------------

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

    runner = MLModelRunner(
        train=preproc_output["splits"]["train"],
        validation=preproc_output["splits"]["val"],
        test=preproc_output["splits"]["test"],
        config=ml_cfg,
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
    ml_metrics_dir = get_results_subdir(series_key, "metrics", "ml")
    ml_artifacts_dir = get_results_subdir(series_key, "artifacts", "ml")
    plots_dir = get_results_subdir(series_key, "plots", "ml")
    processed_dir = get_processed_root(series_key)
    preproc_metrics_dir.mkdir(parents=True, exist_ok=True)
    preproc_artifacts_dir.mkdir(parents=True, exist_ok=True)
    ml_metrics_dir.mkdir(parents=True, exist_ok=True)
    ml_artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "preproc_candidates": preproc_metrics_dir / "candidate_tests_ml_baseline.csv",
        "preproc_selected_config": preproc_artifacts_dir / "selected_config_ml_baseline.json",
        "preproc_train": processed_dir / "preprocessed_train_v1.csv",
        "preproc_val": processed_dir / "preprocessed_val_v1.csv",
        "preproc_test": processed_dir / "preprocessed_test_v1.csv",
        "grid": ml_metrics_dir / "grid_baseline.csv",
        "summary": ml_metrics_dir / "summary_baseline.csv",
        "forecasts": ml_metrics_dir / "forecasts_baseline.csv",
        "feature_selection": ml_metrics_dir / "feature_selection_baseline.csv",
        "winner": ml_artifacts_dir / "winner_params_baseline.json",
        "config": ml_artifacts_dir / "config_baseline.json",
    }

    candidate_df.to_csv(out_paths["preproc_candidates"], index=False)
    save_selected_preprocessing_config(selected_cfg, out_paths["preproc_selected_config"])
    for split_name in ("train", "val", "test"):
        preproc_output["splits"][split_name].rename("value").reset_index().to_csv(out_paths[f"preproc_{split_name}"], index=False)

    output["grid"].to_csv(out_paths["grid"], index=False)
    output["summary"].to_csv(out_paths["summary"], index=False)
    output["forecast_table"].to_csv(out_paths["forecasts"], index=False)
    output["feature_selection_report"].to_csv(out_paths["feature_selection"], index=False)

    pd.Series(output["winner_params"]).to_json(out_paths["winner"], indent=2)
    pd.Series(output["config"]).to_json(out_paths["config"], indent=2)

    plot_paths = save_ml_plots(output, plots_dir, suffix="baseline")

    print(f"Step 4 baseline completed. Winner: {output['winner']}")
    print("Saved outputs:")
    for name, path in out_paths.items():
        print(f"- {name}: {path}")
    for name, path in plot_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML baseline for a configured series.")
    parser.add_argument("--series", default=TARGET_SERIES_KEY, help="Series key configured in config.py")
    args = parser.parse_args()
    run_baseline(series_key=args.series)
