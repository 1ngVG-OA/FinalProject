"""Esecuzione baseline dello Step 4 ML non neurale.

Script operativo per lanciare preprocessing profile=ml, training dei modelli
e salvataggio completo di metriche/artifact/plot.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET_SERIES_KEY = "production_total"


def run_baseline() -> None:
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

    dataset_path = ROOT / "Datasets" / "Tavola_1.14.csv"
    series = load_target_series(dataset_path, target=TARGET_SERIES_KEY)

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
    # Persistenza risultati su Results/metrics, Results/artifacts e plot
    # ------------------------------------------------------------------

    metrics_dir = ROOT / "Results" / "metrics"
    artifacts_dir = ROOT / "Results" / "artifacts"
    preproc_metrics_dir = metrics_dir / "preprocessing"
    preproc_artifacts_dir = artifacts_dir / "preprocessing"
    ml_metrics_dir = metrics_dir / "ml"
    ml_artifacts_dir = artifacts_dir / "ml"
    plots_dir = ROOT / "Results" / "plots" / "ml"
    preproc_metrics_dir.mkdir(parents=True, exist_ok=True)
    preproc_artifacts_dir.mkdir(parents=True, exist_ok=True)
    ml_metrics_dir.mkdir(parents=True, exist_ok=True)
    ml_artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "preproc_candidates": preproc_metrics_dir / "candidate_tests_ml_baseline.csv",
        "preproc_selected_config": preproc_artifacts_dir / "selected_config_ml_baseline.json",
        "grid": ml_metrics_dir / "grid_baseline.csv",
        "summary": ml_metrics_dir / "summary_baseline.csv",
        "forecasts": ml_metrics_dir / "forecasts_baseline.csv",
        "feature_selection": ml_metrics_dir / "feature_selection_baseline.csv",
        "winner": ml_artifacts_dir / "winner_params_baseline.json",
        "config": ml_artifacts_dir / "config_baseline.json",
    }

    candidate_df.to_csv(out_paths["preproc_candidates"], index=False)
    save_selected_preprocessing_config(selected_cfg, out_paths["preproc_selected_config"])

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
    run_baseline()
