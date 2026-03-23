"""Run Step 4 ML extended experiment (XGBoost enabled) with separate outputs."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Project.models.ml import MLModelRunner, MLStepConfig
from Project.preprocessing import PreprocessingConfig, TimeSeriesPreprocessor, TransformConfig
from Project.preprocessing.descriptive_analysis import load_target_series


def run_extended_step4() -> None:
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

    ml_cfg = MLStepConfig(
        lookback_values=(6, 8, 12),
        feature_selection="importance",
        selected_feature_count=6,
        use_xgboost=True,
        dt_max_depth=(3, 5, None),
        dt_min_samples_leaf=(1, 2, 4),
        rf_n_estimators=(200, 400),
        rf_max_depth=(4, 8, None),
        rf_min_samples_leaf=(1, 2),
        gbr_n_estimators=(200, 400),
        gbr_learning_rate=(0.03, 0.05, 0.1),
        gbr_max_depth=(2, 3),
        xgb_n_estimators=(300, 600),
        xgb_learning_rate=(0.03, 0.05),
        xgb_max_depth=(2, 3),
        xgb_subsample=(0.8, 1.0),
        xgb_colsample_bytree=(0.8, 1.0),
    )

    runner = MLModelRunner(
        train=preproc_output["splits"]["train"],
        validation=preproc_output["splits"]["val"],
        test=preproc_output["splits"]["test"],
        config=ml_cfg,
        original_series=series,
        use_log1p=chosen_cfg.transform.use_log1p,
        diff_order=chosen_cfg.transform.diff_order,
    )
    output = runner.run()

    metrics_dir = root / "Results" / "metrics"
    artifacts_dir = root / "Results" / "artifacts"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "grid": metrics_dir / "tavola_1_14_ml_grid_xgb_v2.csv",
        "summary": metrics_dir / "tavola_1_14_ml_summary_xgb_v2.csv",
        "forecasts": metrics_dir / "tavola_1_14_ml_forecasts_xgb_v2.csv",
        "feature_selection": metrics_dir / "tavola_1_14_ml_feature_selection_xgb_v2.csv",
        "winner": artifacts_dir / "tavola_1_14_ml_winner_params_xgb_v2.json",
        "config": artifacts_dir / "tavola_1_14_ml_config_xgb_v2.json",
    }

    output["grid"].to_csv(out_paths["grid"], index=False)
    output["summary"].to_csv(out_paths["summary"], index=False)
    output["forecast_table"].to_csv(out_paths["forecasts"], index=False)
    output["feature_selection_report"].to_csv(out_paths["feature_selection"], index=False)

    pd.Series(output["winner_params"]).to_json(out_paths["winner"], indent=2)
    pd.Series(output["config"]).to_json(out_paths["config"], indent=2)

    print(f"Step 4 extended completed. Winner: {output['winner']}")
    print("Saved outputs:")
    for name, path in out_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    run_extended_step4()
