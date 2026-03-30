from __future__ import annotations

from Project.evaluation import (
    build_cross_family_comparison,
    build_diebold_mariano_table,
    build_prescriptive_table,
)
from Project.models.ml import MLModelRunner, MLStepConfig
from Project.models.neural import NeuralModelRunner, NeuralStepConfig
from Project.models.statistical import (
    StatisticalModelRunner,
    StatisticalStepConfig,
    infer_seasonal_period_from_index,
)
from Project.preprocessing import PreprocessingConfig, prepare_preprocessing_for_profile
from Project.preprocessing.descriptive_analysis import (
    DescriptivePaths,
    load_target_series,
    run_descriptive_analysis,
)
from config import get_results_root, get_results_subdir, get_series_config, get_series_output_name
from main import (
    _save_comparison_outputs,
    _save_inferential_outputs,
    _save_ml_outputs,
    _save_neural_outputs,
    _save_preprocessing_outputs,
    _save_prescriptive_outputs,
    _save_statistical_outputs,
)

CONSUMPTION_SERIES_KEY = "consumption_total"


def main() -> None:
    series_key = CONSUMPTION_SERIES_KEY
    series_cfg = get_series_config(series_key)
    series_output_name = get_series_output_name(series_key)
    dataset_path = series_cfg.dataset_path
    results_root = get_results_root(series_key)

    print(f"Running pipeline for series: {series_key}")
    print(f"Series output name: {series_output_name}")
    print(f"Series output root: {results_root}")

    desc_paths = DescriptivePaths(
        dataset_path=dataset_path,
        results_metrics_dir=get_results_subdir(series_key, "metrics", "descriptive"),
        results_plots_dir=get_results_subdir(series_key, "plots", "descriptive"),
    )
    descriptive_outputs = run_descriptive_analysis(desc_paths, target=series_key)

    print("Descriptive analysis completed.")
    for name, file_path in descriptive_outputs.items():
        print(f"- {name}: {file_path}")

    series = load_target_series(dataset_path, target=series_key)

    preproc, preproc_output, candidate_df, selected_cfg = prepare_preprocessing_for_profile(
        series=series,
        profile="statistical",
        base_config=PreprocessingConfig(run_shapiro=True),
    )
    preproc_outputs = _save_preprocessing_outputs(
        series_key,
        preproc,
        preproc_output,
        candidate_df,
        selected_cfg,
    )

    print("Preprocessing completed.")
    for name, file_path in preproc_outputs.items():
        print(f"- {name}: {file_path}")

    seasonal_period = infer_seasonal_period_from_index(preproc_output["splits"]["train"].index)
    stat_cfg = StatisticalStepConfig(
        d_values=(0, 1),
        seasonal_period=seasonal_period,
    )
    stat_runner = StatisticalModelRunner(
        train=preproc_output["splits"]["train"],
        validation=preproc_output["splits"]["val"],
        test=preproc_output["splits"]["test"],
        config=stat_cfg,
        original_series=series,
        use_log1p=selected_cfg.transform.use_log1p,
        diff_order=selected_cfg.transform.diff_order,
    )
    stat_output = stat_runner.run()
    stat_paths = _save_statistical_outputs(series_key, stat_output)

    print(f"Statistical Step completed. Winner: {stat_output['winner']}")
    for name, file_path in stat_paths.items():
        print(f"- {name}: {file_path}")

    ml_cfg = MLStepConfig(
        lookback_values=(2, 3, 4),
        feature_selection="importance",
        selected_feature_count=3,
        cv_folds=3,
        overfitting_lambda=0.5,
        use_xgboost=False,
        dt_max_depth=(3, None),
        dt_min_samples_leaf=(1, 2),
        rf_n_estimators=(200,),
        rf_max_depth=(5, None),
        rf_min_samples_leaf=(1, 2),
        gbr_n_estimators=(200,),
        gbr_learning_rate=(0.05, 0.1),
        gbr_max_depth=(2, 3),
    )
    ml_runner = MLModelRunner(
        train=preproc_output["splits"]["train"],
        validation=preproc_output["splits"]["val"],
        test=preproc_output["splits"]["test"],
        config=ml_cfg,
        original_series=series,
        use_log1p=selected_cfg.transform.use_log1p,
        diff_order=selected_cfg.transform.diff_order,
    )
    ml_output = ml_runner.run()
    ml_paths = _save_ml_outputs(series_key, ml_output)

    print(f"Step 4 ML completed. Winner: {ml_output['winner']}")
    for name, file_path in ml_paths.items():
        print(f"- {name}: {file_path}")

    _, neural_preproc_output, _, neural_selected_cfg = prepare_preprocessing_for_profile(
        series=series,
        profile="neural",
        base_config=PreprocessingConfig(run_shapiro=True),
    )
    neural_cfg = NeuralStepConfig(
        candidate_models=("mlp", "lstm"),
        lookback_values=(2, 4, 6),
        mlp_hidden_sizes=(8, 16),
        mlp_activations=("relu", "tanh"),
        mlp_dropouts=(0.0, 0.1),
        lstm_hidden_sizes=(8, 16),
        lstm_num_layers=(1,),
        lstm_dropouts=(0.0,),
        batch_sizes=(8, 16),
        learning_rates=(1e-3,),
        weight_decays=(0.0,),
        max_epochs=200,
        patience=20,
        seed=42,
        device="cpu",
    )
    neural_runner = NeuralModelRunner(
        train=neural_preproc_output["splits"]["train"],
        validation=neural_preproc_output["splits"]["val"],
        test=neural_preproc_output["splits"]["test"],
        config=neural_cfg,
        original_series=series,
        preprocessing_config=neural_selected_cfg,
    )
    neural_output = neural_runner.run()
    neural_paths = _save_neural_outputs(series_key, neural_output)

    print(f"Step 5 Neural completed. Winner: {neural_output['winner']}")
    for name, file_path in neural_paths.items():
        print(f"- {name}: {file_path}")

    comparison_output = build_cross_family_comparison(stat_output, ml_output, neural_output)
    comparison_paths = _save_comparison_outputs(series_key, comparison_output)

    print(
        "Cross-family comparison completed. "
        f"Global winner: {comparison_output['global_winner']['family']}::{comparison_output['global_winner']['model']}"
    )
    for name, file_path in comparison_paths.items():
        print(f"- {name}: {file_path}")

    inferential_df = build_diebold_mariano_table(comparison_output["winner_forecasts"])
    inferential_paths = _save_inferential_outputs(series_key, inferential_df)

    print("Inferential statistics completed.")
    for name, file_path in inferential_paths.items():
        print(f"- {name}: {file_path}")

    prescriptive_df = build_prescriptive_table(
        comparison_output["family_winners"],
        comparison_output["winner_forecasts"],
        series,
    )
    prescriptive_paths = _save_prescriptive_outputs(series_key, prescriptive_df)

    print("Prescriptive analytics completed.")
    for name, file_path in prescriptive_paths.items():
        print(f"- {name}: {file_path}")


if __name__ == "__main__":
    main()
