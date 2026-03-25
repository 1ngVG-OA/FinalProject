# Automatic configuration utilities for Step 2 preprocessing.
#
# This module centralizes how preprocessing settings are chosen and persisted.
# It supports four core operations:
#
# 1. evaluate multiple transformation candidates on the training split,
# 2. select the best candidate with deterministic ranking rules,
# 3. save/load the selected configuration as JSON,
# 4. build and run a preprocessor aligned with the selected configuration.
#
# The intent is to keep Step 2 reproducible and avoid ad-hoc manual choices
# between runs.
#

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Literal
import json

import pandas as pd

from Project.preprocessing.time_series_preprocessor import (
    OutlierConfig,
    PreprocessingConfig,
    SplitConfig,
    TimeSeriesPreprocessor,
    TransformConfig,
)

# Supported preprocessing profiles by downstream model family.
PreprocessingProfile = Literal["statistical", "ml", "neural"]

# Candidate transformation configurations for statistical models.
STATISTICAL_PREPROCESSING_CANDIDATES: tuple[TransformConfig, ...] = (
    TransformConfig(use_log1p=False, diff_order=0, scale_method="none"),
    TransformConfig(use_log1p=False, diff_order=1, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=1, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=2, scale_method="none"),
)

# Candidate transformation configurations for ML models.
ML_PREPROCESSING_CANDIDATES: tuple[TransformConfig, ...] = (
    TransformConfig(use_log1p=False, diff_order=0, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=0, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=1, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=1, scale_method="standard"),
    TransformConfig(use_log1p=True, diff_order=1, scale_method="minmax"),
)

# Candidate transformation configurations for neural models.
NEURAL_PREPROCESSING_CANDIDATES: tuple[TransformConfig, ...] = (
    TransformConfig(use_log1p=False, diff_order=0, scale_method="standard"),
    TransformConfig(use_log1p=True, diff_order=0, scale_method="standard"),
    TransformConfig(use_log1p=True, diff_order=1, scale_method="standard"),
    TransformConfig(use_log1p=False, diff_order=0, scale_method="minmax"),
    TransformConfig(use_log1p=True, diff_order=0, scale_method="minmax"),
)

PREPROCESSING_CANDIDATES_BY_PROFILE: dict[PreprocessingProfile, tuple[TransformConfig, ...]] = {
    "statistical": STATISTICAL_PREPROCESSING_CANDIDATES,
    "ml": ML_PREPROCESSING_CANDIDATES,
    "neural": NEURAL_PREPROCESSING_CANDIDATES,
}

# Backward-compatible default: same profile used previously.
DEFAULT_PREPROCESSING_CANDIDATES = STATISTICAL_PREPROCESSING_CANDIDATES


def get_preprocessing_candidates(profile: PreprocessingProfile) -> tuple[TransformConfig, ...]:
    """Return candidate transformations for a model-family profile."""

    return PREPROCESSING_CANDIDATES_BY_PROFILE[profile]

def select_best_transform_config(candidate_df: pd.DataFrame) -> TransformConfig:
    #Select the best transformation candidate from evaluation results.
    #
    # The ranking is deterministic and prioritizes stationary training series.
    # Ranking criteria (in order):
    #
    # 1. both_stationary: True when both ADF and KPSS indicate stationarity,
    # 2. higher KPSS p-value,
    # 3. lower differencing order,
    # 4. lower ADF p-value,
    # 5. prefer log1p when all previous criteria tie.
    #
    #Args:
    #     candidate_df: DataFrame returned by evaluate_candidates containing at
    #         least transformation fields and stationarity-test columns.
    #
    #Returns:
    #    The selected TransformConfig.
    #
    #Raises:
    #    ValueError: If candidate_df does not include required columns.
    #

    required_cols = {
        "use_log1p",
        "power_exponent",
        "diff_order",
        "scale_method",
        "adf_pvalue_train",
        "kpss_pvalue_train",
        "adf_stationary_train",
        "kpss_stationary_train",
    }
    missing = sorted(required_cols.difference(candidate_df.columns))
    if missing:
        raise ValueError(f"candidate_df missing required columns: {missing}")

    ranked = candidate_df.copy()
    ranked["both_stationary"] = (
        ranked["adf_stationary_train"].astype(bool)
        & ranked["kpss_stationary_train"].astype(bool)
    )
    ranked = ranked.sort_values(
        by=[
            "both_stationary",
            "kpss_pvalue_train",
            "diff_order",
            "adf_pvalue_train",
            "use_log1p",
        ],
        ascending=[False, False, True, True, False],
    )

    best = ranked.iloc[0]
    return TransformConfig(
        use_log1p=bool(best["use_log1p"]),
        power_exponent=None if pd.isna(best["power_exponent"]) else float(best["power_exponent"]),
        diff_order=int(best["diff_order"]),
        scale_method=str(best["scale_method"]),
    )


def select_best_transform_config_for_profile(
    candidate_df: pd.DataFrame,
    profile: PreprocessingProfile,
) -> TransformConfig:
    """Select the best transform using profile-specific ranking criteria."""

    required_cols = {
        "use_log1p",
        "power_exponent",
        "diff_order",
        "scale_method",
        "adf_pvalue_train",
        "kpss_pvalue_train",
        "adf_stationary_train",
        "kpss_stationary_train",
    }
    missing = sorted(required_cols.difference(candidate_df.columns))
    if missing:
        raise ValueError(f"candidate_df missing required columns: {missing}")

    ranked = candidate_df.copy()
    ranked["both_stationary"] = (
        ranked["adf_stationary_train"].astype(bool)
        & ranked["kpss_stationary_train"].astype(bool)
    )
    ranked["uses_scaling"] = ranked["scale_method"].astype(str).str.lower() != "none"

    # Profile-specific ordering priorities:
    # - statistical: maximize stationarity and keep transformations parsimonious.
    # - ml: prioritize stable transforms with low differencing and optional scaling.
    # - neural: prioritize scaled pipelines, then stationarity and smooth transforms.
    if profile == "statistical":
        ranked = ranked.sort_values(
            by=[
                "both_stationary",
                "kpss_pvalue_train",
                "diff_order",
                "adf_pvalue_train",
                "use_log1p",
            ],
            ascending=[False, False, True, True, False],
        )
    elif profile == "ml":
        ranked = ranked.sort_values(
            by=[
                "both_stationary",
                "diff_order",
                "uses_scaling",
                "kpss_pvalue_train",
                "adf_pvalue_train",
                "use_log1p",
            ],
            ascending=[False, True, False, False, True, False],
        )
    elif profile == "neural":
        ranked = ranked.sort_values(
            by=[
                "uses_scaling",
                "both_stationary",
                "diff_order",
                "kpss_pvalue_train",
                "adf_pvalue_train",
                "use_log1p",
            ],
            ascending=[False, False, True, False, True, False],
        )
    else:
        raise ValueError(f"Unsupported preprocessing profile: {profile}")

    best = ranked.iloc[0]
    return TransformConfig(
        use_log1p=bool(best["use_log1p"]),
        power_exponent=None if pd.isna(best["power_exponent"]) else float(best["power_exponent"]),
        diff_order=int(best["diff_order"]),
        scale_method=str(best["scale_method"]),
    )

def save_selected_preprocessing_config(config: PreprocessingConfig, output_path: Path) -> Path:
    # Persist a preprocessing configuration to JSON.
    #
    # Args:
    #     config: PreprocessingConfig to serialize.
    #     output_path: Destination JSON path. Parent directories are created
    #         automatically if missing.
    #
    # Returns:
    #     The resolved output path used for writing.
    #

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(config)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path

def load_selected_preprocessing_config(input_path: Path) -> PreprocessingConfig:
    # Load a preprocessing configuration from a JSON artifact.
    #
    # Args:
    #     input_path: Path to a JSON file previously produced by
    #         save_selected_preprocessing_config.
    #
    # Returns:
    #     A reconstructed PreprocessingConfig instance.
    #
    # Notes:
    #     - The JSON file is expected to contain at least the keys for split,
    #       transform, and outliers.
    #     - Missing optional keys fall back to safe defaults:
    #       run_shapiro=False and shapiro_max_n=5000.
    #

    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    split_cfg = payload.get("split", {})
    transform_cfg = payload.get("transform", {})
    outlier_cfg = payload.get("outliers", {})

    return PreprocessingConfig(
        split=SplitConfig(**split_cfg),
        transform=TransformConfig(**transform_cfg),
        outliers=OutlierConfig(**outlier_cfg),
        run_shapiro=bool(payload.get("run_shapiro", False)),
        shapiro_max_n=int(payload.get("shapiro_max_n", 5000)),
    )

def prepare_preprocessing_from_candidates(
    series: pd.Series,
    base_config: PreprocessingConfig | None = None,
    candidate_cfgs: Iterable[TransformConfig] = DEFAULT_PREPROCESSING_CANDIDATES,
) -> tuple[TimeSeriesPreprocessor, dict, pd.DataFrame, PreprocessingConfig]:
    # Build and run preprocessing using automatic candidate selection.
    #
    # Workflow:
    #     1) evaluate candidate transform configurations,
    #     2) select best transform,
    #     3) merge it into a final PreprocessingConfig,
    #     4) run preprocess() with the selected configuration.
    #
    # Args:
    #     series: Raw univariate time series to preprocess.
    #     base_config: Base configuration carrying split/outlier/test settings.
    #         If None, a default PreprocessingConfig(run_shapiro=True) is used.
    #     candidate_cfgs: Iterable of TransformConfig candidates to evaluate.
    #
    # Returns:
    #     A 4-tuple:
    #         - preproc: configured TimeSeriesPreprocessor,
    #         - preproc_output: output dictionary returned by preprocess(),
    #         - candidate_df: candidate evaluation table,
    #         - selected_cfg: final selected PreprocessingConfig.
    #

    if base_config is None:
        base_config = PreprocessingConfig(run_shapiro=True)

    selector = TimeSeriesPreprocessor(series, base_config)
    candidate_df = selector.evaluate_candidates(candidate_cfgs)
    best_transform = select_best_transform_config(candidate_df)

    selected_cfg = PreprocessingConfig(
        split=base_config.split,
        transform=best_transform,
        outliers=base_config.outliers,
        run_shapiro=base_config.run_shapiro,
        shapiro_max_n=base_config.shapiro_max_n,
    )

    preproc = TimeSeriesPreprocessor(series, selected_cfg)
    preproc_output = preproc.preprocess()

    return preproc, preproc_output, candidate_df, selected_cfg


def prepare_preprocessing_for_profile(
    series: pd.Series,
    profile: PreprocessingProfile,
    base_config: PreprocessingConfig | None = None,
) -> tuple[TimeSeriesPreprocessor, dict, pd.DataFrame, PreprocessingConfig]:
    """Build and run preprocessing using candidates tied to a model profile."""

    if base_config is None:
        base_config = PreprocessingConfig(run_shapiro=True)

    selector = TimeSeriesPreprocessor(series, base_config)
    candidate_cfgs = get_preprocessing_candidates(profile)
    candidate_df = selector.evaluate_candidates(candidate_cfgs)
    best_transform = select_best_transform_config_for_profile(candidate_df, profile)

    selected_cfg = PreprocessingConfig(
        split=base_config.split,
        transform=best_transform,
        outliers=base_config.outliers,
        run_shapiro=base_config.run_shapiro,
        shapiro_max_n=base_config.shapiro_max_n,
    )

    preproc = TimeSeriesPreprocessor(series, selected_cfg)
    preproc_output = preproc.preprocess()

    return preproc, preproc_output, candidate_df, selected_cfg