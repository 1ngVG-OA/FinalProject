"""Utilities to auto-select and persist preprocessing decisions.

This module turns Step 2 preprocessing into a single source of truth:
- evaluate candidate transform configs,
- select the best candidate with deterministic ranking rules,
- persist/load the selected configuration as a JSON artifact,
- build a preprocessor already aligned with the selected settings.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable
import json

import pandas as pd

from Project.preprocessing.time_series_preprocessor import (
    OutlierConfig,
    PreprocessingConfig,
    SplitConfig,
    TimeSeriesPreprocessor,
    TransformConfig,
)


DEFAULT_PREPROCESSING_CANDIDATES: tuple[TransformConfig, ...] = (
    TransformConfig(use_log1p=False, diff_order=0, scale_method="none"),
    TransformConfig(use_log1p=False, diff_order=1, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=1, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=2, scale_method="none"),
)


def select_best_transform_config(candidate_df: pd.DataFrame) -> TransformConfig:
    """Select the best transform config from candidate stationarity report.

    Ranking policy:
    1) prefer rows where ADF and KPSS both indicate stationarity,
    2) then maximize KPSS p-value,
    3) then prefer lower differencing order,
    4) then minimize ADF p-value,
    5) deterministic tie-break by log flag.
    """

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


def save_selected_preprocessing_config(config: PreprocessingConfig, output_path: Path) -> Path:
    """Persist selected preprocessing config as JSON artifact."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(config)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_selected_preprocessing_config(input_path: Path) -> PreprocessingConfig:
    """Load preprocessing config artifact and rebuild dataclass structure."""

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
    """Run candidate search and return preprocessor output with selected config."""

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