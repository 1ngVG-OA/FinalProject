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

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from Project.models.statistical.model_config import (
    _aicc,
    build_original_scale_context,
    compute_metrics,
    validation_original_metrics,
)
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


# Point-2 backtest defaults (reduced SARIMA grid + composite score weight).
BACKTEST_P_VALUES: tuple[int, ...] = (0, 1)
BACKTEST_D_VALUES: tuple[int, ...] = (0,)
BACKTEST_Q_VALUES: tuple[int, ...] = (0, 1)
BACKTEST_COMPOSITE_LAMBDA: float = 0.5
BACKTEST_MAXITER: int = 120
DRIFT_GUARD_MAX_ABS_MBE_ORIG: float = 2000.0


def _candidate_bias_penalty_orig(
    original_series: pd.Series,
    val_index: pd.Index,
    cfg: TransformConfig,
) -> float:
    """Estimate original-scale abs bias for a zero-forecast baseline.

    The baseline forecast is identically zero in transformed space. This
    highlights transformation-induced drift during inverse mapping,
    especially for second differencing.
    """
    if val_index.empty:
        return float("nan")

    # No inverse mapping available for non-log or unsupported differencing.
    if not cfg.use_log1p or cfg.diff_order not in (1, 2):
        return float("nan")

    raw = pd.to_numeric(original_series, errors="coerce").dropna().astype(float)
    if raw.empty:
        return float("nan")

    val_start = val_index.min()
    x_log = np.log1p(raw)
    pred_zero = pd.Series(0.0, index=val_index)

    try:
        if cfg.diff_order == 1:
            seed_log = float(x_log[x_log.index < val_start].iloc[-1])
            log_pred = seed_log + pred_zero.cumsum()
            pred_orig = np.expm1(log_pred)
        else:
            x_d1 = x_log.diff().dropna()
            seed_d1 = float(x_d1[x_d1.index < val_start].iloc[-1])
            seed_log = float(x_log[x_log.index < val_start].iloc[-1])
            d1_pred = seed_d1 + pred_zero.cumsum()
            log_pred = seed_log + d1_pred.cumsum()
            pred_orig = np.expm1(log_pred)
    except Exception:
        return float("nan")

    true_orig = raw.reindex(val_index)
    aligned = pd.concat(
        [
            pd.Series(true_orig, index=val_index, name="y_true"),
            pd.Series(pred_orig, index=val_index, name="y_pred"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return float("nan")

    return float(abs((aligned["y_pred"] - aligned["y_true"]).mean()))


def _run_candidate_stat_backtest(
    train: pd.Series,
    validation: pd.Series,
    original_series: pd.Series,
    cfg: TransformConfig,
) -> dict[str, object]:
    """Run a reduced-grid SARIMA validation backtest for one candidate."""

    best_row: dict[str, object] | None = None

    orig_context = build_original_scale_context(
        original_series=original_series,
        use_log1p=cfg.use_log1p,
        diff_order=cfg.diff_order,
        val_start=validation.index.min(),
    )

    for p in BACKTEST_P_VALUES:
        for d in BACKTEST_D_VALUES:
            for q in BACKTEST_Q_VALUES:
                order = (p, d, q)
                seasonal_order = (0, 0, 0, 0)
                try:
                    model = SARIMAX(
                        train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit = model.fit(disp=False, maxiter=BACKTEST_MAXITER)
                    pred_val = pd.Series(
                        np.asarray(fit.forecast(steps=len(validation))),
                        index=validation.index,
                    )

                    metrics = compute_metrics(validation, pred_val)
                    metrics_orig = validation_original_metrics(
                        pred_val,
                        orig_context,
                        cfg.diff_order,
                    )

                    has_orig_metrics = metrics_orig is not None
                    rmse_rank = (
                        float(metrics_orig["rmse"])
                        if has_orig_metrics
                        else float("inf")
                    )
                    abs_mbe_rank = (
                        float(metrics_orig["abs_mbe"])
                        if has_orig_metrics
                        else float("inf")
                    )
                    score = rmse_rank + BACKTEST_COMPOSITE_LAMBDA * abs_mbe_rank

                    row = {
                        "best_order": str(order),
                        "best_seasonal_order": str(seasonal_order),
                        "aicc_best": _aicc(float(fit.aic), n=len(train), k=int(fit.params.shape[0])),
                        "rmse_val_backtest": metrics["rmse"],
                        "mae_val_backtest": metrics["mae"],
                        "mbe_val_backtest": metrics["mbe"],
                        "abs_mbe_val_backtest": metrics["abs_mbe"],
                        "rmse_val_orig_backtest": np.nan if metrics_orig is None else metrics_orig["rmse"],
                        "mae_val_orig_backtest": np.nan if metrics_orig is None else metrics_orig["mae"],
                        "mbe_val_orig_backtest": np.nan if metrics_orig is None else metrics_orig["mbe"],
                        "abs_mbe_val_orig_backtest": np.nan if metrics_orig is None else metrics_orig["abs_mbe"],
                        "has_orig_backtest": has_orig_metrics,
                        "rank_rmse_backtest": rmse_rank,
                        "rank_abs_mbe_backtest": abs_mbe_rank,
                        "score_backtest": score,
                    }

                    if (
                        best_row is None
                        or row["score_backtest"] < best_row["score_backtest"] - 1e-12
                        or (
                            abs(row["score_backtest"] - best_row["score_backtest"]) <= 1e-12
                            and row["rank_rmse_backtest"] < best_row["rank_rmse_backtest"] - 1e-12
                        )
                        or (
                            abs(row["score_backtest"] - best_row["score_backtest"]) <= 1e-12
                            and abs(row["rank_rmse_backtest"] - best_row["rank_rmse_backtest"]) <= 1e-12
                            and row["aicc_best"] < best_row["aicc_best"]
                        )
                    ):
                        best_row = row
                except Exception:
                    continue

    if best_row is None:
        return {
            "best_order": None,
            "best_seasonal_order": None,
            "aicc_best": np.nan,
            "rmse_val_backtest": np.nan,
            "mae_val_backtest": np.nan,
            "mbe_val_backtest": np.nan,
            "abs_mbe_val_backtest": np.nan,
            "rmse_val_orig_backtest": np.nan,
            "mae_val_orig_backtest": np.nan,
            "mbe_val_orig_backtest": np.nan,
            "abs_mbe_val_orig_backtest": np.nan,
            "rank_rmse_backtest": np.nan,
            "rank_abs_mbe_backtest": np.nan,
            "score_backtest": np.nan,
        }

    return best_row

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
    ranked["rank_abs_mbe_zero_val"] = ranked.get("abs_mbe_zero_val", pd.Series(np.nan, index=ranked.index))
    ranked["rank_abs_mbe_zero_val_orig"] = ranked.get("abs_mbe_zero_val_orig", pd.Series(np.nan, index=ranked.index))
    ranked["rank_abs_mbe_zero_val"] = ranked["rank_abs_mbe_zero_val"].fillna(float("inf"))
    ranked["rank_abs_mbe_zero_val_orig"] = ranked["rank_abs_mbe_zero_val_orig"].fillna(float("inf"))
    ranked["is_diff2"] = ranked["diff_order"].astype(int).ge(2)
    ranked["score_backtest"] = ranked.get("score_backtest", pd.Series(np.nan, index=ranked.index))
    ranked["rank_score_backtest"] = ranked["score_backtest"].fillna(float("inf"))
    ranked["abs_mbe_val_orig_backtest"] = ranked.get(
        "abs_mbe_val_orig_backtest", pd.Series(np.nan, index=ranked.index)
    )
    ranked["drift_guard_excluded"] = (
        ranked["is_diff2"]
        & ranked["abs_mbe_val_orig_backtest"].notna()
        & ranked["abs_mbe_val_orig_backtest"].gt(DRIFT_GUARD_MAX_ABS_MBE_ORIG)
    )
    ranked["drift_guard_rank"] = ranked["drift_guard_excluded"].astype(int)

    # Profile-specific ordering priorities:
    # - statistical: maximize stationarity and keep transformations parsimonious.
    # - ml: prioritize stable transforms with low differencing and optional scaling.
    # - neural: prioritize scaled pipelines, then stationarity and smooth transforms.
    if profile == "statistical":
        ranked = ranked.sort_values(
            by=[
                "drift_guard_rank",
                "rank_score_backtest",
                "both_stationary",
                "kpss_pvalue_train",
                "is_diff2",
                "rank_abs_mbe_zero_val_orig",
                "rank_abs_mbe_zero_val",
                "diff_order",
                "adf_pvalue_train",
                "use_log1p",
            ],
            ascending=[True, True, False, False, True, True, True, True, True, False],
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

    # Add bias/drift proxy metrics from a zero-forecast baseline on validation.
    bias_metrics = []
    backtest_metrics = []
    for cfg in candidate_cfgs:
        tmp = TimeSeriesPreprocessor(
            series,
            PreprocessingConfig(
                split=base_config.split,
                transform=cfg,
                outliers=base_config.outliers,
                run_shapiro=base_config.run_shapiro,
                shapiro_max_n=base_config.shapiro_max_n,
            ),
        )
        tmp_out = tmp.preprocess()
        val_series = tmp_out["splits"]["val"]
        abs_mbe_zero_val = float(abs(pd.Series(0.0, index=val_series.index).sub(val_series).mean()))
        abs_mbe_zero_val_orig = _candidate_bias_penalty_orig(series, val_series.index, cfg)
        bias_metrics.append(
            {
                "use_log1p": cfg.use_log1p,
                "power_exponent": cfg.power_exponent,
                "diff_order": cfg.diff_order,
                "scale_method": cfg.scale_method,
                "abs_mbe_zero_val": abs_mbe_zero_val,
                "abs_mbe_zero_val_orig": abs_mbe_zero_val_orig,
            }
        )

        # Point 2: mini empirical backtest on statistical profile only.
        if profile == "statistical":
            backtest_row = _run_candidate_stat_backtest(
                train=tmp_out["splits"]["train"],
                validation=val_series,
                original_series=series,
                cfg=cfg,
            )
            backtest_row.update(
                {
                    "use_log1p": cfg.use_log1p,
                    "power_exponent": cfg.power_exponent,
                    "diff_order": cfg.diff_order,
                    "scale_method": cfg.scale_method,
                    "backtest_lambda": BACKTEST_COMPOSITE_LAMBDA,
                }
            )
            backtest_metrics.append(backtest_row)

    if bias_metrics:
        bias_df = pd.DataFrame(bias_metrics)
        candidate_df = candidate_df.merge(
            bias_df,
            on=["use_log1p", "power_exponent", "diff_order", "scale_method"],
            how="left",
        )

    if backtest_metrics:
        backtest_df = pd.DataFrame(backtest_metrics)
        candidate_df = candidate_df.merge(
            backtest_df,
            on=["use_log1p", "power_exponent", "diff_order", "scale_method"],
            how="left",
        )
        candidate_df["drift_guard_threshold"] = DRIFT_GUARD_MAX_ABS_MBE_ORIG
        candidate_df["drift_guard_excluded"] = (
            candidate_df["diff_order"].astype(int).ge(2)
            & candidate_df["abs_mbe_val_orig_backtest"].notna()
            & candidate_df["abs_mbe_val_orig_backtest"].gt(DRIFT_GUARD_MAX_ABS_MBE_ORIG)
        )
        candidate_df["rank_backtest"] = candidate_df["score_backtest"].rank(method="dense", ascending=True)

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