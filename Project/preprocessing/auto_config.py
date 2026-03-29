# Configurazione e logica per la selezione automatica delle trasformazioni di preprocessing.
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

# Definizione di un tipo letterale per i profili di preprocessing, che categorizza le trasformazioni candidate in base alla famiglia di modelli target (statistical, ml, neural).
PreprocessingProfile = Literal["statistical", "ml", "neural"]

# Trasformazioni candidate per il profilo "statistical", che include combinazioni di log1p, differenziazione e scaling parsimonioso, con un focus sulla stazionarietà.
STATISTICAL_PREPROCESSING_CANDIDATES: tuple[TransformConfig, ...] = (
    TransformConfig(use_log1p=False, diff_order=0, scale_method="none"),
    TransformConfig(use_log1p=False, diff_order=1, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=1, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=2, scale_method="none"),
)

# Trasformazioni candidate per il profilo "ml", che include combinazioni di log1p, differenziazione e scaling più aggressive, con un focus sulla preparazione dei dati per modelli di machine learning.
ML_PREPROCESSING_CANDIDATES: tuple[TransformConfig, ...] = (
    TransformConfig(use_log1p=False, diff_order=0, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=0, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=1, scale_method="none"),
    TransformConfig(use_log1p=True, diff_order=1, scale_method="standard"),
    TransformConfig(use_log1p=True, diff_order=1, scale_method="minmax"),
)

# Trasformazioni candidate per il profilo "neural", che include combinazioni di log1p, differenziazione e scaling standard o min-max, con un focus sulla preparazione dei dati per modelli neurali.
NEURAL_PREPROCESSING_CANDIDATES: tuple[TransformConfig, ...] = (
    TransformConfig(use_log1p=False, diff_order=0, scale_method="standard"),
    TransformConfig(use_log1p=True, diff_order=0, scale_method="standard"),
    TransformConfig(use_log1p=True, diff_order=1, scale_method="standard"),
    TransformConfig(use_log1p=False, diff_order=0, scale_method="minmax"),
    TransformConfig(use_log1p=True, diff_order=0, scale_method="minmax"),
)

# Mappatura dei profili di preprocessing alle rispettive trasformazioni candidate, che consente di selezionare automaticamente le configurazioni da valutare in base al tipo di modello target.
PREPROCESSING_CANDIDATES_BY_PROFILE: dict[PreprocessingProfile, tuple[TransformConfig, ...]] = {
    "statistical": STATISTICAL_PREPROCESSING_CANDIDATES,
    "ml": ML_PREPROCESSING_CANDIDATES,
    "neural": NEURAL_PREPROCESSING_CANDIDATES,
}


# Parametri e configurazioni per il backtest di validazione SARIMA utilizzato come proxy per valutare la stabilità e il bias indotto dalle trasformazioni candidate, con l'obiettivo di identificare potenziali problemi di drift durante l'inversione delle trasformazioni.
BACKTEST_P_VALUES: tuple[int, ...] = (0, 1)
BACKTEST_D_VALUES: tuple[int, ...] = (0,)
BACKTEST_Q_VALUES: tuple[int, ...] = (0, 1)
BACKTEST_COMPOSITE_LAMBDA: float = 0.5
BACKTEST_MAXITER: int = 120
DRIFT_GUARD_MAX_ABS_MBE_ORIG: float = 2000.0

# Funzione che penalizza le configurazioni che mostrano un alto bias di previsione (mbe) sulla serie originale durante un backtest SARIMA, escludendole dalla selezione finale per il profilo "statistical" al fine di mitigare potenziali problemi di drift quando si invertono le trasformazioni.
def _candidate_bias_penalty_orig(
    original_series: pd.Series,
    val_index: pd.Index,
    cfg: TransformConfig,
) -> float:
    
    if val_index.empty:
        return float("nan")
    # La penalizzazione è applicata solo se la trasformazione include log1p e differenziazione, poiché sono le più suscettibili di introdurre bias di drift quando si invertono le trasformazioni su serie non stazionarie o con trend marcati. Se queste condizioni non sono soddisfatte, restituisco NaN per indicare che il calcolo del bias non è applicabile o significativo per quella configurazione.
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

# Esegue un backtest di validazione SARIMA ridotto su una suddivisione train/validation, 
# valutando le previsioni sulla serie originale per identificare configurazioni candidate che introducono bias di previsione (mbe) elevato.
def _run_candidate_stat_backtest(
    train: pd.Series,
    validation: pd.Series,
    original_series: pd.Series,
    cfg: TransformConfig,
) -> dict[str, object]:

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

# Il set di trasformazioni candidate di default utilizzato quando non si specifica un profilo, che attualmente coincide con il profilo "statistical" 
# per garantire un focus sulla stazionarietà e la parsimonia delle trasformazioni.
DEFAULT_PREPROCESSING_CANDIDATES = STATISTICAL_PREPROCESSING_CANDIDATES

# Funzione che restituisce le configurazioni di trasformazione candidate per un dato profilo di preprocessing, 
# consentendo di selezionare automaticamente le trasformazioni da valutare in base alla famiglia di modelli target.
def get_preprocessing_candidates(profile: PreprocessingProfile) -> tuple[TransformConfig, ...]:
    
    return PREPROCESSING_CANDIDATES_BY_PROFILE[profile]

# Funzione che seleziona la migliore configurazione di trasformazione da un DataFrame di candidate, ordinando in base a criteri di stazionarietà e parsimonia.
def select_best_transform_config(candidate_df: pd.DataFrame) -> TransformConfig:

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

# Funzione che seleziona la migliore configurazione di trasformazione da un DataFrame di candidate, ordinando in base a criteri specifici per il profilo (statistical, ml, neural).
def select_best_transform_config_for_profile(
    candidate_df: pd.DataFrame,
    profile: PreprocessingProfile,
) -> TransformConfig:

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

    # Priorità di ordinamento per la selezione della configurazione migliore, che varia in base al profilo:
    # - statistical: priorità a configurazioni che superano il drift guard, poi stazionarietà, poi parsimonia (diff_order), poi p-value dei test di stazionarietà, poi penalizzazione del bias di previsione sulla serie originale, infine uso di log1p.
    # - ml: priorità a configurazioni stazionarie, poi parsimonia (diff_order), poi uso di scaling, poi p-value dei test di stazionarietà, infine uso di log1p.
    # - neural: priorità a configurazioni che usano scaling, poi stazionarietà, poi parsimonia (diff_order), poi p-value dei test di stazionarietà, infine uso di log1p.
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

# Funzione che salva la configurazione di preprocessing selezionata in un file JSON, consentendo di conservare un record della configurazione utilizzata per il preprocessing e facilitando la riproducibilità.
def save_selected_preprocessing_config(config: PreprocessingConfig, output_path: Path) -> Path:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(config)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path

# Funzione che carica una configurazione di preprocessing da un file JSON, consentendo di ripristinare una configurazione precedentemente salvata e utilizzarla per il preprocessing.
def load_selected_preprocessing_config(input_path: Path) -> PreprocessingConfig:

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

# Funzione che costruisce ed esegue il preprocessing utilizzando la selezione automatica dei candidati.
#
# Workflow:
#     1) valutare le configurazioni di trasformazione candidate,
#     2) selezionare la migliore trasformazione,
#     3) integrarla in una configurazione finale di PreprocessingConfig,
#     4) eseguire preprocess() con la configurazione selezionata.
def prepare_preprocessing_from_candidates(
    series: pd.Series,
    base_config: PreprocessingConfig | None = None,
    candidate_cfgs: Iterable[TransformConfig] = DEFAULT_PREPROCESSING_CANDIDATES,
) -> tuple[TimeSeriesPreprocessor, dict, pd.DataFrame, PreprocessingConfig]:
 
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

# Funzione che costruisce ed esegue il preprocessing utilizzando la selezione automatica dei candidati specifica per profilo, integrando metriche di bias di previsione e backtest di validazione per identificare configurazioni che potrebbero introdurre problemi di drift.
# Workflow:
#     1) valutare le configurazioni di trasformazione candidate per il profilo specificato,
#     2) arricchire la valutazione con metriche di bias di previsione su una baseline zero-forecast e con un backtest di validazione SARIMA sulla serie originale,
#     3) selezionare la migliore trasformazione utilizzando criteri specifici per il profilo (statistical, ml, neural) che tengono conto di stazionarietà, parsimonia, uso di scaling e penalizzazione del bias di previsione,
#     4) integrarla in una configurazione finale di PreprocessingConfig,
#     5) eseguire preprocess() con la configurazione selezionata.
def prepare_preprocessing_for_profile(
    series: pd.Series,
    profile: PreprocessingProfile,
    base_config: PreprocessingConfig | None = None,
) -> tuple[TimeSeriesPreprocessor, dict, pd.DataFrame, PreprocessingConfig]:

    if base_config is None:
        base_config = PreprocessingConfig(run_shapiro=True)

    selector = TimeSeriesPreprocessor(series, base_config)
    candidate_cfgs = get_preprocessing_candidates(profile)
    candidate_df = selector.evaluate_candidates(candidate_cfgs)

    # Point 1: calcolo di metriche di bias di previsione per ciascuna configurazione candidata, utilizzando una baseline zero-forecast sulla suddivisione di validazione, sia sulla serie trasformata che sulla serie originale (in quest'ultimo caso solo per configurazioni che includono log1p e differenziazione), al fine di identificare configurazioni che introducono un bias elevato e potrebbero causare problemi di drift quando si invertono le trasformazioni.
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

        # Point 2: esecuzione di un backtest di validazione SARIMA ridotto sulla suddivisione train/validation, valutando le previsioni sulla serie originale per identificare configurazioni candidate che introducono bias di previsione (mbe) elevato, al fine di escludere configurazioni che potrebbero causare problemi di drift quando si invertono le trasformazioni su serie non stazionarie o con trend marcati.
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