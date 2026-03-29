"""Terza fase del progetto: ricerca e refit del modello SARIMA.

Questo modulo contiene la logica di generazione candidati, valutazione su validation
e rifit finale su train+validation per il modello statistico SARIMA.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .model_config import (
    StatisticalStepConfig,
    _aicc,
    build_original_scale_context,
    compute_metrics,
    validate_original_series,
    validate_split,
    validation_original_metrics,
)


class SarimaRunner:
    """Runner SARIMA su split fisso train/validation/test.

    La classe esegue una grid search sulle configurazioni candidate, seleziona
    il miglior modello in base alle metriche di validazione e permette il refit
    finale sul blocco train+validation.
    """

    def __init__(
        self,
        train: pd.Series,
        validation: pd.Series,
        test: pd.Series,
        config: StatisticalStepConfig | None = None,
        original_series: pd.Series | None = None,
        use_log1p: bool = False,
        diff_order: int = 0,
    ) -> None:
        self.config = config or StatisticalStepConfig()
        self.train = validate_split(train, "train")
        self.validation = validate_split(validation, "validation")
        self.test = validate_split(test, "test")
        self.train_validation = pd.concat([self.train, self.validation])
        self.original_series = validate_original_series(original_series)
        self.use_log1p = bool(use_log1p)
        self.diff_order = int(diff_order)
        self._orig_context = build_original_scale_context(
            self.original_series,
            self.use_log1p,
            self.diff_order,
            self.validation.index.min(),
        )

    # ------------------------------------------------------------------
    # Generazione configurazioni candidate
    # ------------------------------------------------------------------

    def _sarima_candidates(self) -> list[dict[str, Any]]:
        seasonal_period = max(1, int(self.config.seasonal_period))
        candidates: list[dict[str, Any]] = []

        for p, d, q in product(
            self.config.p_values, self.config.d_values, self.config.q_values
        ):
            if seasonal_period > 1:
                for ps, ds, qs in product(
                    self.config.p_seasonal_values,
                    self.config.d_seasonal_values,
                    self.config.q_seasonal_values,
                ):
                    candidates.append(
                        {
                            "order": (p, d, q),
                            "seasonal_order": (ps, ds, qs, seasonal_period),
                        }
                    )
            else:
                candidates.append(
                    {"order": (p, d, q), "seasonal_order": (0, 0, 0, 0)}
                )

        return candidates

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    def fit_sarima_grid(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Esegue la grid search SARIMA valutando ogni candidato su validation.

        Restituisce:
        - results: tabella completa dei candidati ordinata per ranking
          (rank_rmse_val, rank_abs_mbe_val, aicc).
        - best: dizionario con fit/configurazione/riga e ranking del miglior
          candidato selezionato.
        """
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for cfg in self._sarima_candidates():
            try:
                model = SARIMAX(
                    self.train,
                    order=cfg["order"],
                    seasonal_order=cfg["seasonal_order"],
                    enforce_stationarity=self.config.enforce_stationarity,
                    enforce_invertibility=self.config.enforce_invertibility,
                )
                fit = model.fit(disp=False, maxiter=self.config.maxiter)
                pred_val = pd.Series(
                    np.asarray(fit.forecast(steps=len(self.validation))),
                    index=self.validation.index,
                )

                metrics = compute_metrics(self.validation, pred_val)
                metrics_orig = validation_original_metrics(
                    pred_val, self._orig_context, self.diff_order
                )
                k = int(fit.params.shape[0])
                aic = float(fit.aic)

                row = {
                    "order": str(cfg["order"]),
                    "seasonal_order": str(cfg["seasonal_order"]),
                    "rmse_val": metrics["rmse"],
                    "mae_val": metrics["mae"],
                    "mape_val": metrics["mape"],
                    "mbe_val": metrics["mbe"],
                    "abs_mbe_val": metrics["abs_mbe"],
                    "rmse_val_orig": np.nan if metrics_orig is None else metrics_orig["rmse"],
                    "mae_val_orig": np.nan if metrics_orig is None else metrics_orig["mae"],
                    "mape_val_orig": np.nan if metrics_orig is None else metrics_orig["mape"],
                    "mbe_val_orig": np.nan if metrics_orig is None else metrics_orig["mbe"],
                    "abs_mbe_val_orig": np.nan if metrics_orig is None else metrics_orig["abs_mbe"],
                    "aic": aic,
                    "aicc": _aicc(aic, n=len(self.train), k=k),
                    "k_params": k,
                }
                rows.append(row)

                rank_rmse_now = (
                    row["rmse_val_orig"]
                    if not pd.isna(row["rmse_val_orig"])
                    else row["rmse_val"]
                )
                rank_abs_mbe_now = (
                    row["abs_mbe_val_orig"]
                    if not pd.isna(row["abs_mbe_val_orig"])
                    else row["abs_mbe_val"]
                )

                best = _update_best(
                    best, fit, cfg, row, rank_rmse_now, rank_abs_mbe_now
                )

            except Exception:
                continue

        if not rows or best is None:
            raise RuntimeError(
                "SARIMA grid search failed for all candidate configurations"
            )

        results = pd.DataFrame(rows)
        results = results.assign(
            rank_rmse_val=results["rmse_val_orig"].fillna(results["rmse_val"])
        )
        results = results.assign(
            rank_abs_mbe_val=results["abs_mbe_val_orig"].fillna(
                results["abs_mbe_val"]
            )
        )
        results = results.sort_values(
            ["rank_rmse_val", "rank_abs_mbe_val", "aicc"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        return results, best

    # ------------------------------------------------------------------
    # Refit su train+validation
    # ------------------------------------------------------------------

    def refit(self, cfg: dict[str, Any]) -> Any:
        """Esegue il refit della configurazione SARIMA scelta su train+validation."""
        model = SARIMAX(
            self.train_validation,
            order=cfg["order"],
            seasonal_order=cfg["seasonal_order"],
            enforce_stationarity=self.config.enforce_stationarity,
            enforce_invertibility=self.config.enforce_invertibility,
        )
        return model.fit(disp=False, maxiter=self.config.maxiter)


# ---------------------------------------------------------------------------
# Helper privati
# ---------------------------------------------------------------------------

def _update_best(
    best: dict[str, Any] | None,
    fit: Any,
    cfg: dict[str, Any],
    row: dict[str, Any],
    rank_rmse_now: float,
    rank_abs_mbe_now: float,
) -> dict[str, Any]:
    """Aggiorna il best candidate usando ranking RMSE -> abs_MBE -> AICc."""
    entry = {
        "fit": fit,
        "cfg": cfg,
        "row": row,
        "rank_rmse": rank_rmse_now,
        "rank_abs_mbe": rank_abs_mbe_now,
    }
    if best is None:
        return entry

    rmse_best = best["rank_rmse"]
    if rank_rmse_now < rmse_best - 1e-12:
        return entry
    if abs(rank_rmse_now - rmse_best) <= 1e-12:
        mbe_best = best["rank_abs_mbe"]
        if rank_abs_mbe_now < mbe_best - 1e-12:
            return entry
        if abs(rank_abs_mbe_now - mbe_best) <= 1e-12 and row["aicc"] < best["row"]["aicc"]:
            return entry
    return best
