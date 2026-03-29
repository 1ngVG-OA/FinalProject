"""Orchestratore dei modelli statistici per lo Step 3.

Il modulo delega grid search, refit e valutazione SARIMA ai componenti dedicati,
restituendo un output unico con metriche, forecast e diagnostica residui.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_config import (
    StatisticalStepConfig,
    validate_original_series,
    validate_split,
)
from .evaluation import (
    build_forecast_table,
    build_residuals_table,
    build_summary_table,
    select_winner,
)
from .plotting import save_statistical_plots
from .sarima import SarimaRunner


class StatisticalModelRunner:
    """Esegue lo Step 3 SARIMA seguendo un protocollo condiviso.

    Internamente delega la ricerca iperparametri e il refit finale a
    :class:`SarimaRunner`.
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

        self._sarima_runner = SarimaRunner(
            train, validation, test,
            config=self.config,
            original_series=original_series,
            use_log1p=use_log1p,
            diff_order=diff_order,
        )

    def run(self) -> dict[str, Any]:
        """Esegue l'intero workflow Step 3 e restituisce tutti gli artifact."""

        sarima_grid_df, sarima_best = self._sarima_runner.fit_sarima_grid()

        sarima_val_pred = pd.Series(
            np.asarray(sarima_best["fit"].forecast(len(self.validation))),
            index=self.validation.index,
        )

        sarima_final = self._sarima_runner.refit(sarima_best["cfg"])

        sarima_test_pred = pd.Series(
            np.asarray(sarima_final.forecast(len(self.test))), index=self.test.index
        )

        summary = build_summary_table(
            validation=self.validation,
            test=self.test,
            sarima_best=sarima_best,
            sarima_val_pred=sarima_val_pred,
            sarima_test_pred=sarima_test_pred,
            sarima_final=sarima_final,
            sarima_orig_context=self._sarima_runner._orig_context,
            diff_order=self.diff_order,
            train_validation_len=len(self.train_validation),
        )

        winner, best_row = select_winner(summary)

        sarima_resid = (
            self.validation
            - pd.Series(np.asarray(sarima_val_pred), index=self.validation.index)
        ).dropna()

        residual_diagnostics = build_residuals_table(
            "sarima", sarima_resid, self.config.ljung_box_lags
        )

        forecast_table = build_forecast_table(
            validation=self.validation,
            test=self.test,
            sarima_val_pred=sarima_val_pred,
            sarima_test_pred=sarima_test_pred,
        )

        return {
            "sarima_grid": sarima_grid_df,
            "summary": summary,
            "winner": winner,
            "winner_params": dict(best_row),
            "residual_diagnostics": residual_diagnostics,
            "forecast_table": forecast_table,
            "validation_actual": self.validation,
            "test_actual": self.test,
            "sarima_val_pred": pd.Series(
                np.asarray(sarima_val_pred), index=self.validation.index
            ),
            "sarima_test_pred": pd.Series(
                np.asarray(sarima_test_pred), index=self.test.index
            ),
            "sarima_validation_residuals": sarima_resid,
            "original_series": self.original_series,
            "use_log1p": self.use_log1p,
            "diff_order": self.diff_order,
        }

    @staticmethod
    def save_plots(output: dict[str, Any], out_dir: Path, suffix: str | None = None) -> dict[str, Path]:
        """Salva i grafici Step 3 tramite il modulo di plotting dedicato."""
        return save_statistical_plots(output, out_dir, suffix=suffix)
