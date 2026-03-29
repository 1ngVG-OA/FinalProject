"""Runner dello Step 4 per modelli ML non neurali con feature lag."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from .features import (
    build_lagged_dataset,
    build_model_feature_matrix,
    select_features,
)
from .model_config import (
    MLStepConfig,
    compute_metrics,
    original_scale_metrics_for_segment,
)

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


class MLModelRunner:
    """Esegue selezione e valutazione dei modelli ML non neurali (Step 4)."""

    def __init__(
        self,
        train: pd.Series,
        validation: pd.Series,
        test: pd.Series,
        config: MLStepConfig | None = None,
        original_series: pd.Series | None = None,
        use_log1p: bool = False,
        diff_order: int = 0,
    ) -> None:
        self.config = config or MLStepConfig()
        self.train = MLStepConfig.validate_split(train, "train")
        self.validation = MLStepConfig.validate_split(validation, "validation")
        self.test = MLStepConfig.validate_split(test, "test")
        self.train_validation = pd.concat([self.train, self.validation])
        self.original_series = MLStepConfig.validate_original_series(original_series)
        self.use_log1p = bool(use_log1p)
        self.diff_order = int(diff_order)

    def _candidate_configs(self) -> list[dict[str, Any]]:
        # ------------------------------------------------------------------
        # Generazione configurazioni candidate
        # ------------------------------------------------------------------

        cfg = self.config
        rows: list[dict[str, Any]] = []

        for lookback, max_depth, min_leaf in product(
            cfg.lookback_values, cfg.dt_max_depth, cfg.dt_min_samples_leaf
        ):
            rows.append(
                {
                    "model": "decision_tree",
                    "lookback": int(lookback),
                    "params": {
                        "max_depth": max_depth,
                        "min_samples_leaf": int(min_leaf),
                        "random_state": cfg.random_state,
                    },
                }
            )

        for lookback, n_est, max_depth, min_leaf in product(
            cfg.lookback_values,
            cfg.rf_n_estimators,
            cfg.rf_max_depth,
            cfg.rf_min_samples_leaf,
        ):
            rows.append(
                {
                    "model": "random_forest",
                    "lookback": int(lookback),
                    "params": {
                        "n_estimators": int(n_est),
                        "max_depth": max_depth,
                        "min_samples_leaf": int(min_leaf),
                        "random_state": cfg.random_state,
                        "n_jobs": 1,
                    },
                }
            )

        for lookback, n_est, lr, max_depth in product(
            cfg.lookback_values,
            cfg.gbr_n_estimators,
            cfg.gbr_learning_rate,
            cfg.gbr_max_depth,
        ):
            rows.append(
                {
                    "model": "gradient_boosting",
                    "lookback": int(lookback),
                    "params": {
                        "n_estimators": int(n_est),
                        "learning_rate": float(lr),
                        "max_depth": int(max_depth),
                        "random_state": cfg.random_state,
                    },
                }
            )

        if cfg.use_xgboost and XGBRegressor is not None:
            for lookback, n_est, lr, max_depth, subs, col in product(
                cfg.lookback_values,
                cfg.xgb_n_estimators,
                cfg.xgb_learning_rate,
                cfg.xgb_max_depth,
                cfg.xgb_subsample,
                cfg.xgb_colsample_bytree,
            ):
                rows.append(
                    {
                        "model": "xgboost",
                        "lookback": int(lookback),
                        "params": {
                            "n_estimators": int(n_est),
                            "learning_rate": float(lr),
                            "max_depth": int(max_depth),
                            "subsample": float(subs),
                            "colsample_bytree": float(col),
                            "objective": "reg:squarederror",
                            "random_state": cfg.random_state,
                            "n_jobs": 1,
                        },
                    }
                )

        return rows

    @staticmethod
    def _build_estimator(model_name: str, params: dict[str, Any]) -> Any:
        if model_name == "decision_tree":
            return DecisionTreeRegressor(**params)
        if model_name == "random_forest":
            return RandomForestRegressor(**params)
        if model_name == "gradient_boosting":
            return GradientBoostingRegressor(**params)
        if model_name == "xgboost":
            if XGBRegressor is None:
                raise RuntimeError("xgboost is not available")
            return XGBRegressor(**params)
        raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def _iterative_forecast(
        model: Any,
        seed_series: pd.Series,
        horizon_index: pd.Index,
        lookback: int,
        selected_features: list[str],
    ) -> pd.Series:
        """Genera forecast ricorsivo multi-step sull'orizzonte richiesto."""
        history = list(seed_series.astype(float).to_numpy())
        preds = []

        for _ in range(len(horizon_index)):
            if len(history) < lookback:
                raise RuntimeError("Insufficient history for iterative forecast")
            lag_values = np.asarray(history[-lookback:][::-1], dtype=float)
            feat_map = {f"lag_{i}": lag_values[i - 1] for i in range(1, lookback + 1)}
            row = pd.DataFrame([[feat_map[c] for c in selected_features]], columns=selected_features)
            yhat = float(model.predict(row)[0])
            preds.append(yhat)
            history.append(yhat)

        return pd.Series(np.asarray(preds), index=horizon_index, name="pred")

    def run(self) -> dict[str, Any]:
        """Esegue il workflow completo Step 4 e restituisce tutti gli artifact."""
        grid_rows: list[dict[str, Any]] = []
        best_by_model: dict[str, dict[str, Any]] = {}
        prepared_by_lookback: dict[int, dict[str, Any]] = {}

        for cfg in self._candidate_configs():
            lookback = int(cfg["lookback"])
            if lookback not in prepared_by_lookback:
                lagged = build_lagged_dataset(self.train, self.validation, self.test, lookback)
                selected_features, fs_report = select_features(
                    lagged.X_train,
                    lagged.y_train,
                    method=self.config.feature_selection,
                    n_select=min(self.config.selected_feature_count, lookback),
                    random_state=self.config.random_state,
                )
                lagged_sel = build_model_feature_matrix(lagged, selected_features)
                prepared_by_lookback[lookback] = {
                    "lagged": lagged,
                    "lagged_sel": lagged_sel,
                    "selected_features": selected_features,
                    "fs_report": fs_report,
                }

            prepared = prepared_by_lookback[lookback]
            lagged = prepared["lagged"]
            lagged_sel = prepared["lagged_sel"]
            selected_features = prepared["selected_features"]
            fs_report = prepared["fs_report"]

            model = self._build_estimator(cfg["model"], cfg["params"])
            model.fit(lagged_sel.X_train, lagged_sel.y_train)

            val_pred = self._iterative_forecast(
                model=model,
                seed_series=self.train,
                horizon_index=self.validation.index,
                lookback=lookback,
                selected_features=selected_features,
            )

            val_metrics = compute_metrics(self.validation, val_pred)
            val_orig_metrics = original_scale_metrics_for_segment(
                val_pred,
                original_series=self.original_series,
                use_log1p=self.use_log1p,
                diff_order=self.diff_order,
            )
            rank_rmse = val_orig_metrics["rmse"] if val_orig_metrics is not None else val_metrics["rmse"]
            rank_abs_mbe = val_orig_metrics["abs_mbe"] if val_orig_metrics is not None else val_metrics["abs_mbe"]

            row = {
                "model": cfg["model"],
                "lookback": lookback,
                "feature_selection": self.config.feature_selection,
                "selected_features": ", ".join(selected_features),
                "n_selected_features": len(selected_features),
                "params": str(cfg["params"]),
                "rmse_val": val_metrics["rmse"],
                "mae_val": val_metrics["mae"],
                "mape_val": val_metrics["mape"],
                "mbe_val": val_metrics["mbe"],
                "abs_mbe_val": val_metrics["abs_mbe"],
                "rmse_val_orig": np.nan if val_orig_metrics is None else val_orig_metrics["rmse"],
                "mae_val_orig": np.nan if val_orig_metrics is None else val_orig_metrics["mae"],
                "mape_val_orig": np.nan if val_orig_metrics is None else val_orig_metrics["mape"],
                "mbe_val_orig": np.nan if val_orig_metrics is None else val_orig_metrics["mbe"],
                "abs_mbe_val_orig": np.nan if val_orig_metrics is None else val_orig_metrics["abs_mbe"],
                "rank_rmse_val": rank_rmse,
                "rank_abs_mbe_val": rank_abs_mbe,
            }
            grid_rows.append(row)

            best = best_by_model.get(cfg["model"])
            cand = {
                "cfg": cfg,
                "lagged": lagged,
                "selected_features": selected_features,
                "fs_report": fs_report,
                "rank_rmse": rank_rmse,
                "rank_abs_mbe": rank_abs_mbe,
                "row": row,
            }
            if best is None:
                best_by_model[cfg["model"]] = cand
            else:
                if rank_rmse < best["rank_rmse"] - 1e-12:
                    best_by_model[cfg["model"]] = cand
                elif abs(rank_rmse - best["rank_rmse"]) <= 1e-12:
                    if rank_abs_mbe < best["rank_abs_mbe"] - 1e-12:
                        best_by_model[cfg["model"]] = cand

        if not grid_rows:
            raise RuntimeError("Step 4 ML grid search failed for all candidate configurations")

        grid_df = pd.DataFrame(grid_rows).sort_values(
            ["model", "rank_rmse_val", "rank_abs_mbe_val"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        summary_rows: list[dict[str, Any]] = []
        forecasts = {
            "split": ["validation"] * len(self.validation) + ["test"] * len(self.test),
            "timestamp": list(self.validation.index) + list(self.test.index),
            "actual": list(self.validation.values) + list(self.test.values),
        }
        feature_reports: list[pd.DataFrame] = []

        for model_name, best in best_by_model.items():
            cfg = best["cfg"]
            lookback = int(cfg["lookback"])
            selected_features = best["selected_features"]

            model = self._build_estimator(model_name, cfg["params"])
            lagged = best["lagged"]
            lagged_sel = build_model_feature_matrix(lagged, selected_features)

            train_val_X = pd.concat([lagged_sel.X_train, lagged_sel.X_val], axis=0)
            train_val_y = pd.concat([lagged_sel.y_train, lagged_sel.y_val], axis=0)
            model.fit(train_val_X, train_val_y)

            val_pred = self._iterative_forecast(
                model=model,
                seed_series=self.train,
                horizon_index=self.validation.index,
                lookback=lookback,
                selected_features=selected_features,
            )
            test_pred = self._iterative_forecast(
                model=model,
                seed_series=self.train_validation,
                horizon_index=self.test.index,
                lookback=lookback,
                selected_features=selected_features,
            )

            val_metrics = compute_metrics(self.validation, val_pred)
            test_metrics = compute_metrics(self.test, test_pred)
            val_orig_metrics = original_scale_metrics_for_segment(
                val_pred,
                original_series=self.original_series,
                use_log1p=self.use_log1p,
                diff_order=self.diff_order,
            )
            test_orig_metrics = original_scale_metrics_for_segment(
                test_pred,
                original_series=self.original_series,
                use_log1p=self.use_log1p,
                diff_order=self.diff_order,
            )

            summary_rows.append(
                {
                    "model": model_name,
                    "lookback": lookback,
                    "feature_selection": self.config.feature_selection,
                    "selected_features": ", ".join(selected_features),
                    "n_selected_features": len(selected_features),
                    "best_params": str(cfg["params"]),
                    "rmse_val": val_metrics["rmse"],
                    "mae_val": val_metrics["mae"],
                    "mape_val": val_metrics["mape"],
                    "mbe_val": val_metrics["mbe"],
                    "abs_mbe_val": val_metrics["abs_mbe"],
                    "rmse_val_orig": np.nan if val_orig_metrics is None else val_orig_metrics["rmse"],
                    "mae_val_orig": np.nan if val_orig_metrics is None else val_orig_metrics["mae"],
                    "mape_val_orig": np.nan if val_orig_metrics is None else val_orig_metrics["mape"],
                    "mbe_val_orig": np.nan if val_orig_metrics is None else val_orig_metrics["mbe"],
                    "abs_mbe_val_orig": np.nan if val_orig_metrics is None else val_orig_metrics["abs_mbe"],
                    "rmse_test": test_metrics["rmse"],
                    "mae_test": test_metrics["mae"],
                    "mape_test": test_metrics["mape"],
                    "mbe_test": test_metrics["mbe"],
                    "abs_mbe_test": test_metrics["abs_mbe"],
                    "rmse_test_orig": np.nan if test_orig_metrics is None else test_orig_metrics["rmse"],
                    "mae_test_orig": np.nan if test_orig_metrics is None else test_orig_metrics["mae"],
                    "mape_test_orig": np.nan if test_orig_metrics is None else test_orig_metrics["mape"],
                    "mbe_test_orig": np.nan if test_orig_metrics is None else test_orig_metrics["mbe"],
                    "abs_mbe_test_orig": np.nan if test_orig_metrics is None else test_orig_metrics["abs_mbe"],
                }
            )

            forecasts[f"{model_name}_pred"] = list(val_pred.values) + list(test_pred.values)

            rep = best["fs_report"].copy()
            rep["model"] = model_name
            rep["lookback"] = lookback
            feature_reports.append(rep)

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.assign(rank_rmse_val=summary_df["rmse_val_orig"].fillna(summary_df["rmse_val"]))
        summary_df = summary_df.assign(rank_abs_mbe_val=summary_df["abs_mbe_val_orig"].fillna(summary_df["abs_mbe_val"]))
        summary_df = summary_df.assign(rank_rmse_test=summary_df["rmse_test_orig"].fillna(summary_df["rmse_test"]))
        summary_df = summary_df.assign(rank_abs_mbe_test=summary_df["abs_mbe_test_orig"].fillna(summary_df["abs_mbe_test"]))
        summary_df = summary_df.sort_values(["rank_rmse_val", "rank_abs_mbe_val"], ascending=[True, True]).reset_index(drop=True)

        winner_row = summary_df.iloc[0]
        winner = str(winner_row["model"])

        pred_series: dict[str, dict[str, pd.Series]] = {}
        forecasts_df = pd.DataFrame(forecasts)
        for m in summary_df["model"].astype(str).tolist():
            pred_col = f"{m}_pred"
            pred_series[m] = {
                "val": pd.Series(
                    forecasts_df[pred_col].iloc[: len(self.validation)].to_numpy(),
                    index=self.validation.index,
                ),
                "test": pd.Series(
                    forecasts_df[pred_col].iloc[len(self.validation):].to_numpy(),
                    index=self.test.index,
                ),
            }

        return {
            "grid": grid_df,
            "summary": summary_df,
            "winner": winner,
            "winner_params": dict(winner_row),
            "forecast_table": forecasts_df,
            "feature_selection_report": pd.concat(feature_reports, ignore_index=True) if feature_reports else pd.DataFrame(),
            "validation_actual": self.validation,
            "test_actual": self.test,
            "pred_series": pred_series,
            "config": asdict(self.config),
            "use_log1p": self.use_log1p,
            "diff_order": self.diff_order,
            "original_series": self.original_series,
        }
