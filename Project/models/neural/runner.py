"""Runner for Step 5 torch-based neural models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from Project.preprocessing.time_series_preprocessor import PreprocessingConfig

from .features import WindowedSplits, build_training_windows, build_windowed_splits
from .model_config import (
    NeuralStepConfig,
    compute_metrics,
    original_scale_metrics_for_segment,
    resolve_torch_device,
    seed_everything,
)
from .models import LSTMForecaster, MLPForecaster


class NeuralModelRunner:
    """Run Step 5 neural model selection and evaluation."""

    def __init__(
        self,
        train: pd.Series,
        validation: pd.Series,
        test: pd.Series,
        config: NeuralStepConfig | None = None,
        original_series: pd.Series | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> None:
        self.config = config or NeuralStepConfig()
        self.train = NeuralStepConfig.validate_split(train, "train")
        self.validation = NeuralStepConfig.validate_split(validation, "validation")
        self.test = NeuralStepConfig.validate_split(test, "test")
        self.train_validation = pd.concat([self.train, self.validation], axis=0)
        self.original_series = NeuralStepConfig.validate_original_series(original_series)
        self.preprocessing_config = preprocessing_config
        self.device = resolve_torch_device(self.config.device)

    def _candidate_configs(self) -> list[dict[str, Any]]:
        cfg = self.config
        rows: list[dict[str, Any]] = []

        if "mlp" in cfg.candidate_models:
            for lookback, hidden_size, activation, dropout, batch_size, lr, weight_decay in product(
                cfg.lookback_values,
                cfg.mlp_hidden_sizes,
                cfg.mlp_activations,
                cfg.mlp_dropouts,
                cfg.batch_sizes,
                cfg.learning_rates,
                cfg.weight_decays,
            ):
                rows.append(
                    {
                        "model": "mlp",
                        "lookback": int(lookback),
                        "params": {
                            "hidden_size": int(hidden_size),
                            "activation": str(activation),
                            "dropout": float(dropout),
                            "batch_size": int(batch_size),
                            "learning_rate": float(lr),
                            "weight_decay": float(weight_decay),
                        },
                    }
                )

        if "lstm" in cfg.candidate_models:
            for lookback, hidden_size, num_layers, dropout, batch_size, lr, weight_decay in product(
                cfg.lookback_values,
                cfg.lstm_hidden_sizes,
                cfg.lstm_num_layers,
                cfg.lstm_dropouts,
                cfg.batch_sizes,
                cfg.learning_rates,
                cfg.weight_decays,
            ):
                rows.append(
                    {
                        "model": "lstm",
                        "lookback": int(lookback),
                        "params": {
                            "hidden_size": int(hidden_size),
                            "num_layers": int(num_layers),
                            "dropout": float(dropout),
                            "batch_size": int(batch_size),
                            "learning_rate": float(lr),
                            "weight_decay": float(weight_decay),
                        },
                    }
                )

        return rows

    @staticmethod
    def _reshape_inputs(X: np.ndarray, model_name: str) -> np.ndarray:
        if model_name == "lstm":
            return np.asarray(X, dtype=np.float32).reshape(len(X), X.shape[1], 1)
        return np.asarray(X, dtype=np.float32)

    def _build_model(self, model_name: str, lookback: int, params: dict[str, Any]) -> nn.Module:
        if model_name == "mlp":
            return MLPForecaster(
                input_size=lookback,
                hidden_size=int(params["hidden_size"]),
                activation=str(params.get("activation", "relu")),
                dropout=float(params.get("dropout", 0.0)),
            )
        if model_name == "lstm":
            return LSTMForecaster(
                input_size=1,
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params.get("num_layers", 1)),
                dropout=float(params.get("dropout", 0.0)),
            )
        raise ValueError(f"Unknown neural model: {model_name}")

    def _fit_model(
        self,
        model_name: str,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict[str, Any],
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        fixed_epochs: int | None = None,
    ) -> dict[str, float | int]:
        seed_everything(self.config.seed)
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = Adam(
            model.parameters(),
            lr=float(params["learning_rate"]),
            weight_decay=float(params.get("weight_decay", 0.0)),
        )

        X_train_arr = self._reshape_inputs(X_train, model_name)
        train_dataset = TensorDataset(
            torch.tensor(X_train_arr, dtype=torch.float32),
            torch.tensor(np.asarray(y_train, dtype=np.float32).reshape(-1, 1), dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(params["batch_size"]),
            shuffle=True,
        )

        val_inputs = None
        val_targets = None
        if X_val is not None and y_val is not None:
            val_inputs = torch.tensor(self._reshape_inputs(X_val, model_name), dtype=torch.float32, device=self.device)
            val_targets = torch.tensor(np.asarray(y_val, dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=self.device)

        max_epochs = int(fixed_epochs) if fixed_epochs is not None else int(self.config.max_epochs)
        best_epoch = max_epochs
        best_val_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            if val_inputs is None or val_targets is None or fixed_epochs is not None:
                best_epoch = epoch + 1
                continue

            model.eval()
            with torch.no_grad():
                val_pred = model(val_inputs)
                val_loss = float(criterion(val_pred, val_targets).item())

            if val_loss < best_val_loss - 1e-12:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= int(self.config.patience):
                    break

        if val_inputs is not None and val_targets is not None and fixed_epochs is None:
            model.load_state_dict(best_state)

        return {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else float("nan"),
        }

    def _recursive_forecast(
        self,
        model_name: str,
        model: nn.Module,
        seed_series: pd.Series,
        horizon_index: pd.Index,
        lookback: int,
    ) -> pd.Series:
        """Recursive multi-step forecast using the model as a one-step predictor."""

        history = list(pd.to_numeric(seed_series, errors="coerce").dropna().astype(float).to_numpy())
        preds: list[float] = []

        model.eval()
        with torch.no_grad():
            for _ in range(len(horizon_index)):
                if len(history) < lookback:
                    raise RuntimeError("Insufficient history for recursive neural forecast")
                window = np.asarray(history[-lookback:], dtype=np.float32)
                if model_name == "lstm":
                    x = torch.tensor(window.reshape(1, lookback, 1), dtype=torch.float32, device=self.device)
                else:
                    x = torch.tensor(window.reshape(1, lookback), dtype=torch.float32, device=self.device)
                yhat = float(model(x).detach().cpu().item())
                preds.append(yhat)
                history.append(yhat)

        return pd.Series(np.asarray(preds, dtype=float), index=horizon_index, name="pred")

    def run(self) -> dict[str, Any]:
        """Run neural model selection and return all artifacts."""

        grid_rows: list[dict[str, Any]] = []
        best_by_model: dict[str, dict[str, Any]] = {}
        prepared_by_lookback: dict[int, WindowedSplits] = {}

        for cfg in self._candidate_configs():
            lookback = int(cfg["lookback"])
            if lookback not in prepared_by_lookback:
                prepared_by_lookback[lookback] = build_windowed_splits(
                    self.train,
                    self.validation,
                    self.test,
                    lookback,
                )
            prepared = prepared_by_lookback[lookback]

            model_name = str(cfg["model"])
            seed_everything(self.config.seed)
            model = self._build_model(model_name, lookback, cfg["params"])
            train_stats = self._fit_model(
                model_name,
                model,
                prepared.X_train,
                prepared.y_train,
                cfg["params"],
                X_val=prepared.X_val,
                y_val=prepared.y_val,
            )
            val_pred = self._recursive_forecast(
                model_name,
                model,
                prepared.train_seed,
                self.validation.index,
                lookback,
            )

            val_metrics = compute_metrics(self.validation, val_pred)
            val_orig_metrics = original_scale_metrics_for_segment(
                val_pred,
                original_series=self.original_series,
                preprocessing_config=self.preprocessing_config,
            )
            rank_rmse = val_orig_metrics["rmse"] if val_orig_metrics is not None else val_metrics["rmse"]
            rank_abs_mbe = val_orig_metrics["abs_mbe"] if val_orig_metrics is not None else val_metrics["abs_mbe"]

            row = {
                "model": model_name,
                "lookback": lookback,
                "params": str(cfg["params"]),
                "best_epoch": int(train_stats["best_epoch"]),
                "best_val_loss": float(train_stats["best_val_loss"]),
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

            candidate_info = {
                "cfg": cfg,
                "prepared": prepared,
                "rank_rmse": rank_rmse,
                "rank_abs_mbe": rank_abs_mbe,
                "train_stats": train_stats,
            }
            best = best_by_model.get(model_name)
            if best is None:
                best_by_model[model_name] = candidate_info
            else:
                if rank_rmse < best["rank_rmse"] - 1e-12:
                    best_by_model[model_name] = candidate_info
                elif abs(rank_rmse - best["rank_rmse"]) <= 1e-12 and rank_abs_mbe < best["rank_abs_mbe"] - 1e-12:
                    best_by_model[model_name] = candidate_info

        if not grid_rows:
            raise RuntimeError("Step 5 neural grid search failed for all candidate configurations")

        grid_df = pd.DataFrame(grid_rows).sort_values(
            ["model", "rank_rmse_val", "rank_abs_mbe_val"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        summary_rows: list[dict[str, Any]] = []
        forecasts: dict[str, list[Any]] = {
            "split": ["validation"] * len(self.validation) + ["test"] * len(self.test),
            "timestamp": list(self.validation.index) + list(self.test.index),
            "actual": list(self.validation.values) + list(self.test.values),
        }
        pred_series: dict[str, dict[str, pd.Series]] = {}

        for model_name, best in best_by_model.items():
            cfg = best["cfg"]
            lookback = int(cfg["lookback"])
            params = cfg["params"]
            best_epoch = int(best["train_stats"]["best_epoch"])

            seed_everything(self.config.seed)
            model = self._build_model(model_name, lookback, params)
            X_train_val, y_train_val = build_training_windows(self.train_validation, lookback)
            self._fit_model(
                model_name,
                model,
                X_train_val,
                y_train_val,
                params,
                fixed_epochs=best_epoch,
            )

            val_pred = self._recursive_forecast(model_name, model, self.train, self.validation.index, lookback)
            test_pred = self._recursive_forecast(model_name, model, self.train_validation, self.test.index, lookback)

            val_metrics = compute_metrics(self.validation, val_pred)
            test_metrics = compute_metrics(self.test, test_pred)
            val_orig_metrics = original_scale_metrics_for_segment(
                val_pred,
                original_series=self.original_series,
                preprocessing_config=self.preprocessing_config,
            )
            test_orig_metrics = original_scale_metrics_for_segment(
                test_pred,
                original_series=self.original_series,
                preprocessing_config=self.preprocessing_config,
            )

            summary_rows.append(
                {
                    "model": model_name,
                    "lookback": lookback,
                    "best_params": str(params),
                    "best_epoch": best_epoch,
                    "best_val_loss": float(best["train_stats"]["best_val_loss"]),
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
            pred_series[model_name] = {"val": val_pred, "test": test_pred}

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.assign(rank_rmse_val=summary_df["rmse_val_orig"].fillna(summary_df["rmse_val"]))
        summary_df = summary_df.assign(rank_abs_mbe_val=summary_df["abs_mbe_val_orig"].fillna(summary_df["abs_mbe_val"]))
        summary_df = summary_df.assign(rank_rmse_test=summary_df["rmse_test_orig"].fillna(summary_df["rmse_test"]))
        summary_df = summary_df.assign(rank_abs_mbe_test=summary_df["abs_mbe_test_orig"].fillna(summary_df["abs_mbe_test"]))
        summary_df = summary_df.sort_values(["rank_rmse_val", "rank_abs_mbe_val"], ascending=[True, True]).reset_index(drop=True)

        winner_row = summary_df.iloc[0]
        winner = str(winner_row["model"])
        forecasts_df = pd.DataFrame(forecasts)

        return {
            "grid": grid_df,
            "summary": summary_df,
            "winner": winner,
            "winner_params": dict(winner_row),
            "forecast_table": forecasts_df,
            "validation_actual": self.validation,
            "test_actual": self.test,
            "pred_series": pred_series,
            "config": asdict(self.config),
            "preprocessing_config": self.preprocessing_config,
            "original_series": self.original_series,
            "device": str(self.device),
        }