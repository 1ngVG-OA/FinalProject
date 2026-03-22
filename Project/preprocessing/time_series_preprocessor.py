"""Reusable preprocessing class for univariate time series forecasting.

The class implements the Step 2 pipeline discussed for this project:
- temporal split without leakage,
- trend/seasonality diagnostics,
- local outlier flags based on YoY variations,
- deterministic transformations (log, power, differencing),
- optional scaling (standardization/normalization),
- statistical tests (ADF, KPSS, optional Shapiro-Wilk).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


@dataclass(frozen=True)
class SplitConfig:
    """Temporal split configuration expressed as proportions."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass(frozen=True)
class TransformConfig:
    """Transformation configuration for the preprocessing pipeline."""

    use_log1p: bool = False
    power_exponent: float | None = None
    diff_order: int = 0
    scale_method: str = "none"  # allowed: none, standard, minmax


@dataclass(frozen=True)
class OutlierConfig:
    """Configuration for local outlier detection on YoY changes."""

    window: int = 11
    threshold: float = 3.5


@dataclass(frozen=True)
class PreprocessingConfig:
    """Global preprocessing configuration container."""

    split: SplitConfig = SplitConfig()
    transform: TransformConfig = TransformConfig()
    outliers: OutlierConfig = OutlierConfig()
    run_shapiro: bool = False
    shapiro_max_n: int = 5000


class TimeSeriesPreprocessor:
    """Preprocess a univariate time series in a leakage-safe way."""

    def __init__(self, series: pd.Series, config: PreprocessingConfig | None = None) -> None:
        if config is None:
            config = PreprocessingConfig()

        self.config = config
        self.series = self._validate_series(series)
        self._scaler: StandardScaler | MinMaxScaler | None = None

    @staticmethod
    def _validate_series(series: pd.Series) -> pd.Series:
        """Validate and normalize input series format."""

        if not isinstance(series, pd.Series):
            raise TypeError("series must be a pandas Series")

        x = series.copy()
        x = pd.to_numeric(x, errors="coerce")
        x = x.dropna()

        if x.empty:
            raise ValueError("series has no valid numeric values")

        if not x.index.is_monotonic_increasing:
            x = x.sort_index()

        return x.astype(float)

    def split_series(self, series: pd.Series | None = None) -> dict[str, pd.Series]:
        """Split a series in train/validation/test using temporal order."""

        x = self.series if series is None else self._validate_series(series)
        n = len(x)

        train_end = int(n * self.config.split.train_ratio)
        val_end = int(n * (self.config.split.train_ratio + self.config.split.val_ratio))

        if train_end < 3 or val_end <= train_end or val_end >= n:
            raise ValueError("Invalid split configuration for series length")

        return {
            "train": x.iloc[:train_end].copy(),
            "val": x.iloc[train_end:val_end].copy(),
            "test": x.iloc[val_end:].copy(),
        }

    @staticmethod
    def _apply_log1p(series: pd.Series) -> pd.Series:
        """Apply log1p transform only when valid."""

        if (series <= -1.0).any():
            raise ValueError("log1p requires all values > -1")
        return np.log1p(series)

    @staticmethod
    def _apply_power(series: pd.Series, exponent: float) -> pd.Series:
        """Apply deterministic power transform preserving sign."""

        return np.sign(series) * (np.abs(series) ** exponent)

    def apply_deterministic_transforms(self, series: pd.Series) -> pd.Series:
        """Apply non-fitted transforms (log, power, differencing)."""

        x = series.copy()
        tcfg = self.config.transform

        if tcfg.use_log1p:
            x = self._apply_log1p(x)

        if tcfg.power_exponent is not None:
            x = self._apply_power(x, tcfg.power_exponent)

        if tcfg.diff_order > 0:
            # Apply differencing iteratively (order d), not lag-d differencing.
            for _ in range(tcfg.diff_order):
                x = x.diff()

        return x.dropna()

    def _fit_scaler(self, train: pd.Series) -> None:
        """Fit scaler on train split only to prevent leakage."""

        method = self.config.transform.scale_method.lower()

        if method == "none":
            self._scaler = None
            return
        if method == "standard":
            self._scaler = StandardScaler()
        elif method == "minmax":
            self._scaler = MinMaxScaler()
        else:
            raise ValueError("scale_method must be one of: none, standard, minmax")

        self._scaler.fit(train.to_numpy().reshape(-1, 1))

    def _scale_series(self, series: pd.Series) -> pd.Series:
        """Transform series with fitted scaler preserving index/name."""

        if self._scaler is None:
            return series

        arr = self._scaler.transform(series.to_numpy().reshape(-1, 1)).ravel()
        return pd.Series(arr, index=series.index, name=series.name)

    @staticmethod
    def local_outlier_flags(
        series: pd.Series,
        window: int = 11,
        threshold: float = 3.5,
    ) -> pd.DataFrame:
        """Detect local outliers on YoY changes with MAD + std fallback."""

        x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        yoy = x.diff().dropna()
        min_periods = max(5, window // 2)

        rolling_median = yoy.rolling(window=window, center=True, min_periods=min_periods).median()
        residual = yoy - rolling_median

        rolling_mad = residual.abs().rolling(window=window, center=True, min_periods=min_periods).median()
        rolling_std = residual.rolling(window=window, center=True, min_periods=min_periods).std(ddof=0)

        eps = 1e-9
        modified_z = 0.6745 * residual / (rolling_mad + eps)
        rolling_z = residual / (rolling_std + eps)
        use_std_fallback = rolling_mad.fillna(0.0) < 1e-6
        local_score = modified_z.mask(use_std_fallback, rolling_z)

        is_outlier = (local_score.abs() > threshold).fillna(False)

        return pd.DataFrame(
            {
                "year": yoy.index,
                "yoy_change": yoy.values,
                "rolling_median": rolling_median.values,
                "rolling_mad": rolling_mad.values,
                "local_score": local_score.values,
                "is_local_outlier": is_outlier.values,
            }
        )

    @staticmethod
    def run_stationarity_tests(series: pd.Series, run_shapiro: bool = False, shapiro_max_n: int = 5000) -> dict[str, Any]:
        """Run ADF, KPSS and optional Shapiro-Wilk tests on a series."""

        x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        if len(x) < 12:
            raise ValueError("series too short for robust stationarity tests")

        adf_stat, adf_p, adf_lags, adf_n, adf_cv, _ = adfuller(x, autolag="AIC")

        kpss_warning = ""
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                kpss_stat, kpss_p, kpss_lags, kpss_cv = kpss(x, regression="c", nlags="auto")
                if caught:
                    kpss_warning = str(caught[-1].message)
        except ValueError:
            # In edge cases KPSS can fail on near-constant data.
            kpss_stat, kpss_p, kpss_lags, kpss_cv = np.nan, np.nan, None, {}
            kpss_warning = "KPSS failed: near-constant or unsupported series segment"

        result: dict[str, Any] = {
            "n": int(len(x)),
            "adf_stat": float(adf_stat),
            "adf_pvalue": float(adf_p),
            "adf_used_lags": int(adf_lags),
            "adf_nobs": int(adf_n),
            "adf_stationary_at_05": bool(adf_p < 0.05),
            "kpss_stat": float(kpss_stat) if not pd.isna(kpss_stat) else np.nan,
            "kpss_pvalue": float(kpss_p) if not pd.isna(kpss_p) else np.nan,
            "kpss_used_lags": int(kpss_lags) if kpss_lags is not None else None,
            "kpss_stationary_at_05": bool(kpss_p >= 0.05) if not pd.isna(kpss_p) else False,
            "adf_critical_values": adf_cv,
            "kpss_critical_values": kpss_cv,
            "kpss_note": kpss_warning,
        }

        if run_shapiro:
            sample = x.iloc[: min(len(x), shapiro_max_n)]
            sh_stat, sh_p = stats.shapiro(sample)
            result["shapiro_stat"] = float(sh_stat)
            result["shapiro_pvalue"] = float(sh_p)
            result["shapiro_normal_at_05"] = bool(sh_p >= 0.05)

        return result

    def evaluate_candidates(self, candidates: Iterable[TransformConfig]) -> pd.DataFrame:
        """Evaluate transformation candidates with stationarity tests on train."""

        rows: list[dict[str, Any]] = []

        for cfg in candidates:
            tmp = TimeSeriesPreprocessor(self.series, PreprocessingConfig(
                split=self.config.split,
                transform=cfg,
                outliers=self.config.outliers,
                run_shapiro=self.config.run_shapiro,
                shapiro_max_n=self.config.shapiro_max_n,
            ))
            output = tmp.preprocess()
            tests = output["tests"]["train"]

            rows.append(
                {
                    "use_log1p": cfg.use_log1p,
                    "power_exponent": cfg.power_exponent,
                    "diff_order": cfg.diff_order,
                    "scale_method": cfg.scale_method,
                    "adf_pvalue_train": tests["adf_pvalue"],
                    "kpss_pvalue_train": tests["kpss_pvalue"],
                    "adf_stationary_train": tests["adf_stationary_at_05"],
                    "kpss_stationary_train": tests["kpss_stationary_at_05"],
                }
            )

        return pd.DataFrame(rows)

    def preprocess(self) -> dict[str, Any]:
        """Run the full preprocessing workflow and return all artifacts."""

        transformed_full = self.apply_deterministic_transforms(self.series)
        splits = self.split_series(transformed_full)

        self._fit_scaler(splits["train"])
        scaled_splits = {name: self._scale_series(s) for name, s in splits.items()}

        tests = {
            name: self.run_stationarity_tests(
                s,
                run_shapiro=self.config.run_shapiro,
                shapiro_max_n=self.config.shapiro_max_n,
            )
            for name, s in scaled_splits.items()
            if len(s) >= 12
        }

        outlier_df = self.local_outlier_flags(
            self.series,
            window=self.config.outliers.window,
            threshold=self.config.outliers.threshold,
        )

        split_summary = pd.DataFrame(
            [
                {
                    "split": name,
                    "start": s.index.min(),
                    "end": s.index.max(),
                    "n": int(len(s)),
                }
                for name, s in scaled_splits.items()
            ]
        )

        return {
            "config": asdict(self.config),
            "series_transformed": transformed_full,
            "splits": scaled_splits,
            "split_summary": split_summary,
            "tests": tests,
            "local_outliers": outlier_df,
        }

    def save_preprocessing_plots(self, preproc_output: dict[str, Any], out_dir: Path) -> dict[str, Path]:
        """Generate and save core preprocessing plots."""

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        raw = self.series
        transformed = preproc_output["series_transformed"]
        splits = preproc_output["splits"]
        outliers = preproc_output["local_outliers"]

        plot_paths = {
            "preproc_plot_raw_vs_transformed": out_dir / "tavola_1_14_preproc_raw_vs_transformed_v1.png",
            "preproc_plot_split_view": out_dir / "tavola_1_14_preproc_split_view_v1.png",
            "preproc_plot_acf_pacf": out_dir / "tavola_1_14_preproc_acf_pacf_v1.png",
            "preproc_plot_local_outliers": out_dir / "tavola_1_14_preproc_local_outliers_v1.png",
        }

        # 1) Raw vs transformed series.
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        axes[0].plot(raw.index, raw.values, color="tab:blue", linewidth=2)
        axes[0].set_title("Raw Series")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Value")
        axes[0].grid(alpha=0.25)

        axes[1].plot(transformed.index, transformed.values, color="tab:orange", linewidth=1.8)
        axes[1].set_title("Preprocessed Series (Configured Transform)")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Transformed value")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_paths["preproc_plot_raw_vs_transformed"], dpi=150)
        plt.close(fig)

        # 2) Train/Val/Test split visualization.
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(splits["train"].index, splits["train"].values, label="train", color="tab:blue")
        ax.plot(splits["val"].index, splits["val"].values, label="val", color="tab:green")
        ax.plot(splits["test"].index, splits["test"].values, label="test", color="tab:red")
        ax.set_title("Preprocessed Series Split (Train / Val / Test)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Transformed value")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_paths["preproc_plot_split_view"], dpi=150)
        plt.close(fig)

        # 3) ACF/PACF on transformed train split.
        train_series = splits["train"].dropna()
        n_lags = max(5, min(20, int(len(train_series) / 3)))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(train_series, lags=n_lags, ax=axes[0])
        axes[0].set_title("ACF (Train)")
        plot_pacf(train_series, lags=n_lags, ax=axes[1], method="ywm")
        axes[1].set_title("PACF (Train)")
        fig.tight_layout()
        fig.savefig(plot_paths["preproc_plot_acf_pacf"], dpi=150)
        plt.close(fig)

        # 4) Local outliers over YoY changes.
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(outliers["year"], outliers["yoy_change"], color="tab:blue", linewidth=1.8, label="YoY change")
        ax.plot(
            outliers["year"],
            outliers["rolling_median"],
            color="tab:green",
            linestyle="--",
            linewidth=1.6,
            label="Rolling median",
        )
        local_mask = outliers["is_local_outlier"]
        ax.scatter(
            outliers.loc[local_mask, "year"],
            outliers.loc[local_mask, "yoy_change"],
            color="tab:red",
            s=45,
            label="Local outliers",
            zorder=3,
        )
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.set_title("Local Outliers on YoY Changes (Preprocessing)")
        ax.set_xlabel("Year")
        ax.set_ylabel("YoY change")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_paths["preproc_plot_local_outliers"], dpi=150)
        plt.close(fig)

        return plot_paths
