# Seconda fase del progetto: implementazione del preprocessing per serie temporali.
# Trasformazione, scaling, rilevamento outlier e test di stazionarietà, in modo da preparare i dati per la modellazione predittiva. 
# é progettato per essere flessibile e configurabile, consentendo di valutare diverse combinazioni di trasformazioni e scalature in modo leakage-safe.

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


# ------------------------------------------------------------------
# Configurazioni preprocessing
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SplitConfig:

    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass(frozen=True)
class TransformConfig:

    use_log1p: bool = False
    power_exponent: float | None = None
    diff_order: int = 0
    scale_method: str = "none"  # allowed: none, standard, minmax


@dataclass(frozen=True)
class OutlierConfig:

    window: int = 11 # Deve essere dispari per avere una finestra centrata simmetrica, e sufficientemente ampia per stimare una baseline robusta.
    threshold: float = 3.5 # Soglia per identificare outlier locali basati sui cambiamenti YoY, in termini di deviazioni dalla baseline locale (rolling median), con un approccio robusto che utilizza MAD e fallback alla deviazione standard quando MAD è troppo piccolo.


@dataclass(frozen=True)
class PreprocessingConfig:

    split: SplitConfig = SplitConfig()
    transform: TransformConfig = TransformConfig()
    outliers: OutlierConfig = OutlierConfig()
    run_shapiro: bool = False # Opzione per eseguire il test di normalità Shapiro-Wilk sui residui della serie trasformata, utile per valutare la normalità dei dati dopo le trasformazioni e confrontarla con i risultati dei test di stazionarietà.
    shapiro_max_n: int = 5000 # Numero massimo di campioni da utilizzare per il test di Shapiro-Wilk, che può essere computazionalmente costoso su serie molto lunghe, quindi si limita a un campione rappresentativo dei dati trasformati.


# ------------------------------------------------------------------
# Preprocessor principale
# ------------------------------------------------------------------

class TimeSeriesPreprocessor:
    # Preprocessing class per serie temporali, che include validazione, trasformazioni deterministiche, scaling, rilevamento outlier locali e test di stazionarietà.
    def __init__(self, series: pd.Series, config: PreprocessingConfig | None = None) -> None:
        if config is None:
            config = PreprocessingConfig()

        self.config = config
        self.series = self._validate_series(series)
        self._scaler: StandardScaler | MinMaxScaler | None = None

    # Validazione della serie in ingresso, assicurandosi che sia una pandas Series con valori numerici validi, ordinati temporalmente e senza valori NaN. Viene restituita una copia pulita della serie pronta per le trasformazioni successive.
    @staticmethod
    def _validate_series(series: pd.Series) -> pd.Series:

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

    # Suddivide la serie in train/validation/test in base alla configurazione.
    def split_series(self, series: pd.Series | None = None) -> dict[str, pd.Series]:

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

    # Trasformazione log1p che gestisce valori vicino a zero e negativi, utile per stabilizzare la varianza in serie con questi tipi di valori, e configurabile tramite use_log1p.
    @staticmethod
    def _apply_log1p(series: pd.Series) -> pd.Series:

        if (series <= -1.0).any():
            raise ValueError("log1p requires all values > -1")
        return np.log1p(series)

    # Trasformazione di potenza che preserva il segno, utile per stabilizzare la varianza in serie con valori negativi o vicino a zero, configurabile tramite power_exponent.
    @staticmethod
    def _apply_power(series: pd.Series, exponent: float) -> pd.Series:

        return np.sign(series) * (np.abs(series) ** exponent) 

    # Trasformazioni deterministiche applicate alla serie, come logaritmo, potenza e differenziazione, in modo da preparare i dati per i test di stazionarietà e la modellazione predittiva. 
    # Le trasformazioni vengono applicate in un ordine specifico (log1p -> power -> differencing) e sono configurabili tramite TransformConfig.  
    def apply_deterministic_transforms(self, series: pd.Series) -> pd.Series:

        x = series.copy()
        tcfg = self.config.transform

        if tcfg.use_log1p:
            x = self._apply_log1p(x)

        if tcfg.power_exponent is not None:
            x = self._apply_power(x, tcfg.power_exponent)

        if tcfg.diff_order > 0:
            # La differenziazione viene applicata iterativamente in base all'ordine specificato, con l'obiettivo di rimuovere trend e rendere la serie più stazionaria, se necessario. 
            for _ in range(tcfg.diff_order):
                x = x.diff()

        return x.dropna()

    # Fitting dello scaler sui dati di train, in modo da evitare data leakage. 
    # Il metodo di scaling è configurabile tramite TransformConfig e può essere standardizzazione, min-max scaling o nessuno.
    def _fit_scaler(self, train: pd.Series) -> None:

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

    # Applicazione dello scaler alla serie, utilizzando i parametri appresi sul train. 
    # Se lo scaler è None, restituisce la serie originale senza modifiche.
    def _scale_series(self, series: pd.Series) -> pd.Series:

        if self._scaler is None:
            return series

        arr = self._scaler.transform(series.to_numpy().reshape(-1, 1)).ravel()
        return pd.Series(arr, index=series.index, name=series.name)

    # Rilevamento degli outlier locali basati sui cambiamenti YoY, utilizzando un approccio robusto che combina MAD e deviazione standard come fallback, e restituendo un DataFrame con i dettagli dei cambiamenti YoY, la baseline locale e i flag di outlier.
    @staticmethod
    def local_outlier_flags(
        series: pd.Series,
        window: int = 11,
        threshold: float = 3.5,
    ) -> pd.DataFrame:

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

    # Test di stazionarietà ADF e KPSS sulla serie trasformata, restituendo un dizionario con i risultati dei test, inclusi statistiche, p-value, lags utilizzati e flag di stazionarietà a livello di significatività del 5%.
    @staticmethod
    def run_stationarity_tests(series: pd.Series, run_shapiro: bool = False, shapiro_max_n: int = 5000) -> dict[str, Any]:

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

    # Valutazione di diverse combinazioni di trasformazioni e scalature configurate tramite TransformConfig, 
    # eseguendo il workflow completo di preprocessing per ciascuna combinazione e raccogliendo i risultati dei test di stazionarietà in un DataFrame per confronto.
    def evaluate_candidates(self, candidates: Iterable[TransformConfig]) -> pd.DataFrame:

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

    # Esegue l'intero workflow di preprocessing, applicando le trasformazioni configurate, suddividendo la serie in train/val/test, scalando i dati in modo leakage-safe,
    # eseguendo i test di stazionarietà e rilevando gli outlier locali, restituendo un dizionario con tutti gli artefatti generati durante il processo.
    def preprocess(self) -> dict[str, Any]:

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

    # Salvataggio dei grafici di preprocessing, inclusi raw vs transformed, visualizzazione dei split, ACF/PACF e outlier locali, in un formato configurabile e con percorsi di output specificati.
    def save_preprocessing_plots(self, preproc_output: dict[str, Any], out_dir: Path) -> dict[str, Path]:

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        raw = self.series
        transformed = preproc_output["series_transformed"]
        splits = preproc_output["splits"]
        outliers = preproc_output["local_outliers"]

        plot_paths = {
            "preproc_plot_raw_vs_transformed": out_dir / "raw_vs_transformed.png",
            "preproc_plot_split_view": out_dir / "split_view.png",
            "preproc_plot_acf_pacf": out_dir / "acf_pacf.png",
            "preproc_plot_local_outliers": out_dir / "local_outliers.png",
        }

        # 1) Serie raw vs trasformata.
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

        # 2) Visualizzazione dei split Train/Val/Test.
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

        # 3) ACF/PACF sulla suddivisione train trasformata.
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

        # 4) Outlier locali sulle variazioni YoY.
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
