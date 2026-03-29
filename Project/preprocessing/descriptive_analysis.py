# Prima fase del progetto: analisi descrittiva e pulizia dei dati.
# Questo modulo contiene funzioni per caricare la serie target, calcolare statistiche descrittive, identificare outlier e trend, e generare visualizzazioni esplorative.
# In questo modo, si ottiene una comprensione "qualitativa" e "quantitativa" approfondita della serie temporale prima di procedere alla modellazione e alla previsione.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


TARGET_COLUMN_INDEX = {
    "production_total": 1,
}

TARGET_SERIES_NAME = {
    "production_total": "produzione_lorda_totale",
}

# Classe di configurazione per i percorsi dei file utilizzati nell'analisi descrittiva.
@dataclass(frozen=True)
class DescriptivePaths:
    dataset_path: Path
    results_metrics_dir: Path
    results_plots_dir: Path

# Funzione di parsing per i valori numerici nel formato specifico del dataset ISTAT, che utilizza il punto come separatore delle migliaia e la virgola come separatore decimale.
def _parse_istat_number(value: str) -> float:

    if value is None:
        return np.nan

    s = str(value).strip()
    if s in {"", "....", "-"}:
        return np.nan

    # Il dataset utilizza il punto come separatore delle migliaia e la virgola come separatore decimale.
    s = s.replace(".", "")
    s = s.replace(",", ".")

    try:
        return float(s)
    except ValueError:
        return np.nan

# Funzione principale per caricare la serie target dal dataset CSV, pulire i dati e restituire una Serie Pandas con anni come indice e valori numerici.
def load_target_series(dataset_path: Path, target: str = "production_total") -> pd.Series:

    # Validazione dell'input target e caricamento del dataset CSV.
    if target not in TARGET_COLUMN_INDEX:
        options = ", ".join(sorted(TARGET_COLUMN_INDEX))
        raise ValueError(f"Unsupported target '{target}'. Available targets: {options}")

    # Filtro per righe che rappresentano anni e si estrae la colonna target specificata.
    raw = pd.read_csv(dataset_path, sep=";", header=None, dtype=str)

    year_mask = raw[0].astype(str).str.fullmatch(r"\d{4}") 
    target_col = TARGET_COLUMN_INDEX[target]    
    data = raw.loc[year_mask, [0, target_col]].copy()  
    data.columns = ["year", "value"]   

    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64") 
    data["value"] = data["value"].map(_parse_istat_number)

    data = data.dropna(subset=["year", "value"]).copy()
    data["year"] = data["year"].astype(int)

    series_name = TARGET_SERIES_NAME.get(target, "target_series")
    series = pd.Series(data["value"].values, index=data["year"].values, name=series_name)
    series.index.name = "year"
    return series

# Le seguenti funzioni implementano le varie analisi descrittive richieste, restituendo DataFrame con i risultati tabulari. 
def _frequency_distribution(series: pd.Series, n_bins: int | None = None) -> pd.DataFrame:
    # Calcola la distribuzione di frequenza della serie, suddividendo i valori in classi (bin) e contando le frequenze assolute e relative. 
    x = series.dropna()
    if n_bins is None:
        n_bins = int(np.ceil(np.log2(len(x)) + 1))

    # Utilizza pd.cut per creare le classi e value_counts per contare le frequenze.
    categories = pd.cut(x, bins=n_bins, include_lowest=True)
    abs_freq = categories.value_counts(sort=False)
    rel_freq = abs_freq / abs_freq.sum()

    return pd.DataFrame(
        {
            "class_interval": abs_freq.index.astype(str),
            "absolute_frequency": abs_freq.values,
            "relative_frequency": rel_freq.values,
        }
    )

# Le seguenti funzioni implementano le varie analisi descrittive richieste, restituendo DataFrame con i risultati tabulari.
def _central_tendency(series: pd.Series) -> pd.DataFrame:
    # Calcola le misure di tendenza centrale: media, mediana e moda. 
    # La moda viene calcolata con pandas.Series.mode(), che può restituire più valori in caso di multimodalità; in questo caso, si prende il primo valore come rappresentativo.
    x = series.dropna()
    modes = x.mode()
    mode_value = float(modes.iloc[0]) if not modes.empty else np.nan

    return pd.DataFrame(
        [
            {
                "mean": float(x.mean()),
                "median": float(x.median()),
                "mode": mode_value,
            }
        ]
    )

def _dispersion_measures(series: pd.Series) -> pd.DataFrame:
    
    # Calcola le misure di dispersione: range, varianza, deviazione standard, coefficiente di variazione e interquartile range (IQR).
    # Il coefficiente di variazione è calcolato come deviazione standard divisa per la media, con gestione del caso in cui la media sia zero per evitare divisioni per zero.
    x = series.dropna()
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1

    mean_value = float(x.mean())
    std_value = float(x.std(ddof=1))
    cv_value = std_value / mean_value if mean_value != 0 else np.nan

    return pd.DataFrame(
        [
            {
                "range": float(x.max() - x.min()),
                "variance": float(x.var(ddof=1)),
                "std_dev": std_value,
                "coefficient_of_variation": cv_value,
                "iqr": iqr,
            }
        ]
    )

# La funzione _outlier_table_iqr identifica gli outlier globali utilizzando il metodo dell'intervallo interquartile (IQR)
# Restituisce sia i dettagli degli outlier che un riepilogo statistico.
def _outlier_table_iqr(series: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    x = series.dropna()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (x < lower) | (x > upper) # Maschera booleana per identificare gli outlier globali secondo il criterio IQR
    outliers = x.loc[outlier_mask] # Serie con solo gli outlier identificati

    details = pd.DataFrame(
        {
            "year": outliers.index,
            "value": outliers.values,
        }
    )
    # Riepilogo statistico con i confini di outlier e il numero/percentuale di outlier identificati.
    summary = pd.DataFrame(
        [
            {
                "lower_fence": float(lower),
                "upper_fence": float(upper),
                "num_outliers": int(outlier_mask.sum()),
                "outlier_ratio": float(outlier_mask.mean()),
            }
        ]
    )

    return details, summary

# La funzione _trend_validation esegue un'analisi di regressione lineare per valutare la presenza di un trend significativo nella serie temporale
# Restituisce sia i parametri della regressione che le statistiche di correlazione.
def _trend_validation(series: pd.Series) -> pd.DataFrame:

    x = series.dropna().astype(float)
    years = x.index.to_numpy(dtype=float)
    values = x.to_numpy(dtype=float)
    # Utilizza scipy.stats.linregress per calcolare i parametri della regressione lineare (slope, intercept) e le statistiche di significatività (p-value, r-squared).
    # Calcola anche il coefficiente di correlazione di Spearman, che è una misura non parametrica della correlazione monotona tra anni e valori, utile per identificare trend non lineari.
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
    spearman_rho, spearman_p = stats.spearmanr(years, values)
    
    # Calcola la distribuzione dei cambiamenti anno su anno (YoY) per valutare la direzione del trend e la sua stabilità.
    diff = x.diff().dropna()
    positive_share = float((diff > 0).mean()) if len(diff) else np.nan
    negative_share = float((diff < 0).mean()) if len(diff) else np.nan
    zero_share = float((diff == 0).mean()) if len(diff) else np.nan

    return pd.DataFrame(
        [
            {
                "n_observations": int(len(x)),
                "start_year": int(x.index.min()),
                "end_year": int(x.index.max()),
                "slope_per_year": float(slope),
                "slope_p_value": float(p_value),
                "r_squared": float(r_value**2),
                "spearman_rho": float(spearman_rho),
                "spearman_p_value": float(spearman_p),
                "positive_yoy_share": positive_share,
                "negative_yoy_share": negative_share,
                "zero_yoy_share": zero_share,
                "intercept": float(intercept),
                "slope_std_err": float(std_err),
            }
        ]
    )

# La funzione _local_outliers_on_variation identifica gli outlier locali basati sui cambiamenti anno su anno (YoY) rispetto a una baseline locale calcolata con una mediana mobile.
# Restituisce i dettagli degli outlier locali, un riepilogo statistico e una tabella con i dettagli di tutti i punti, inclusi i cambiamenti YoY, la baseline locale e i punteggi di outlier.
def _local_outliers_on_variation(
    series: pd.Series,
    window: int = 11,
    threshold: float = 3.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    # Calcola i cambiamenti anno su anno (YoY) e utilizza una mediana mobile per stabilire una baseline locale.
    x = series.dropna().astype(float)
    yoy = x.diff().dropna()
    # Il parametro min_periods è impostato a metà della finestra o a 5, per garantire che la mediana mobile venga calcolata solo quando ci sono abbastanza punti dati.
    min_periods = max(5, window // 2)
    rolling_median = yoy.rolling(window=window, center=True, min_periods=min_periods).median()
    residual = yoy - rolling_median
    
    # Calcola sia la deviazione assoluta mediana (MAD) che la deviazione standard per i residui, e utilizza una combinazione di entrambi per identificare gli outlier locali, con una soglia di 3.5 per il punteggio combinato.
    rolling_mad = residual.abs().rolling(window=window, center=True, min_periods=min_periods).median()
    rolling_std = residual.rolling(window=window, center=True, min_periods=min_periods).std(ddof=0)
    eps = 1e-9 # Piccolo valore per evitare divisioni per zero.
    modified_z = 0.6745 * residual / (rolling_mad + eps)   # Il fattore 0.6745 è utilizzato per rendere il punteggio modificato comparabile con la deviazione standard in caso di distribuzione normale.
    rolling_z = residual / (rolling_std + eps)

    use_std_fallback = rolling_mad.fillna(0.0) < 1e-6 # Se la MAD è troppo piccola (indicando poca variabilità), si utilizza la deviazione standard come fallback per il calcolo del punteggio di outlier.
    combined_score = modified_z.mask(use_std_fallback, rolling_z) # Punteggio modificato basato sulla MAD quando è affidabile / Punteggio basato sulla deviazione standard come fallback quando la MAD è troppo piccola.

    flag = combined_score.abs() > threshold

    details = pd.DataFrame(
        {
            "year": yoy.index,
            "yoy_change": yoy.values,
            "local_median_change": rolling_median.values,
            "residual_vs_local": residual.values,
            "rolling_mad": rolling_mad.values,
            "local_score": combined_score.values,
            "is_local_outlier": flag.fillna(False).values,
        }
    )
    # Estrae solo gli outlier locali identificati per un'analisi più approfondita.
    local_outliers = details.loc[details["is_local_outlier"]].copy()
    # Riepilogo statistico con il numero di outlier locali identificati, la loro proporzione rispetto al totale dei punti YoY, e le statistiche descrittive dei cambiamenti YoY.
    summary = pd.DataFrame(
        [
            {
                "window": int(window),
                "threshold": float(threshold),
                "n_yoy_points": int(len(yoy)),
                "num_local_outliers": int(details["is_local_outlier"].sum()),
                "local_outlier_ratio": float(details["is_local_outlier"].mean()),
                "yoy_mean": float(yoy.mean()),
                "yoy_std": float(yoy.std(ddof=1)),
                "yoy_q05": float(yoy.quantile(0.05)),
                "yoy_q95": float(yoy.quantile(0.95)),
            }
        ]
    )

    return details, local_outliers, summary

# La funzione _save_distribution_plots genera una serie di visualizzazioni per esplorare la distribuzione della serie temporale, inclusi grafici di frequenza, densità, boxplot e trend.
def _save_distribution_plots(series: pd.Series, freq_df: pd.DataFrame, out_dir: Path) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    x = series.dropna()

    # 0) Serie temporale base con line plot per visualizzare l'andamento generale della serie nel tempo.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x.index, x.values, color="tab:blue", linewidth=2)
    series_label = str(series.name or "target_series").replace("_", " ").strip()
    ax.set_title(f"Base Time Series - {series_label}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "series_base.png", dpi=150)
    plt.close(fig)

    # 1) Grafico a barre della distribuzione di frequenza (assoluta + relativa empirica).
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    idx = np.arange(len(freq_df))

    bars = ax1.bar(idx, freq_df["absolute_frequency"], alpha=0.75, label="Absolute")
    ax2.plot(idx, freq_df["relative_frequency"], color="tab:red", marker="o", label="Relative")

    ax1.set_title("Frequency Distribution (Binned)")
    ax1.set_xlabel("Class interval")
    ax1.set_ylabel("Absolute frequency")
    ax2.set_ylabel("Relative frequency")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(freq_df["class_interval"], rotation=80)

    ax1.legend([bars], ["Absolute"], loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "frequency_distribution.png", dpi=150)
    plt.close(fig)

    # 2) Istogramma + KDE + densità normale e uniforme adattate.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=20, density=True, alpha=0.5, color="tab:blue", label="Empirical density")

    kde = stats.gaussian_kde(x)
    xx = np.linspace(x.min(), x.max(), 400)
    ax.plot(xx, kde(xx), color="tab:green", linewidth=2, label="KDE")

    mu, sigma = x.mean(), x.std(ddof=1)
    ax.plot(xx, stats.norm.pdf(xx, loc=mu, scale=sigma), color="tab:orange", linewidth=2, label="Normal fit")

    a, b = x.min(), x.max()
    ax.plot(xx, stats.uniform.pdf(xx, loc=a, scale=b - a), color="tab:red", linewidth=2, linestyle="--", label="Uniform fit")

    ax.set_title("Empirical Density vs Normal/Uniform")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "density_comparison.png", dpi=150)
    plt.close(fig)

    # 3) Distribuzione empirica discreta (top 30 frequenze per mantenere la leggibilità).
    discrete = x.value_counts(normalize=True).sort_values(ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(discrete.index.astype(str), discrete.values, color="tab:purple", alpha=0.8)
    ax.set_title("Empirical Discrete Distribution (Top 30 values)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Relative frequency")
    ax.tick_params(axis="x", rotation=80)
    fig.tight_layout()
    fig.savefig(out_dir / "discrete_distribution.png", dpi=150)
    plt.close(fig)

    # 4) Boxplot + Q-Q plot per l'ispezione di outlier e normalità.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot(x, vert=True)
    axes[0].set_title("Boxplot")
    axes[0].set_ylabel("Value")

    stats.probplot(x, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Normal)")

    fig.tight_layout()
    fig.savefig(out_dir / "outliers_qqplot.png", dpi=150)
    plt.close(fig)

    # 4b) Boxplot globale per evidenziare la presenza di outlier globali secondo il criterio IQR, utile per confrontare con i risultati tabulari e identificare visivamente eventuali valori anomali.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(x, vert=True)
    ax.set_title("Global Outliers - Boxplot on Levels")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(out_dir / "global_outliers_boxplot.png", dpi=150)
    plt.close(fig)

    # 5) Grafico di validazione del trend con linea di regressione lineare sovrapposta alla serie temporale, per valutare visivamente la presenza di un trend significativo e confrontarlo con i risultati della regressione lineare calcolata nella funzione _trend_validation.
    years = x.index.to_numpy(dtype=float)
    values = x.to_numpy(dtype=float)
    slope, intercept, _, _, _ = stats.linregress(years, values)
    trend_line = intercept + slope * years

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x.index, x.values, color="tab:blue", linewidth=2, label="Observed")
    ax.plot(x.index, trend_line, color="tab:orange", linestyle="--", linewidth=2, label="Linear trend")
    ax.set_title("Trend Validation: Observed vs Linear Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trend_validation.png", dpi=150)
    plt.close(fig)

    # 6) Grafico dei local outliers basato sui cambiamenti YoY, con evidenziazione degli outlier locali identificati rispetto alla baseline locale.
    yoy = x.diff().dropna()
    rolling_median = yoy.rolling(window=11, center=True, min_periods=5).median()
    residual = yoy - rolling_median
    rolling_mad = residual.abs().rolling(window=11, center=True, min_periods=5).median()
    rolling_std = residual.rolling(window=11, center=True, min_periods=5).std(ddof=0)
    modified_z = 0.6745 * residual / (rolling_mad + 1e-9)
    rolling_z = residual / (rolling_std + 1e-9)
    use_std_fallback = rolling_mad.fillna(0.0) < 1e-6
    local_score = modified_z.mask(use_std_fallback, rolling_z)
    local_outlier_mask = local_score.abs() > 3.5

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yoy.index, yoy.values, color="tab:blue", linewidth=1.8, label="YoY change")
    ax.plot(
        rolling_median.index,
        rolling_median.values,
        color="tab:green",
        linestyle="--",
        linewidth=1.8,
        label="Rolling median (local baseline)",
    )
    ax.scatter(
        yoy.index[local_outlier_mask],
        yoy[local_outlier_mask],
        color="tab:red",
        s=45,
        label="Local outliers",
        zorder=3,
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("Local Outliers on Year-over-Year Changes")
    ax.set_xlabel("Year")
    ax.set_ylabel("YoY change")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "local_outliers_yoy.png", dpi=150)
    plt.close(fig)

    # 6b) Boxplot dei cambiamenti YoY per evidenziare la presenza di outlier locali secondo il criterio basato sui residui rispetto alla baseline locale.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(yoy, vert=True)
    ax.set_title("Local Outliers - Boxplot on YoY Changes")
    ax.set_ylabel("YoY change")
    fig.tight_layout()
    fig.savefig(out_dir / "local_outliers_boxplot.png", dpi=150)
    plt.close(fig)

# Funzione principale che esegue l'intera analisi descrittiva, orchestrando le funzioni di caricamento, 
# calcolo delle statistiche e generazione dei grafici, e salvando tutti i risultati nei percorsi definiti.
def run_descriptive_analysis(
    paths: DescriptivePaths,
    target: str = "production_total",
) -> dict[str, Path]:

    paths.results_metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.results_plots_dir.mkdir(parents=True, exist_ok=True)

    series = load_target_series(paths.dataset_path, target=target)

    freq_df = _frequency_distribution(series) # Calcola la distribuzione di frequenza della serie, suddividendo i valori in classi (bin) e contando le frequenze assolute e relative.
    central_df = _central_tendency(series) # Calcola le misure di tendenza centrale della serie, come media, mediana e moda.
    dispersion_df = _dispersion_measures(series) # Calcola le misure di dispersione della serie, come varianza, deviazione standard e intervallo interquartile.
    outliers_df, outlier_summary_df = _outlier_table_iqr(series) # Identifica gli outlier globali utilizzando il criterio IQR e crea una tabella riepilogativa.
    trend_summary_df = _trend_validation(series) # Valida la presenza di un trend nella serie temporale utilizzando una regressione lineare.
    yoy_details_df, local_outliers_df, local_outliers_summary_df = _local_outliers_on_variation(series) # Identifica gli outlier locali basati sui cambiamenti YoY rispetto alla baseline locale.

    #   Salvataggio dei risultati tabulari.
    output_paths = {
        "series": paths.results_metrics_dir / "series_clean.csv",
        "frequency": paths.results_metrics_dir / "frequency_distribution.csv",
        "central_tendency": paths.results_metrics_dir / "central_tendency.csv",
        "dispersion": paths.results_metrics_dir / "dispersion_measures.csv",
        "outliers": paths.results_metrics_dir / "outliers_iqr.csv",
        "outlier_summary": paths.results_metrics_dir / "outliers_summary.csv",
        "trend_summary": paths.results_metrics_dir / "trend_summary.csv",
        "yoy_variation": paths.results_metrics_dir / "yoy_variation_details.csv",
        "local_outliers": paths.results_metrics_dir / "local_outliers_yoy.csv",
        "local_outlier_summary": paths.results_metrics_dir / "local_outliers_summary.csv",
    }

    series.reset_index().to_csv(output_paths["series"], index=False)
    freq_df.to_csv(output_paths["frequency"], index=False)
    central_df.to_csv(output_paths["central_tendency"], index=False)
    dispersion_df.to_csv(output_paths["dispersion"], index=False)
    outliers_df.to_csv(output_paths["outliers"], index=False)
    outlier_summary_df.to_csv(output_paths["outlier_summary"], index=False)
    trend_summary_df.to_csv(output_paths["trend_summary"], index=False)
    yoy_details_df.to_csv(output_paths["yoy_variation"], index=False)
    local_outliers_df.to_csv(output_paths["local_outliers"], index=False)
    local_outliers_summary_df.to_csv(output_paths["local_outlier_summary"], index=False)

    _save_distribution_plots(series, freq_df, paths.results_plots_dir)

    return output_paths
