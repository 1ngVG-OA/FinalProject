"""Configurazione centrale della pipeline forecasting multi-serie.

Il modulo definisce le serie supportate, la regola di naming degli output e
helper condivisi per risolvere dataset, cartelle `Results/<serie>/...` e
`Datasets/processed/<serie>/...`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


# ------------------------------------------------------------------
# Path base di progetto
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Datasets"
RESULTS_DIR = BASE_DIR / "Results"
PROCESSED_DIR = DATA_DIR / "processed"


# ------------------------------------------------------------------
# Costanti globali
# ------------------------------------------------------------------

SEED = 1234
DEFAULT_SERIES_KEY = "production_total"


# ------------------------------------------------------------------
# Configurazione serie
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SeriesConfig:
    """Configurazione di una singola serie studiata dalla pipeline."""

    key: str
    dataset_path: Path
    target_column_index: int
    target_series_name: str
    display_name: str
    frequency: str | None = None
    notes: str | None = None
    output_name: str | None = None


SERIES_REGISTRY: dict[str, SeriesConfig] = {
    "production_total": SeriesConfig(
        key="production_total",
        dataset_path=DATA_DIR / "Tavola_1.14.csv",
        target_column_index=1,
        target_series_name="produzione_lorda_totale",
        display_name="Produzione lorda totale energia",
        frequency="YE",
        notes="Serie annuale ISTAT derivata da Tavola_1.14.",
    ),
    "consumption_total": SeriesConfig(
        key="consumption_total",
        dataset_path=DATA_DIR / "Tavola_1.14.csv",
        target_column_index=12,
        target_series_name="consumo_totale",
        display_name="Consumo totale energia",
        frequency="YE",
        notes="Ultima colonna di Tavola_1.14; valori osservati dal 1931.",
    ),
}


# ------------------------------------------------------------------
# Helper naming e path
# ------------------------------------------------------------------

def derive_output_name(series_key: str) -> str:
    """Deriva un nome cartella leggibile e stabile dalla chiave serie."""

    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", str(series_key).strip()) if token]
    if not tokens:
        raise ValueError("series_key must contain at least one alphanumeric token")
    return "".join(token[:1].upper() + token[1:] for token in tokens)


def get_series_config(series_key: str = DEFAULT_SERIES_KEY) -> SeriesConfig:
    """Restituisce la configurazione della serie richiesta."""

    try:
        return SERIES_REGISTRY[series_key]
    except KeyError as exc:
        available = ", ".join(sorted(SERIES_REGISTRY))
        raise KeyError(f"Unsupported series_key '{series_key}'. Available series: {available}") from exc


def get_series_output_name(series_key: str = DEFAULT_SERIES_KEY) -> str:
    """Restituisce il nome cartella output associato alla serie."""

    cfg = get_series_config(series_key)
    return cfg.output_name or derive_output_name(cfg.key)


def get_results_root(series_key: str = DEFAULT_SERIES_KEY) -> Path:
    """Restituisce la root Results dedicata alla serie."""

    return RESULTS_DIR / get_series_output_name(series_key)


def get_processed_root(series_key: str = DEFAULT_SERIES_KEY) -> Path:
    """Restituisce la root dei dati preprocessati dedicata alla serie."""

    return PROCESSED_DIR / get_series_output_name(series_key)


def get_results_subdir(series_key: str, category: str, step: str) -> Path:
    """Restituisce una sottocartella `Results/<serie>/<category>/<step>`."""

    return get_results_root(series_key) / category / step
