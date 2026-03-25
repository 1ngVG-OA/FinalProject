"""Run Step 3 statistical diagnostics for baseline vs extended experiments."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Project.models.statistical.analysis import build_diagnostic_tables, format_diagnostic_summary
from Project.preprocessing.descriptive_analysis import load_target_series

TARGET_SERIES_KEY = "production_total"


def run_diagnostic() -> None:
    metrics_dir = ROOT / "Results" / "metrics" / "statistical"
    raw = load_target_series(ROOT / "Datasets" / "Tavola_1.14.csv", target=TARGET_SERIES_KEY)
    diagnostic, shortlist = build_diagnostic_tables(raw, metrics_dir)

    out_diag = metrics_dir / "diagnostic.csv"
    out_shortlist = metrics_dir / "shortlist.csv"
    diagnostic.to_csv(out_diag, index=False)
    shortlist.to_csv(out_shortlist, index=False)

    print(format_diagnostic_summary(shortlist))
    print(f"Saved diagnostic: {out_diag}")
    print(f"Saved shortlist: {out_shortlist}")


if __name__ == "__main__":
    run_diagnostic()