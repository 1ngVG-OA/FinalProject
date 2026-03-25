"""Compare Step 3 statistical baseline vs extended experiments."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Project.models.statistical.analysis import build_baseline_extended_comparison, format_comparison_summary
from Project.preprocessing.descriptive_analysis import load_target_series

TARGET_SERIES_KEY = "production_total"


def run_compare() -> None:
    metrics_dir = ROOT / "Results" / "metrics" / "statistical"
    raw = load_target_series(ROOT / "Datasets" / "Tavola_1.14.csv", target=TARGET_SERIES_KEY)
    combined, display = build_baseline_extended_comparison(raw, metrics_dir)

    print(format_comparison_summary(combined, display))

    out_path = metrics_dir / "comparison.csv"
    display.drop(columns="experiment", errors="ignore").to_csv(out_path, index=False)
    print(f"Saved comparison: {out_path}")


if __name__ == "__main__":
    run_compare()