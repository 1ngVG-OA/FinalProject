"""Compare Step 5 neural baseline and extended experiments."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Project.models.neural.analysis import build_baseline_extended_comparison, format_comparison_summary


def run_compare() -> None:
    metrics_dir = ROOT / "Results" / "metrics" / "neural"
    combined, display = build_baseline_extended_comparison(metrics_dir)

    print(format_comparison_summary(combined, display))

    out_path = metrics_dir / "comparison.csv"
    display.to_csv(out_path, index=False)
    print(f"Saved comparison: {out_path}")


if __name__ == "__main__":
    run_compare()