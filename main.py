"""Top-level orchestrator that runs all series-specific pipelines."""

from __future__ import annotations

import json

import pandas as pd

from config import RESULTS_DIR
from Mains.main_demographic import run_pipeline_demographic
from Mains.main_ogrin import run_pipeline_ogrin


def run_all_pipelines() -> tuple[pd.DataFrame, dict]:
    """Run all series-specific pipelines and persist aggregate outputs.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Combined metrics DataFrame and combined parameter report.
    """
    demographic_metrics, demographic_params = run_pipeline_demographic()
    ogrin_metrics, ogrin_params = run_pipeline_ogrin()

    metrics_df = pd.concat([demographic_metrics, ogrin_metrics], ignore_index=True)
    metrics_df = metrics_df.sort_values(["series", "test_rmse"], ascending=[True, True]).reset_index(drop=True)
    metrics_df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    all_params = {**demographic_params, **ogrin_params}
    with open(RESULTS_DIR / "best_models_params.json", "w", encoding="utf-8") as f:
        json.dump(all_params, f, indent=2, default=str)

    print("All pipelines completed.")
    print(f"Saved: {(RESULTS_DIR / 'metrics_summary.csv')}")
    print(f"Saved: {(RESULTS_DIR / 'best_models_params.json')}")

    return metrics_df, all_params


if __name__ == "__main__":
    run_all_pipelines()
