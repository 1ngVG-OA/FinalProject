"""Main pipeline package.

Exports:
- per-series entry points (fully self-contained, customisable independently)
"""

from Mains.main_demographic import run_pipeline_demographic
from Mains.main_ogrin import run_pipeline_ogrin

__all__ = [
    "run_pipeline_demographic",
    "run_pipeline_ogrin",
]
