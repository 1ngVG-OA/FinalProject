"""Inferential statistics exports for final forecast evaluation."""

from .diebold_mariano import build_diebold_mariano_table, diebold_mariano_test

__all__ = [
    "build_diebold_mariano_table",
    "diebold_mariano_test",
]