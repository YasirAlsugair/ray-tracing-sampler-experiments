"""Repository-local inputs for the poster v7 candidate."""
from __future__ import annotations

from pathlib import Path


POSTER = Path(__file__).resolve().parents[1]
DATA = POSTER / "artifacts" / "data_v6"

# V7 reuses the frozen measured summaries introduced with v6.
DIMENSION_FITS = DATA / "dimension_fits.json"
ARM_SUMMARY = DATA / "matched_condition_summary.json"
FRONTIER_REAL = DATA / "superconductor_frontier.csv"

OUT_DIR = POSTER / "artifacts" / "figures_v7"
