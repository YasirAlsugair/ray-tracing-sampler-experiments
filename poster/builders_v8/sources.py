"""Portable, repository-local inputs for poster v8."""
from __future__ import annotations

from pathlib import Path


POSTER = Path(__file__).resolve().parents[1]
DATA = POSTER / "artifacts" / "data_v6"

# Plot-ready inputs frozen in v6 and reused without changing any values.
HARDER_BNN_POSTER = DATA / "harder_bnn_poster.npz"
MINIBATCH_GRADIENT_DATA = DATA / "minibatch_gradient_data.npz"
DIMENSION_FITS = DATA / "dimension_fits.json"
ARM_SUMMARY = DATA / "matched_condition_summary.json"
FRONTIER_REAL = DATA / "superconductor_frontier.csv"
GAIA_ENSEMBLE_50 = DATA / "exp7_ensemble_50.npz"
GAIA_CALIBRATION_SOURCE = DATA / "gaia_calibration_by_group_v4.pdf"
GAIA_NLL_SOURCE = DATA / "gaia_nll_vs_members_v4.pdf"

OUT_DIR = POSTER / "artifacts" / "figures_v8"
