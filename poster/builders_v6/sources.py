"""Portable inputs for the poster v6 candidate.

Every normal figure build reads only committed files below ``poster/``.  The
heavier experiments that produced these frozen inputs live elsewhere and are
recorded in ``README.md``; they are not required to rebuild the poster.
"""
from __future__ import annotations

from pathlib import Path


POSTER = Path(__file__).resolve().parents[1]
DATA = POSTER / "artifacts" / "data_v6"

# Figure 1: posterior draws on the deliberately harder one-dimensional BNN.
HARDER_BNN_POSTER = DATA / "harder_bnn_poster.npz"

# Figure 3: plot-ready summaries of the UCI minibatch-gradient diagnostic.
MINIBATCH_GRADIENT_DATA = DATA / "minibatch_gradient_data.npz"

# Figure 5: measured tolerance, exponent erosion, and compute frontier.
DIMENSION_FITS = DATA / "dimension_fits.json"
ARM_SUMMARY = DATA / "matched_condition_summary.json"
FRONTIER_REAL = DATA / "superconductor_frontier.csv"

# Gaia panels: predictions from the completed 50-member deep ensemble.
GAIA_ENSEMBLE_50 = DATA / "exp7_ensemble_50.npz"
GAIA_CALIBRATION_SOURCE = DATA / "gaia_calibration_by_group_v4.pdf"
GAIA_NLL_SOURCE = DATA / "gaia_nll_vs_members_v4.pdf"

OUT_DIR = POSTER / "artifacts" / "figures_v6"
