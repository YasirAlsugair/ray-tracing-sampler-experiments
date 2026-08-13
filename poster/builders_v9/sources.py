"""Every artifact this poster reads, named explicitly.

Nothing here is discovered from the working tree. A figure may not be built from
an artifact set that is not listed below.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDIES = ROOT / "studies"
EXTERNAL = ROOT / "external" / "ray-tracing-sampler-experiments"

# Noise tolerance against step size: sigma_c per sampler per step, jitter arm.
SCALING_FITS = STUDIES / "06_blr_noise_tolerance_scaling/artifacts/full/scaling_fits.json"
SCALING_DECISIONS = STUDIES / "06_blr_noise_tolerance_scaling/artifacts/full/decisions.json"

# Compute frontier on real data (UCI Superconductivity) and its synthetic twin.
FRONTIER_REAL = STUDIES / "11_superconductor_compute_frontier/artifacts/full/frontier.csv"
FRONTIER_SYNTHETIC = STUDIES / "07_blr_minibatch_compute_frontier/artifacts/full/frontier.csv"
FRONTIER_SYNTHETIC_FITS = \
    STUDIES / "07_blr_minibatch_compute_frontier/artifacts/full/frontier_fits.json"

# 1D Bayesian neural network predictive bands against a trust-gated gold posterior.
BNN_BANDS = STUDIES / "16_bnn_gap_predictive_bands/artifacts/full/bands.npz"
BNN_SUMMARY = STUDIES / "16_bnn_gap_predictive_bands/artifacts/full/summary.json"

# A deliberately harder synthetic function for the poster introduction.  Its
# central withheld interval contains two turns, so one optimized network and a
# posterior over networks make visibly different claims.  The posterior is an
# independently trust-gated NUTS reference and is stored in raw response units.
HARDER_BNN_POSTER = \
    STUDIES / "25_bnn_harder_truth_poster/artifacts/full/poster_data.npz"
HARDER_BNN_SUMMARY = \
    STUDIES / "25_bnn_harder_truth_poster/artifacts/full/summary.json"

# Direction of the error when the exactness gate is dropped: RT narrow, HMC wide.
STATIONARY_BIAS = STUDIES / "05_blr_isotropic_stationary_bias/artifacts/full/summary.csv"

# The step exponent and noise tolerance at D in {16, 81, 256, 1024}, synthetic jitter.
DIMENSION_FITS = STUDIES / "15_noise_tolerance_dimension_scaling/artifacts/full/fits.json"

# The step exponent measured without the step/trajectory confound.
ARM_SUMMARY = STUDIES / "22_matched_condition_noise_arms/artifacts/full/summary.json"

# Gaia [alpha/M] regression: one global residual-scatter parameter versus a
# heteroscedastic network that learns a per-star residual scatter.  These are
# upstream sampler artifacts mirrored from YasirAlsugair/ray-tracing-sampler-experiments.
GAIA_ANALYSIS_PACK = EXTERNAL / "results/tables/exp7_analysis_pack.npz"
GAIA_SPECTRA_PAYLOAD = EXTERNAL / "results/tables/exp7_gaia_spectra_payload.npz"
GAIA_HETERO_PACK = EXTERNAL / "results/tables/exp7h_pack3.npz"

# UCI Superconductivity raw table. Unlike everything above this is not a study
# artifact: the mini-batch gradient figure recomputes gradients from the rows
# themselves, through rtbench's loader at blr_superconductor(n=2048).
SUPERCONDUCTIVITY_RAW = ROOT / "data/raw/superconductivity_uci.zip"

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "figures_v9"
