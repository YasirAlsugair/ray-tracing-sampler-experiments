# exp7 reference run (2026-07-26/27, overnight)

Minibatch ray tracing on the Gaia [alpha/M] regression posterior, RunPod
5090s, about 6 GPU hours and $6 across two pods (both stopped). This is the
reference for the exp7 workbook; do the workbook by hand, then compare.
The presentation notebook is exp7_gaia_posterior.ipynb (fully executed).

## Setup

Sample: the paper's pristine cut reproduced at 126,156 stars (paper
123,804, within 1.9 percent; the "good" stage lands 263,722 vs 202,970
because this table's error columns are not exactly the astroNN ones). Split
seed 2003, 80/20 = 100,924 / 25,232, X standardized on train. Payload:
results/tables/exp7_gaia_pristine.npz. Cut script:
experiments/exp7_gaia_prep.py (workbook section 0 answer key, do not peek
early).

Target: Gaussian NLL with sigma = ALPHA_M_ERR per star, T = 1. Model: MLP
110-64-64-1 tanh, D = 11,329. Priors: Glorot per layer (bias sigma 0.1),
and N(0,1) for the head-to-head. Sampler: vendor minibatch Raytracer,
batch 1024, Eq. 33 gate every 30 steps. Driver:
experiments/exp7_gaia_driver.py.

## Scorecard (held-out 25,232 stars)

    RMSE   global mean            0.0927
    RMSE   kNN (k=10)             0.0742
    RMSE   N(0,1) chain           0.0704   (500k steps, prior ablation)
    RMSE   THE CHAIN (gated 5e-6) 0.0492   (2M steps, Eq. 33 on, acc 0.38)
    z std: chain 6.2 (spread is a lower bound), N(0,1) 7.8  (1.0 = calibrated)
    member spread (median): chain 0.0020 (lower bound), vs label err 0.0064

Validation: an unadjusted control arm (dt 1e-4, 500k steps, local file
exp7_rtub_dt0.0001.npz, not in the analysis pack) reproduces the chain's
predictions to one percent (RMSE 0.0486) with ~4x larger member spreads
(z std 4.4), so the gate changed the cost, not the answer. The chain
beats k-NN by a third. The uncertainties are at least 4x too small:
residuals ~0.049 vs claimed ~0.010. An intrinsic scatter s ~ 0.05 dex
(workbook section 2, choice 2) is the obvious next run.

## The findings

1. The prior decides what the budget buys. Glorot: born on its shell
   (weighted norm 11,235 of D = 11,329), whole budget spent fitting, RMSE
   0.049. N(0,1), same sampler and budget: starts 99 percent saturated,
   ends at 2.9x the misfit (4.78M vs 1.64M nats), RMSE 0.070, barely at
   k-NN level, z std 7.8, pinned to its own shell (11,233) throughout.
   exp6's 20,000-trajectory walk was a prior problem, not a sampler
   problem.

2. Eq. 33 is a diagnostic, not a filter, on this heteroscedastic target.
   Blind at the prior draw (sigma_sto 26M nats for Glorot, 3.5B for
   N(0,1); every dt accepts ~1.0, so the tuning ladder must run from a
   FITTED state). At the fit (sigma_sto 135k) it correctly all-rejects
   the dt 1e-4 dynamics: the luminosity ledger claims ~10M spurious nats
   per 30-step window. Tuned by the paper's recipe the ladder reads
   0.04 / 0.20 / 0.36 / 0.38 / 0.38 for dt 2e-5 to 1e-6: recovery right
   where the dt^2 bias scaling predicts, but the plateau 0.38 is the
   gate's own noise ceiling, so the tuned gate's decisions are coin
   flips. The 2M-step production arm at dt 5e-6 held acceptance 0.38,
   still crept in norm (19.2k to 24.4k), and collapsed the member spread
   to 0.002. Echoes Behroozi dropping MH at GPT-2 scale, now with a
   mechanism. Never read a drift rule without acceptance next to it: a
   frozen all-reject chain passes any stationarity test.

3. The chain of record keeps the gate (per Yasir's call): its decisions
   at the tuned dt are noise, but the agreement with the unadjusted
   control (0.0492 vs 0.0486) is itself evidence the knob-controlled
   dynamics were clean. The cost is a 20x smaller step, so the member
   spread is a lower bound; the misfit was also still drifting slowly at
   the end, so posterior spreads are provisional twice over.

4. GaiaXpy: coefficients are stored BP-first (swapped halves produce a
   seam at 640 nm). At fixed Teff and [M/H], high-alpha giants are 20-30
   percent brighter below 450 nm (shape-normalized stacks), and the
   network's saliency, obtained by calibrating the input gradient
   (calibration is linear in coefficients), concentrates in the same blue
   region. Data and model agree where alpha lives.

## Files

    exp7_gaia_posterior.ipynb                     the presentation notebook
    results/tables/exp7_rt33_dt0.0001.npz         Glorot gated base arm, 500k
    results/tables/exp7_rt33_dt0.0001_part2.npz   frozen all-reject leg
    results/tables/exp7_rtub_dt0.0001.npz         Glorot ungated control, 500k
    results/tables/exp7_rt33_dt5e-06.npz          Glorot gated production, 2M
    results/tables/exp7n_rt33_dt0.0001.npz        N(0,1) chain, 500k
    results/tables/exp7_gaia_spectra_payload.npz  test-split coeffs for GaiaXpy

## Open items

- Intrinsic scatter s as an 11,330th parameter, then recheck z std.
- Longer ungated run (or exact-chain finisher) before quoting spreads.
- Early NaNs in the window ledger at the saturated start (auto-accepted by
  an already-blind gate); harmless here, worth a guard.
