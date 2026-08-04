# exp7 analysis tasks

Scoped follow-up analysis on the finished exp7 chains (Gaia XP to [alpha/M]
regression, sampled with minibatch ray tracing). Background and the full run
story: `exp7_reference_run.md` and the executed `exp7_gaia_posterior.ipynb`.
The chains themselves are not in git (0.5 GB); everything below runs from
two small files that are.

## Files

`results/tables/exp7_analysis_pack.npz` (24 MB), one array per line:

    member_preds_scatter   (50, 25232)  THE CHAIN: corrected noise model,
                                        converged (all drift checks, acc 0.99)
    member_s_scatter       (50,)        each member's intrinsic scatter s
    member_preds_gated5e6  (50, 25232)  previous record (collapsed spread)
    member_preds_normal    (50, 25232)  N(0,1)-prior ablation (comparison only)
    grad_mean_std_coords   (25232, 110) member-mean d(pred)/d(std. coefficient)
    norm_mu, norm_sd       (110,)       the standardization vectors
    test_y, test_yerr      (25232,)     APOGEE [alpha/M] and its error
    teff, mh, sids         (25232,)     Teff, [M/H], Gaia DR3 source_id

`results/tables/exp7_gaia_spectra_payload.npz` (21 MB): raw `coeffs` and
`coeff_errs` (25232, 110) for the same test stars, for GaiaXpy.

Members are 50 evenly thinned states from the chain's final quarter. The
predictive for star i: mean over members = mu_i, spread over members =
tau_i, and the total claimed variance now includes the sampled intrinsic
scatter: tau_i^2 + test_yerr_i^2 + s^2. The chain of record converged
(misfit, weighted norm, and s all pass the drift rule at acceptance
0.99), its spread is genuine (median tau 0.012), and its overall
calibration is healthy: z std 0.948.

```python
import numpy as np
p = np.load("results/tables/exp7_analysis_pack.npz")
mu = p["member_preds_scatter"].mean(axis=0)
tau = p["member_preds_scatter"].std(axis=0)
s_med = np.median(p["member_s_scatter"])
z = (p["test_y"] - mu) / np.sqrt(tau**2 + p["test_yerr"]**2 + s_med**2)
```

## Task 1: calibration anatomy

Update: the global problem is solved. The corrected chain calibrates at
z std 0.948 with a single sampled s = 0.045. The question is now finer
and more interesting: is that health uniform? Bin z std by Teff, [M/H],
label error, and |alpha|; a flat 1.0 across all bins means one global s
truly suffices, structure means s wants to be a function of stellar
parameters. Second question, the tails: the stars beyond |z| of 4 (the
half-dex misses that dominate sigma_sto) are the case file for a
Student-t likelihood; count them, locate them in the HR diagram, and
check overlap with the star-card gallery's worst misses.

## Task 2: star-card gallery

One card = the star's calibrated spectrum plus an inset histogram of its
50 member predictions against the APOGEE value. Section 8 of the notebook
has the working GaiaXpy pattern (archive column names, first 55
coefficients are BP, feed zeros for the correlations and say on the figure
that the error band is approximate). Good sets: largest |mu - y|, largest
tau, lowest and highest alpha, and a few ordinary stars as controls.

## Task 3: saliency across the HR diagram

`grad_mean_std_coords` is the network's input gradient per star, averaged
over the chain's members, in standardized coordinates; divide by
`norm_sd` to get raw coefficient space. Calibration is linear in the
coefficients, so a gradient vector can be fed through gaiaxpy `calibrate`
exactly like a spectrum, giving sensitivity vs wavelength. Bin stars by
Teff (and by [M/H] if it looks interesting), average the gradient within
each bin, calibrate each average, and overlay. The one-star version
(notebook section 8) concentrates below 450 nm; the question is whether
that holds across the temperature range.

## Task 4: the alpha signature at finer metallicity

Notebook section 8 builds a shape-normalized high-vs-low-alpha stack in
one Teff and [M/H] box. Repeat in [M/H] slices (say four bins over -0.6 to
+0.3, matched Teff within each) to see whether the blue-end signature
moves or scales with metallicity.

## Caveats that must survive into any figure

Update (2026-08-04): the chain of record is now the HETEROSCEDASTIC
chain (the network predicts a per-star scatter sigma(x)); its 50
members are in `results/tables/exp7h_pack3.npz` (converged at 16M
steps; z std 1.09 with flat calibration across label-error bins). Load
them with `experiments/exp7_gaia_hetero.py`'s `make_model`/`load_flat`
and compute var_i = tau_i^2 + err_i^2 + sigma(x_i)^2 per star. The
`member_preds_scatter` arrays here are the global-s chain: still the
best point predictor (RMSE 0.0466) and the right comparison baseline,
but its per-star calibration is uneven (z std 0.39 to 1.43 across
label-error bins). The gated5e6 arrays are the older record (spread
collapsed by the gate, comparison only), and the N(0,1) chain is a
prior ablation, not a competitor. Changes to the sampler or the target
definition remain out of scope here and go through Yasir and Josh.
