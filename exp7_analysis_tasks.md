# exp7 analysis tasks

Scoped follow-up analysis on the finished exp7 chains (Gaia XP to [alpha/M]
regression, sampled with minibatch ray tracing). Background and the full run
story: `exp7_reference_run.md` and the executed `exp7_gaia_posterior.ipynb`.
The chains themselves are not in git (0.5 GB); everything below runs from
two small files that are.

## Files

`results/tables/exp7_analysis_pack.npz` (20 MB), one array per line:

    member_preds_gated5e6  (50, 25232)  THE CHAIN: gated dt 5e-6, Eq. 33 on
    member_preds_normal    (50, 25232)  N(0,1)-prior ablation (comparison only)
    grad_mean_std_coords   (25232, 110) member-mean d(pred)/d(std. coefficient)
    norm_mu, norm_sd       (110,)       the standardization vectors
    test_y, test_yerr      (25232,)     APOGEE [alpha/M] and its error
    teff, mh, sids         (25232,)     Teff, [M/H], Gaia DR3 source_id

`results/tables/exp7_gaia_spectra_payload.npz` (21 MB): raw `coeffs` and
`coeff_errs` (25232, 110) for the same test stars, for GaiaXpy.

Members are 50 evenly thinned states from the chain's final quarter. The
predictive for star i: mean over members = mu_i, spread over members =
tau_i, total claimed variance = tau_i^2 + test_yerr_i^2. One property of
the chain of record to carry into every caption: the Eq. 33 gate ran at
its noise ceiling, which slows mixing, so tau is a LOWER BOUND on the
posterior spread (an unadjusted control arm, not shipped here, gives
spreads about 4x larger with the same predictions).

```python
import numpy as np
p = np.load("results/tables/exp7_analysis_pack.npz")
mu = p["member_preds_gated5e6"].mean(axis=0)
tau = p["member_preds_gated5e6"].std(axis=0)
z = (p["test_y"] - mu) / np.sqrt(tau**2 + p["test_yerr"]**2)
```

## Task 1: calibration anatomy

Known: z has std about 6 overall (should be 1; about 4.4 even under the
control arm's larger spreads, so the miscalibration is real, not a
thinning artifact). Unknown: where the excess lives. Bin z by Teff,
[M/H], label error, and |alpha| itself; plot z std per bin. The question
to answer in one paragraph: is the miscalibration flat (one global
intrinsic scatter s, with sigma_i^2 = err_i^2 + s^2, fixes it; estimate s
from the residuals) or structured (s must depend on stellar parameters,
which changes the next sampling run).

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

The chain of record is `member_preds_gated5e6`, Metropolis-checked
throughout; its tau is a lower bound (see above), and the chain's misfit
was still drifting slowly at the end of the runs, so all spreads are
provisional; say so in captions. The N(0,1) chain is a prior ablation,
not a competitor. Changes to the sampler or the target definition are out
of scope here and go through Yasir and Josh.
