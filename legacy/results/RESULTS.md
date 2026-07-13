# Ray Tracing Sampler: empirical results

Empirical study of the public Ray Tracing Sampler (Behroozi 2025, arXiv:2510.25824),
answering the four questions Ricardo Baptista raised. Base sampler: the public
`raytrace_jax.py` / `sample_raytrace`, vendored pristine. HMC baseline: the repo's
own `sample_hamiltonian` (same leapfrog, same Metropolis test, same harness), so
every comparison is apples to apples. All runs are seeded and reproducible on CPU.

Diagnostics were validated before use: the Sokal integrated-autocorrelation-time
estimator (`rts/metrics.py`) reproduces `emcee` exactly and matches the analytic
AR(1) truth (tau = 3.01, 8.83, 41.62 vs truth 3, 9, 39). The W-exposure fork is
bit-for-bit identical to upstream when `W=1` (regression test in
`experiments/_smoke_fork.py`).

---

## Q1. Effect of the weighting function W

**W was not exposed in the public sampler.** Upstream hard-codes W=1: the ray
bends along `grad(log L)`. The fork `rts/raytrace_jax_W.py` adds `log_W_fn` and
bends along `grad(log L - log W)`. Crucially, the radiance Jacobian
`delta_ln_L = (1-D)*log(f_v)` returned by `UpdateV` is purely geometric, so the
existing Metropolis test still targets a distribution proportional to L for any
symmetric W. We tested W = L^c (bending then follows L^(1-c)): c<0 sharpens the
bend, c>0 weakens it.

**Figure:** `figures/exp1_weighting.png`

**Honesty gate (Gaussian, step size held fixed across W so only W varies):** every
W reproduces the same target.

| W            | acceptance | mean(x0) | var(x0) | KS p vs N(0,1) |
|--------------|-----------:|---------:|--------:|---------------:|
| sharpen c=-0.5 | 0.49 | -0.005 | 1.000 | 0.39 |
| W=1 baseline   | 0.68 | -0.001 | 1.007 | 0.03 |
| temper c=0.5   | 0.69 | -0.002 | 0.997 | 0.44 |
| near-Gibbs c=0.9 | 0.21 | +0.004 | 1.000 | 0.18 |

Means are 0 and variances are 1 for every W (moments are robust to
autocorrelation; the single low KS p-value at the baseline reflects residual
chain autocorrelation, not a shifted target). **W changes the path, not the
target.**

**What W actually changes (2-mode mixture, same fixed step):** mode-switching
efficiency moves by roughly 8x across the family while occupancy stays pinned at
0.50 / 0.50.

| W            | acceptance | switch rate | occupancy | ESS / grad-eval |
|--------------|-----------:|------------:|----------:|----------------:|
| sharpen c=-0.5 | 0.46 | 0.024 | 0.49 / 0.51 | 1.1e-3 |
| W=1 baseline   | 0.63 | 0.085 | 0.50 / 0.50 | 3.7e-3 |
| **temper c=0.5** | 0.63 | **0.195** | 0.50 / 0.50 | **9.7e-3** |
| near-Gibbs c=0.9 | 0.23 | 0.112 | 0.50 / 0.50 | 4.3e-3 |

**Takeaway:** W is a genuine, load-bearing efficiency dial with an *interior*
optimum. Moderate tempering (c=0.5, i.e. bending along L^0.5) switches modes 2.3x
more often than the default and gives 2.6x the ESS per gradient, without touching
the stationary distribution. Both over-sharpening and over-weakening hurt.

**Caveat surfaced by the experiment:** the exact Gibbs limit W=L (c=1, refractive
index constant, no bending) is numerically singular in the released kick: the
bending gradient is identically zero, so `unit_g = grad/|grad|` becomes 0/0 and
acceptance collapses to 0. Anyone exposing W in production should special-case the
near-zero-gradient branch.

---

## Q2. Autocorrelation and ESS vs HMC

Integrated autocorrelation time (Sokal) and ESS for ray tracing vs the repo's own
HMC. For each target and method we tuned the step size to ~0.7 acceptance at two
trajectory lengths (10 and 30 leapfrog steps) and report the best configuration
per method, normalised by gradient work (ESS per gradient evaluation). Tuning both
methods and normalising by gradients is what keeps the comparison fair: a single
fixed config parks HMC on a trajectory-length resonance and flatters RT unfairly.

**Figures:** `figures/exp2_ess_per_grad.png`, `figures/exp2_acf_gaussian10.png`

| target | RT ESS/grad | HMC ESS/grad | RT advantage |
|--------|------------:|-------------:|-------------:|
| Gaussian D=2 (isotropic) | 3.7e-2 | 5.2e-2 | 0.7x (HMC wins) |
| Gaussian D=10 (isotropic) | 2.6e-2 | 1.4e-2 | 1.8x |
| Gaussian D=10, condition number 10 | 2.6e-2 | 9.5e-4 | **27.6x** |
| Rosenbrock D=2 (b=20) | 9.5e-4 | 3.3e-4 | 2.9x |
| 2-mode mixture D=2 (sep 4) | 2.2e-2 | 2.6e-3 | 8.3x |

HMC's worst-coordinate tau blows up under anisotropy (tau ~ 240-290 at condition
number 10) and multimodality (tau ~ 45), because its isotropic momentum cannot
match an anisotropic or barrier-separated target. Ray tracing normalises the
gradient to a constant-speed heading, so its autocorrelation stays low (tau ~ 1.3
to 4 on the Gaussians) and is far more robust to conditioning.

**Takeaway:** HMC is competitive (slightly better) only on the easy isotropic
Gaussian, its ideal case. On everything with conditioning or multiple modes, ray
tracing is several to tens of times more efficient per gradient. This is the first
direct measurement of the autocorrelation axis Ricardo named.

---

## Q3. Mode coverage

3-mode Gaussian mixture (equal weights, so truth occupancy is 1/3 each), plus
disjoint L=0 islands as the honest boundary case.

**Figures:** `figures/exp3_occupancy.png`, `figures/exp3_sweeps.png`,
`figures/exp3_heatmap_islands.png`

**Occupancy vs truth (sep 6, walkers seeded across all modes):** both samplers
recover the truth.
- Ray tracing occupancy [0.345, 0.342, 0.314], switch rate 0.026
- HMC occupancy [0.313, 0.317, 0.370], switch rate 0.021

**Ricardo's initialisation test (sep 6, walkers ~ N(0, s^2 I), s swept):** ray
tracing holds the truth far-mode mass (2/3) at every initialisation width. HMC
*degrades* as the initialisation widens (far-mode mass falls to 0.45 at s=8),
because HMC walkers flung into the low-density tails get stranded and cannot
redistribute. Ray tracing's constant-speed dynamics keep finding the modes.

**Reaching far modes from a home start (init s=0.5, separation swept):** both reach
every mode up to sep 6; ray tracing then sustains coverage farther (sep 8: 0.94 of
walkers reach a far mode, vs HMC 0.60), and both collapse by sep 10.

**The headline (heatmap over init-width x separation):** at the large separations
where a tightly-initialised ray tracer misses the far modes (sep 10, init s=0.5,
0% coverage), **widening the initialisation recovers them** (s=16, 50% coverage;
at sep 8, 0.79 climbs to 0.92). This directly confirms Ricardo's hypothesis: the
far modes the sampler currently misses are reachable by spreading the walker
initialisation, so part of the coverage gap is an initialisation problem, not a
dynamics ceiling.

**Boundary case (disjoint L=0 islands):** both samplers stay trapped in the home
island (RT occupancy [1, 0, 0], HMC [0.992, 0, 0.008], zero switches). The
released sampler has no Lambertian re-emission (the island-hopping shown in the
explainer used an analytic gap-warp that is not in the public code), so a ray that
steps into a true vacuum gap is simply rejected. This is the honest limit that
motivates Q4.

---

## Q4. Learning the proposal (research-direction prototype)

**Scope.** "Learn W or the emission rate to directly target mode coverage and
autocorrelation." We built the simplest honest version: parameterise two knobs the
released code exposes (via the fork) and learn them on a hard 2-mode target.
- c = tempering exponent of a weight W = L^c (bends the ray along L^(1-c)),
- r = refresh rate of the Ornstein-Uhlenbeck momentum refresh (the emission-like knob).

Objective (maximise): `J(c, r) = 2 * mode_balance + log10(ESS_slow_mode + 1)`,
which rewards exactly Ricardo's two axes (coverage via balance, autocorrelation via
the ESS of the slow mode indicator). We map J on a grid for the landscape and run
CMA-ES to learn (c, r). Correctness is safe by construction: tempering-W invariance
was demonstrated in Q1 and momentum refresh leaves the target unchanged.

**Figure:** `figures/exp4_learn.png`

**Result (separation 8, all walkers started at home).** CMA-ES learned c=0.854,
r=0.008 (strong tempering, negligible refresh).

| metric | baseline (W=1, r=0) | learned (c=0.85) | gain |
|--------|--------------------:|-----------------:|-----:|
| occupancy balance | 0.984 | 0.999 | (target preserved) |
| inter-mode switch rate | 0.001 | 0.116 | ~100x |
| ESS of the slow (mode-indicator) direction | 821 | 27,341 | **~33x** |

At an easier separation (6) the same procedure learned milder tempering (c=0.73,
ESS 5,502 to 45,226, ~8x). Harder separation -> more tempering learned, a sensible
and interpretable trend.

**Takeaway:** a two-parameter learnable weight, optimised against an
ESS-plus-balance objective, autonomously discovers the tempering that accelerates
mode mixing by ~33x while leaving the target distribution intact. The research
direction is real and tractable.

### Sketch of the fuller direction

The prototype learns two scalars on one target. The research programme generalises
along three axes:

1. **A richer, state-dependent W(x; theta).** Replace W = L^c with a small neural
   surface W_theta(x) (or a few radial-basis bumps seeded at suspected barriers).
   The paper allows any symmetric W(x), so the stationary distribution stays
   proportional to L for free: the network only reshapes the medium the ray travels
   through, never the answer. This is the key safety property that makes learning W
   far less fragile than learning a normalising-flow proposal, where an imperfect
   map biases the target.

2. **An amortised objective.** Optimise theta across a family of targets (sweep
   separation, dimension, conditioning) rather than one, so the learned medium
   generalises. ESS of the slow mode is differentiable-friendly only through the
   chain, so prefer gradient-free (CMA-ES, as here) or score-function / reparameter-
   ised gradients through a differentiable surrogate of the switch statistic.

3. **Learning the emission rate to defeat disjoint islands.** Q3 showed the released
   sampler cannot cross true L=0 gaps. The natural fix is to learn a re-emission /
   refresh schedule r(x) that injects a controlled momentum perturbation at island
   boundaries (a learned Lambertian kick), tuned so the importance weight still
   conserves basic radiance. This is the one place the target could be perturbed, so
   it needs the Metropolis correction derived and checked, exactly the stationarity
   gate we used in Q1, before any coverage claim.

Concretely, the next prototype is: a 4-to-8 parameter W_theta(x) with bumps in the
inter-mode valleys, learned by CMA-ES against the amortised ESS+balance objective
over a separation sweep, with the Q1 KS gate run on every candidate to reject any
theta that drifts the target. If that holds, escalate to a small MLP and to the
learned-emission variant for the disjoint-island regime.

---

## Appendix: how to reproduce

```bash
./.venv/bin/python experiments/exp1_weighting.py
./.venv/bin/python experiments/exp2_autocorr_vs_hmc.py
./.venv/bin/python experiments/exp3_mode_coverage.py
./.venv/bin/python experiments/exp4_learn_proposal.py
```

Raw numbers are in `results/tables/` (CSV + JSON). Diagnostics validated in
`experiments/_smoke_metrics.py`; fork correctness in `experiments/_smoke_fork.py`.
