# Exp 5: ResNet-50 on MNIST, train then sample (proof of concept)

**Date:** 2026-07-06/07. **Ask (Josh):** train a ResNet-50 on MNIST, then sample from it.
**Verdict: the pipeline works end to end.** The ray tracing sampler runs as a drop-in
torch optimizer on a 23.5M-parameter network on a MacBook (MPS, float32), explores the
weight posterior 53x further than HMC at identically tuned settings, and its posterior
ensemble matches the Adam point estimate's accuracy. The comparison also includes the
modern stochastic-gradient samplers (SGLD, SGHMC, MCLMC), with one instructive failure
(MCLMC) that the validation mask caught automatically.

## Setup

- **Model:** stock torchvision `resnet50(num_classes=10)`, 1-channel MNIST repeated to
  3 channels. D = 23,528,522 parameters. Trained 6 epochs of Adam (~3 min on the M5 Pro):
  test accuracy 98.93%, eval-mode train CE 0.00138.
- **Posterior:** ln L = -SCALE x (mean CE per image), flat prior, no augmentation,
  BatchNorm buffers frozen (net always in eval mode, so the posterior is well-defined).
  SCALE = N_train/(2 x 0.1) = 3e5, the paper's ResNet-34 regime (D_eff ~ N_train), so the
  posterior should sit ~0.1 nats/image above the trained point.
- **Recipe (paper App. E / Sec. 3.4):** the vendor `Raytracer` optimizer as sampler, one
  fresh 256-image minibatch per step, NO Metropolis test (unadjusted, as the paper does
  once gradient noise dominates), GPT-2-style post-hoc masking of samples whose validation
  CE exceeds seed + 0.1 + 2 sigma (threshold 0.174). 3000 steps per chain, burn-in 1000,
  ensemble members every 250 steps.
- **Tuning rule (all five samplers, identical):** pilot sweep over a 5-point step-size
  grid, keep the largest step that survives 150 steps without the monitor loss rising
  1 nat. This is the paper's own hand-tuning protocol, and it is the surface the
  auto-tuning project aims to replace.
- Same minibatch stream (same generator seed) for every chain.

## Results (3000 steps each, identical pilot-tuned settings)

| sampler | step | it/s | |dx| from seed | post val CE | tau_probe (steps) | ensemble acc | ens NLL | ens ECE |
|---|---|---|---|---|---|---|---|---|
| Adam point estimate | | | 0 | 0.055 | | 98.93% | 0.0342 | 0.0048 |
| Ray tracing | 1e-5 | 3.5 | **141.3 (53%)** | 0.083 | 59 | 98.94% (9/9) | 0.0345 | 0.0083 |
| HMC (vendor) | 1e-5 | 3.2 | 2.7 (1%) | 0.056 | 10 (floor, artifact) | 98.93% (9/9) | 0.0344 | 0.0050 |
| SGLD | 1e-7 | 3.5 | 83.7 (31%) | 0.060 | 27 | **99.05%** (9/9) | **0.0277** | **0.0025** |
| SGHMC (alpha=0.1) | 1e-8 | 3.2 | 118.2 (44%) | 0.070 | 19 | 98.98% (9/9) | 0.0274 | 0.0052 |
| MCLMC | 0.3 | 2.5 | 824.2 (308%) | 0.307 | 18 | **masked out (0/9)** | | |

(|dx| relative to the seed parameter norm |x0| = 267. tau from Sokal autocorrelation of
2,560 probe class probabilities, post-burn, lower bounds at this chain length. Full
traces in `tables/exp5_trace_*.npz`, figures `exp5_all_loss.png`, `exp5_all_kinetic.png`,
`exp5_all_tau.png`, `exp5_pilot.png`, per-run summaries `exp5_summary.json`,
`exp5b_summary.json`, merged `exp5_all_summary.csv`.)

## Findings

1. **RTS works at deep-learning scale on a laptop.** Drop-in `torch.optim` usage, 3.5
   sampler steps/s at batch 256 on MPS. Constant speed held exactly: KE/D = 0.500 every
   step of the run (this is structural, the kick is a pure rotation).

2. **Exploration is the headline contrast.** At the same pilot-chosen dt = 1e-5, ray
   tracing moved 141 units (53% of the seed norm, near its ballistic limit of 145) and
   settled near the intended 0.1 nats band, while vendor HMC moved 2.7 units (1%): the
   gradient kick acts as a restoring force that pins it near the minimum, so in 3000
   steps it effectively never left the point estimate. Echoes the paper's MLP exploration
   result (their RT explored |dx|/|x| ~ 3.6 vs Adam 0.3).

3. **The tau column would lie without the |dx| column.** HMC's tau = 10 steps is the
   estimator floor: a chain that has not moved produces probe traces that are pure
   minibatch-evaluation noise, which decorrelates instantly. Mixing numbers are only
   meaningful next to exploration distance.

4. **No stochastic-gradient heating was visible at these settings.** HMC's KE/D stayed
   at 0.500 (0 vendor momentum resets). At dt = 1e-5, batch 256, the noise-heating
   channel is negligible over 3000 steps; the differentiator here is exploration speed,
   not stability. (The sigma_c robustness story lives at larger dt x SCALE or smaller
   batches: a natural follow-up sweep, where the paper predicts RT's advantage grows as
   sigma_c ~ 1/dt vs HMC's 1/sqrt(dt).) SGHMC showed the friction-noise balance exactly:
   KE/D settled at 0.527, slightly above equilibrium, noise in, friction out.

5. **Posterior ensembles match or beat the point estimate.** RT members individually
   drop to 97.5-98.6% (they sit at posterior temperature, not at the minimum) and their
   9-member ensemble recovers 98.94%: the classic Bayesian signature, and evidence the
   samples are genuinely spread rather than degraded. SGLD, which stayed closer to the
   minimum, gave the best ensemble (99.05%, NLL 0.0277 vs point 0.0342, ECE halved).
   At this horizon the samplers trade exploration against per-member fidelity; RT sits
   at the exploration end of that frontier at fixed compute.

6. **MCLMC is the instructive failure.** Its pilot passed at eps = 0.3 (150-step survival)
   but the 3000-step chain never became stationary: CE climbed 0.19 -> 0.24 -> 0.71,
   moving 824 units (3.1x the seed norm, 92% of its ballistic limit). Constant speed kept
   its KE pinned (no blowup), but with the bend-per-step and refresh both tiny it was
   near-free flight out of the basin. Per unit arc length its dynamics bend exactly as
   much as RT's (same underlying geometry); it simply traveled 6x the arc in the same
   step budget with 6x sparser gradient sampling along the path, under-resolving the
   landscape. The GPT-2-style validation mask rejected 9/9 of its ensemble members:
   the safety net worked as designed.

## What this says about auto-tuning (the actual research direction)

Every number above required hand-set knobs: step size (pilot sweep), SCALE (a guess at
D_eff), refresh rate, run length, burn-in cut, mask threshold. Three concrete lessons:

- **Survival is not stationarity.** The 150-step pilot rule accepted MCLMC's eps = 0.3,
  which fails catastrophically at 3000 steps. An auto-tuner needs a stationarity
  criterion (loss-trace drift detection), not a divergence criterion.
- **The same rule can under- and over-shoot.** SGLD/SGHMC were pinned by stability to
  steps so small they undershot the target temperature band; MCLMC overshot. RT happened
  to land in the band, but there is no guarantee its plateau is converged rather than
  slowly drifting (see caveats).
- **Fixed-acceptance-style targets do not exist here** (no Metropolis), so the natural
  auto-tune observables are the loss-band error (actual vs intended Delta-loss), the
  KE trace (for heating-prone samplers), and function-space autocorrelation per gradient.

## Caveats (honesty spine)

- All chains are **unadjusted** (no Metropolis): approximate MCMC with fixed
  hyperparameters, exactly as the paper frames its NN runs, not exact sampling.
- **3000 steps is a short horizon** (the paper's ResNet ran 5e5 epochs, GPT-2 burn-in
  ~3M steps). RT's plateau near the target band may itself still be drifting on longer
  horizons: the MCLMC result proves such drift can hide. A 3-10x longer RT run is the
  cheapest next check.
- Single seed per sampler, single chain each; MNIST is easy and accuracy headroom is
  ~zero, so ensemble NLL/ECE differences are suggestive, not conclusive.
- tau values are lower bounds (200 post-burn probe evaluations).
- fp32 on MPS throughout (the paper ran GPT-2 in BF16, so this exceeds their precision).

## Repro

```
cd /Users/yasiralsugair/UofT/empirical
./.venv/bin/python experiments/exp5_resnet_mnist_train.py     # ~3 min (MPS)
./.venv/bin/python experiments/exp5_resnet_mnist_sample.py    # ~45 min: RT + vendor HMC
./.venv/bin/python experiments/exp5b_newer_hmc.py             # ~70 min: SGLD, SGHMC, MCLMC
```
