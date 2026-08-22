# Toward Scalable Bayesian Uncertainty for Machine Learning with the Ray Tracing Sampler

KAUST Academy Summer School (KASP) 2026, hosted at the University of Toronto Data
Sciences Institute (SUDS).
Student: Yasir Alsugair. Mentor: Prof. Joshua Speagle. Co-supervisor: Prof. Ricardo Baptista.

Experiments on the public Ray Tracing Sampler (Behroozi 2025, arXiv:2510.25824), a
gradient-based MCMC method that carries a unit direction on a sphere instead of an
unbounded momentum, so a noisy minibatch gradient can steer the chain but cannot heat
it. The sampler is the author's released `raytrace_torch.py`, vendored pristine under
`vendor/ray-tracing-sampler/` with its licenses. Everything here is torch, fp32.

## Submission map

| Deliverable | Path |
|---|---|
| Final report (14 pp) | `docs/Report_Final.pdf` |
| Poster (A0 landscape) | `poster/kasp_poster_v1.pdf` |
| Weekly reports 1-3 | `docs/Report_1.pdf`, `docs/Report_2.pdf`, `docs/Report_3.pdf` |
| MNIST ground truth (exp6) | `exp6_mnist_posterior.ipynb` |
| Gaia campaign (exp7) | `exp7_gaia_posterior.ipynb`, `exp7_reference_run.md` |

## The two experiment tracks

**exp6, ground truth on MNIST.** Small networks (MLP D = 50,890, CNN D = 12,810) where
full-batch exact chains are affordable, so every later minibatch shortcut has something
to be checked against. Start at
[`exp6_mnist_posterior.ipynb`](exp6_mnist_posterior.ipynb) (executed, figures included).

**exp7, the Gaia campaign.** Gaia XP coefficients (55 BP + 55 RP) to [alpha/M] for
126,156 giant stars with APOGEE labels, the target behind Laroche & Speagle 2025
(ApJ 979, 5). An 110-64-64-1 tanh MLP, D = 11,394, sampled with minibatch ray tracing
and the Eq. 33 noise-softened gate, then certified with an exact full-batch finisher.
Start at [`exp7_gaia_posterior.ipynb`](exp7_gaia_posterior.ipynb); the full campaign
log, including every convergence decision, is in
[`exp7_reference_run.md`](exp7_reference_run.md).

## Data

- **MNIST** downloads automatically through torchvision on first run.
- **Gaia XP / APOGEE**: Zenodo record 14041773 (Laroche & Speagle 2025),
  <https://zenodo.org/records/14041773>. Public, no NDA. The pristine giant cut used
  here (126,156 stars, split seed 2003, 80/20) is reproduced by the notebook from that
  download; the derived arrays are not committed because of size.

## exp6 in detail

The study: an MLP (51k params) and a small CNN (13k) with the loss-to-posterior
relation derived exactly (ln L = -N * CE, prior N(0,1) or flat), sampled full batch
with the Metropolis test on, against deep ensemble and MC dropout baselines. It covers:

- the acceptance cliff and how acceptance depends on where the chain is,
- the long transient toward the prior shell at ||theta||^2 = D, with the fit and
  weight-norm terms tracked separately along the chain,
- a flat vs N(0,1) prior comparison (the flat chain looks converged on the target
  trace while the weight norm grows without bound),
- a D=100 Gaussian known-answer check through the same pipeline,
- a minibatch study on identical settings comparing the paper's noise-softened accept
  test (Eq. 33), the plain test on noisy estimates (Eq. 19), and unadjusted dynamics,
  including the paper's step-size tuning recipe run as written and the softened test
  extended to formal convergence (the only chain in the study to pass the
  stationarity rule),
- an uncertainty-quality comparison (error-retention curves, reliability, member
  scaling, flag overlap) and a fake-image test (noise, Fashion-MNIST, inverted
  digits) measuring who stays confident on garbage and whose uncertainty flags it.

## Layout

```
exp6_mnist_posterior.ipynb      the notebook (start here)
colab_runner.ipynb              runs the scripts on a Colab GPU
experiments/
  exp6_simple_mnist_train.py    MLP + CNN point estimates
  exp6_ensemble.py              10-seed deep ensembles
  exp6_sample_metropolis.py     exact full-batch chains, Metropolis on
  exp6_minibatch.py             minibatch sweep: unadjusted / eq33 / eq19 / tune33
  exp6_figures.py               the three result figures
results/EXP6_DERIVATION.md      the loss-to-posterior derivation
results/figures/, results/tables/   figures, logs, and small artifacts
rts/metrics.py                  Sokal autocorrelation time, ESS
plots/style.py                  shared palette
vendor/                         upstream sampler, unmodified
legacy/                         earlier studies: the analytic comparison and the
                                ResNet-50/MNIST five-sampler run (SGLD, SGHMC,
                                MCLMC included), with their chain traces
```

## Running

```
python3.12 -m venv .venv
./.venv/bin/pip install torch torchvision numpy scipy matplotlib
./.venv/bin/python experiments/exp6_simple_mnist_train.py
./.venv/bin/python experiments/exp6_ensemble.py
./.venv/bin/python experiments/exp6_sample_metropolis.py run
./.venv/bin/python experiments/exp6_minibatch.py            # unadjusted sweep
./.venv/bin/python experiments/exp6_minibatch.py eq33       # noise-softened gate
./.venv/bin/python experiments/exp6_minibatch.py eq19       # plain gate
./.venv/bin/python experiments/exp6_minibatch.py tune33     # step-size recipe
```

The notebook loads saved artifacts from `results/tables/` when present and recomputes
anything missing (set `RECOMPUTE = True` to force). MNIST downloads automatically.

To continue a chain rather than restart it: the `results/tables/exp6_last_state_*.npz`
files hold each big chain's final parameter vector plus its settings (the full sample
files exceed GitHub's size limit). Load `theta` and pass it to `run(..., theta0=...)`
in `exp6_sample_metropolis.py` (exact chains) or `run_arm(..., initial_state=...)` in
`exp6_minibatch.py` (minibatch chains; reuse the stored `sigma_sto` for the same
Eq. 33 gate). Momenta are redrawn every trajectory, so this continues the same chain.
The CNN Eq. 33 chain needs no state file: its legs are all committed, and
`exp6_minibatch.py` resumes it from the last leg automatically.

Datasets, checkpoints, and the large chain and sweep snapshot files
(`exp6_rt_chain_mlp*.npz`, the 20,000-trajectory `exp6_rt_chain_cnn.npz`,
`exp6_mb*_dt*.npz`, 0.1 to 3.8 GB each) are not committed; the notebook's executed
outputs carry the numbers, and the scripts reproduce the files. The earlier
600-trajectory CNN chain is still in git history, and `exp6_last_state_cnn.npz`
continues the 20,000-trajectory rerun (produced on a rented GPU via the script's
`runlegs` mode).

## exp7: running the Gaia campaign

```
./.venv/bin/pip install torch numpy scipy matplotlib astropy
./.venv/bin/python experiments/exp7_predictive_figures.py     # predictive figures
```

The campaign itself was run on rented GPUs over several weeks (roughly $85 to $90 of
cloud time) and is not a single-command reproduction. `exp7_reference_run.md` records
every arm, its settings, its drift series, and why it was kept or dropped, so any single
rung can be re-run from the settings printed there. The noise-model ladder is the thing
to read first: held-out z standard deviation falls 6.88 -> 6.23 -> 0.95 -> 1.08 as the
likelihood gains a scatter term, a fitted scale, and a heteroscedastic sigma(x). The
chain of record is the rung-2 heteroscedastic chain; the rung-3 Student-t chain
(nu = 5.48, certified at 8.5M steps) has the best held-out NLL of any method tried.

## Poster

`poster/kasp_poster_v1.tex` builds the A0 landscape KASP poster:

```
cd poster && pdflatex kasp_poster_v1.tex
```

Needs a LaTeX installation with `beamerposter` (TinyTeX is enough) and the logo files in
`poster/figures/`. `suds_poster_v9.tex` is the earlier SUDS version this was adapted
from, joint work with Chuxuan Ai.

## Honest scope

Validated here: robustness to stochastic-gradient noise, with measured step-size laws
(sigma_c proportional to h^-1 for ray tracing, h^-1/2 for HMC), a companion theoretical
derivation by the mentor that matches the published tolerances within 30%, and a working
application on a real survey at D = 11,394.

Partial: the cost story. At matched accuracy the gate-passing saving was about 1.5x at
batch 256 on one problem. Larger speedup figures from these runs come from
configurations that fail their own accuracy gate and are not quoted.

Open: width honesty (4-5% PIT under-coverage), anisotropic noise in practice (whitening
untested), single chains only (no multi-chain R-hat), scale beyond D = 11.4k, and mode
coverage on disconnected targets (not exercised here, no claim made).
