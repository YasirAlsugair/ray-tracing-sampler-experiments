# Ray Tracing Sampler: MNIST posterior experiments

Experiments on the public Ray Tracing Sampler (Behroozi 2025, arXiv:2510.25824)
applied to small MNIST networks, all torch, fp32, reproducible on a laptop. The
sampler is the author's released `raytrace_torch.py`, vendored pristine under
`vendor/ray-tracing-sampler/` with its licenses.

**Start here: [`exp6_mnist_posterior.ipynb`](exp6_mnist_posterior.ipynb)** (executed,
figures included; each results section says how to read its numbers).

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
  stationarity rule).

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

Datasets, checkpoints, and the large chain and sweep snapshot files
(`exp6_rt_chain_mlp*.npz`, `exp6_mb*_dt*.npz`, 0.1 to 3.8 GB each) are not committed;
the notebook's executed outputs carry the numbers, and the scripts reproduce the files.
