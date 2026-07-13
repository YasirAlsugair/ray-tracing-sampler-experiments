# Earlier studies, kept for reference

Two older sampler-comparison studies that predate the MNIST posterior notebook
in the repo root. They are archived here as-is.

1. **Analytic (artificial-data) study** (`experiments/exp1_...exp4_*.py`,
   sampler and target code in `rts/`, writeup `results/RESULTS.md`): ray
   tracing vs HMC-family samplers on closed-form 2D targets, including
   autocorrelation and mode-coverage comparisons.

2. **ResNet-50 on MNIST, five samplers** (`experiments/exp5*.py`, writeup
   `results/EXP5_RESNET50.md`): ray tracing, the vendor's HMC, SGLD, SGHMC,
   and MCLMC on one ResNet-50/MNIST posterior, same minibatch stream and the
   same tuning rule for every sampler (largest step surviving a short pilot).
   The five chain traces are in `results/tables/exp5_trace_*.npz`.

These scripts expect the repo-root layout (the same venv, `vendor/`, and
MNIST download); to rerun one, copy its files over the corresponding paths in
the root. The harness only needs a log density, so porting the samplers to
another framework is straightforward.
