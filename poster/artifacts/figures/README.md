# Figure provenance (poster_v2)

Built by `../../build_figures.py` (run with `empirical/.venv/bin/python`):

- `typical_set.pdf`: analytic chi-squared dimension sweep (redesigned 2026-08-05).
- `shell_diagnostic.pdf`: real chains from the repo's stochastic-gradient harness
  (`explainer/sg_experiment.py` loop bodies + `explainer/physics.snell_kick`),
  4 chains x 60,000 steps per sampler, D = 100 standard Gaussian, sigma = 20,
  HMC dt = 0.03 with full refresh every M = 30 (the `s1_radius.npz` configuration),
  RT ds = 0.3 with partial refresh f = 0.03. lnL traces asserted bit-identical to
  the untouched `run_hmc_sto` / `run_rts_sto`. No Metropolis test, matching the
  paper's Eq. 34 stochastic-gradient experiment. Chain data + seeds + split R-hat:
  `../shell_chains.npz` (see its `provenance` key).

Extracted, not built here (this tree was reconstructed on 2026-08-05 from the
cloud-session `suds_poster.pdf`, whose `build_figures.py` and `artifacts/` never
reached this machine):

- `fig0_posterior_draws_harder.pdf`, `fig3_scaling_demonstration.pdf`: vector
  page-crops of the cloud-session poster PDF (content untouched).
- `hmc_vs_rt_noise.png`: embedded image extracted at native resolution
  (3018 x 1885) with its alpha mask.
- `../../figures/dept_logo_trim.png`, `dsi_logo_standalone.png`,
  `kaust_academy_logo_trim.png`: same extraction.

If the cloud session's originals become available, drop them in over the
extracted copies.
