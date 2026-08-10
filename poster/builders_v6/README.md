# Poster v6 candidate: figures and provenance

These builders write the seven figure PDFs consumed by
`poster/suds_poster_v6.tex` to `poster/artifacts/figures_v6/`.  A normal build
reads only files committed inside this repository; AutoRT is not required.

From `poster/`, regenerate all figures and compile the board with:

```bash
python builders_v6/build_all.py
pdflatex -interaction=nonstopmode -halt-on-error suds_poster_v6.tex
```

Normal Python requirements are NumPy and Matplotlib.  The Gaia style-preserving
pass also requires Poppler's `pdftops` and Ghostscript's `ps2pdf` on `PATH`.

## Frozen inputs

`artifacts/data_v6/` contains compact inputs rather than raw training data or
full chains:

| File | Used by | Provenance |
|---|---|---|
| `harder_bnn_poster.npz` | `data_bands_figure.py` | AutoRT study 25, trust-gated NUTS reference |
| `minibatch_gradient_data.npz` | `minibatch_gradient_figure.py` | AutoRT UCI Superconductivity BLR diagnostic, seed 20260814 |
| `dimension_fits.json` | tolerance panel | AutoRT study 15 |
| `matched_condition_summary.json` | exponent-erosion panel | AutoRT study 22 |
| `superconductor_frontier.csv` | compute panel | AutoRT study 11 |
| `exp7_ensemble_50.npz` | both Gaia panels | preserved seeds 0--9 plus newly trained seeds 10--49 |
| `gaia_calibration_by_group_v4.pdf` | Gaia calibration panel | locked vector source before v6 styling |
| `gaia_nll_vs_members_v4.pdf` | Gaia NLL panel | locked vector source before v6 styling |

The Gaia builder starts from the locked v4 vector PDFs retained in
`artifacts/data_v6/`, changes their local method palette and reference-rule
weight, removes `(tuned)` from SGHMC, and replaces the deep-ensemble results
with the committed 50-member predictions.  It preserves the other plotted
values and vector geometry.

## Expensive provenance steps

These are not part of a normal poster build.

To regenerate `minibatch_gradient_data.npz`, use the AutoRT environment and raw
UCI archive:

```bash
python builders_v6/prepare_minibatch_gradient_data.py \
  --autort-root /path/to/AutoRT
```

To retrain Gaia ensemble members 10--49, first place the upstream, local-only
`exp7_gaia_pristine.npz` in `results/tables/`, then run:

```bash
python builders_v6/train_gaia_ensemble.py
```

The training script uses the committed seed-0--9 predictions and weights in
`results/tables/`.  It requires PyTorch and is deliberately separate from the
figure build.
