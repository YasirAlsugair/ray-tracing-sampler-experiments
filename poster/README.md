# SUDS showcase poster (Aug 14, 2026)

`suds_poster_v4.tex` and `.pdf` are the preserved collaborative baseline.
`suds_poster_v6.tex` and `.pdf` are a separate candidate revision for review;
v6 does not replace v4.

Both are 4 ft x 3 ft boards built with beamerposter at scale 1.32. Compile from
this directory:

```
pdflatex suds_poster_v4.tex
pdflatex suds_poster_v6.tex
```

Any TeX Live with beamerposter works (TinyTeX is enough). All figure paths are
relative. V4 reads `artifacts/figures/`; v6 first checks
`artifacts/figures_v6/` and then falls back to the v4 assets and `figures/`.

## Conventions (please keep)

- No em dashes anywhere, in prose or labels. Use commas, colons, parentheses.
- The v4 palette remains fixed: ray tracing gold `#D99A1B`, HMC-family red
  `#D1495B`, reference blue `#1F6FB4`, point/MAP gray `#8A8D99`.
- V6 uses figure-local palettes when that improves the individual panel. Keep
  meanings consistent within each figure, and do not introduce a red-green
  comparison.
- Text must survive 4 ft x 3 ft printing: anything inside a figure needs to
  land at 24 pt or larger after the slot magnification.
- Every number and curve comes from recorded experiment data in this repo or
  the campaign packs. Do not retouch numbers in the tex without regenerating
  the figure behind them.

## Regenerating v6 figures

The committed PDFs in `artifacts/figures_v6/` are the candidate board's direct
inputs. They can all be rebuilt from the frozen inputs in
`artifacts/data_v6/`:

```
python builders_v6/build_all.py
```

See `builders_v6/README.md` for the input manifest, software requirements, and
the optional provenance steps that recreate the minibatch diagnostic and train
the 50-member Gaia ensemble.

## Regenerating v4 figures

Committed PDFs in `artifacts/figures/` are the board's inputs; you do not need
to regenerate anything to edit the poster.

Run from this directory, in the repo environment (torch, numpy, scipy,
matplotlib):

- `build_exp7_figures.py` rebuilds the exp7 Gaia panels from the committed
  packs in `../results/tables/`.
- `exp7_gaia_panels_suds.py` rebuilds the three column-3 Gaia panels
  (structure, members, imposters) into `artifacts/exp7_suds_figs/`; copy the
  ones you want into `artifacts/figures/`.

`build_figures.py`, `shell_measured_suds.py`, and `fig_typical_set.py` are
kept for provenance but read data outside this repo (Yasir's local tree);
ask Yasir to rerun those.
