# SUDS showcase poster (Aug 14, 2026)

`suds_poster_v4.tex` is the poster source; `suds_poster_v4.pdf` is the current
compiled board (4 ft x 3 ft, beamerposter at scale 1.32). Edit the tex, then:

```
pdflatex suds_poster_v4.tex
```

Any TeX Live with beamerposter works (TinyTeX is enough). All figure paths are
relative: the `\figslot` macro looks in `artifacts/figures/` first, then
`figures/` (which holds only the three logos). Compile from this directory.

## Conventions (please keep)

- No em dashes anywhere, in prose or labels. Use commas, colons, parentheses.
- Palette and meanings are fixed: ray tracing gold `#D99A1B`, HMC-family red
  `#D1495B`, reference blue `#1F6FB4`, point/MAP gray `#8A8D99`. Red is for
  the HMC family only.
- Text must survive 4 ft x 3 ft printing: anything inside a figure needs to
  land at 24 pt or larger after the slot magnification.
- Every number and curve comes from recorded experiment data in this repo or
  the campaign packs. Do not retouch numbers in the tex without regenerating
  the figure behind them.

## Regenerating figures

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
