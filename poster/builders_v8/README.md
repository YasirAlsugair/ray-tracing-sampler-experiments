# Poster v8: source and figure provenance

V8 is a non-destructive merge: columns 1--2 come from AutoRT's current v7,
the Gaia block comes from the current poster in this repository, and the
Conclusion is intentionally still provisional. Every PDF used by the poster is
pinned in `artifacts/figures_v8/`, so compiling v8 never changes v7.

From `poster/`, regenerate the portable code-built figures and compile:

```bash
../.venv/bin/python builders_v8/build_all.py
pdflatex -interaction=nonstopmode -halt-on-error suds_poster_v8.tex
```

The builders require NumPy, SciPy, and Matplotlib. They read only the frozen
plot-ready inputs already committed under `artifacts/data_v6/`:

- `dimension_fits.json`
- `harder_bnn_poster.npz`
- `matched_condition_summary.json`
- `minibatch_gradient_data.npz`
- `superconductor_frontier.csv`

The Gaia PDFs are retained as locked analysis outputs because rebuilding them
requires the fitted Gaia chains and catalog payloads under `results/tables/`:

- `gaia_calibration_by_group.pdf` is the updated calibration panel.
- `exp7_cloud_poster.pdf` and `exp7_marginal_poster.pdf` are the current
  simplified predictive panels.

The exact predictive-panel generator is preserved as
`builders_v8/exp7_predictive_poster.py`. The calibration styling/50-member
generator is preserved as `builders_v8/gaia_calibration_figure.py`. Their PDFs
are committed so a normal poster build does not depend on rerunning the Gaia
analysis or changing the selected stars.
