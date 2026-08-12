# Poster v7 candidate: figure overrides

V7 is an additive revision. It falls back to the committed v6 and earlier
assets, and keeps only its changed figures in `artifacts/figures_v7/`.

From `poster/`, regenerate the two code-built v7 figures and compile the board:

```bash
python builders_v7/build_all.py
pdflatex -interaction=nonstopmode -halt-on-error suds_poster_v7.tex
```

The builders require NumPy and Matplotlib. The measured-results strip reads the
same frozen summaries already committed under `artifacts/data_v6/`:

- `dimension_fits.json`
- `matched_condition_summary.json`
- `superconductor_frontier.csv`

The three Gaia PDFs are retained as locked visual inputs:

- `gaia_calibration_by_group.pdf` is the updated calibration panel.
- `exp7_cloud_poster.pdf` and `exp7_marginal_poster.pdf` are the current
  predictive panels produced by `poster/exp7_predictive_poster.py`.

Their PDFs are committed so poster compilation does not depend on rerunning the
Gaia analysis or changing the selected stars.
