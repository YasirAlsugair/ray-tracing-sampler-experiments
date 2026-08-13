# Poster v9 figure provenance

V9 preserves the finalized AutoRT poster content and the current Gaia block.
Its figures are pinned in `artifacts/figures_v9/`, so compiling v9 does not
alter v8 or depend on fallback resolution.

From `poster/` in the AutoRT checkout, rebuild the AutoRT-generated figures
and compile the poster:

```bash
../.venv/bin/python builders_v9/build_all.py
pdflatex -interaction=nonstopmode -halt-on-error suds_poster_v9.tex
```

In `ray-tracing-sampler-experiments`, compile directly from the committed
`figures_v9` assets. The non-Gaia builders are retained there for provenance
but require the AutoRT study inputs mirrored by `sources.py`.

The three Gaia PDFs are the current locked outputs from
`ray-tracing-sampler-experiments`:

- `gaia_calibration_by_group.pdf`
- `exp7_cloud_poster.pdf`
- `exp7_marginal_poster.pdf`

Their generators are retained as `gaia_calibration_figure.py` and
`exp7_predictive_poster.py`; rebuilding them requires the fitted Gaia analysis
packs in Yasir's repository. The PDFs remain committed so normal poster builds
do not depend on rerunning that analysis.
