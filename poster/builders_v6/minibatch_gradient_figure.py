#!/usr/bin/env python3
"""Render the frozen UCI minibatch-gradient diagnostic used by poster v6.

The saved polar coordinates retain the exact 81-dimensional gradient-length
ratios and angles; they are not a two-coordinate projection.  The plot-ready
input is regenerated, when needed, by ``prepare_minibatch_gradient_data.py``.
Normal poster builds require only NumPy and Matplotlib.
"""
from __future__ import annotations

from pathlib import Path
import sys

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from style import INK, MUTED, save, use_poster_style


NOISE_CORAL = "#E07A5F"
MEAN_CORAL = "#C9553D"
REFERENCE_NAVY = "#19324D"
ANNOTATION_SIZE = 18.5
CAPTION_SIZE = 18.5


def _load(path):
    data = np.load(path, allow_pickle=False)
    required = {
        "batches", "mean_angle", "q10_angle", "q90_angle",
        "left_arrow_x", "left_arrow_y", "drawn_batch",
        "drawn_mean_angle", "n_train", "dimension",
    }
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"missing minibatch-gradient arrays: {sorted(missing)}")

    batches = np.asarray(data["batches"], dtype=int)
    mean = np.asarray(data["mean_angle"], dtype=float)
    low = np.asarray(data["q10_angle"], dtype=float)
    high = np.asarray(data["q90_angle"], dtype=float)
    x = np.asarray(data["left_arrow_x"], dtype=float)
    y = np.asarray(data["left_arrow_y"], dtype=float)
    if not (batches.shape == mean.shape == low.shape == high.shape):
        raise ValueError("batch and angular-summary arrays must have equal shapes")
    if x.shape != y.shape:
        raise ValueError("left-panel arrow arrays must have equal shapes")
    if int(data["dimension"]) != 81 or int(data["n_train"]) != int(batches[-1]):
        raise ValueError("unexpected target metadata in minibatch-gradient input")
    return data, batches, mean, low, high, x, y


def figure_minibatch_gradient(out_dir, data_path):
    data, batches, mean, low, high, x, y = _load(data_path)
    drawn_batch = int(data["drawn_batch"])
    drawn_mean = float(data["drawn_mean_angle"])
    n_train = int(data["n_train"])

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.2, 4.05), gridspec_kw={"width_ratios": [1.0, 1.5]})

    for axis in (left, right):
        for spine in axis.spines.values():
            spine.set_linewidth(0.9)
        axis.tick_params(axis="both", which="major", direction="out",
                         length=4.0, width=0.9)

    reach = float(np.quantile(np.hypot(x, y), 0.86)) * 1.03
    ring = np.linspace(0, 2 * np.pi, 240)
    left.plot(np.cos(ring), np.sin(ring), ls=(0, (5, 4)),
              color=MUTED, linewidth=1.6)
    left.annotate("exact gradient", xy=(0.44, 0.515), xytext=(0.02, 0.78),
                  xycoords="axes fraction", textcoords="axes fraction",
                  fontsize=ANNOTATION_SIZE, color=INK, va="bottom",
                  arrowprops=dict(arrowstyle="-", color=INK, linewidth=1.3,
                                  shrinkA=3, shrinkB=3))
    for xi, yi in zip(x, y):
        left.annotate("", xy=(xi, yi), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color=NOISE_CORAL,
                                      alpha=0.24, linewidth=1.55,
                                      shrinkA=0, shrinkB=0))
    left.annotate("", xy=(1.0, 0.0), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color=REFERENCE_NAVY,
                                  linewidth=3.6, shrinkA=0, shrinkB=0,
                                  mutation_scale=26))
    left.plot(0, 0, "o", color=REFERENCE_NAVY, markersize=7, zorder=6)
    left.set_xlim(-reach, reach)
    left.set_ylim(-reach, reach)
    left.set_aspect("equal")
    left.set_xticks([])
    left.set_yticks([])
    for spine in ("left", "bottom"):
        left.spines[spine].set_visible(False)
    left.set_xlabel(f"$B={drawn_batch}$; mean error {drawn_mean:.0f}$^\\circ$",
                    fontsize=CAPTION_SIZE, labelpad=12)

    right.axhline(90.0, color=INK, ls=(0, (6, 4)), linewidth=2.0, zorder=2)
    right.text(batches[-1], 92.0, "uninformative direction",
               fontsize=ANNOTATION_SIZE, color=INK, va="bottom", ha="right")
    right.fill_between(batches, low, high, color=NOISE_CORAL,
                       alpha=0.14, zorder=3)
    right.plot(batches, mean, "-o", color=MEAN_CORAL,
               linewidth=3.0, markersize=9, zorder=4)
    marked_index = np.flatnonzero(batches == drawn_batch)
    if len(marked_index) != 1:
        raise ValueError("drawn batch must occur exactly once")
    marked = mean[marked_index[0]]
    right.plot([drawn_batch], [marked], "o", color=REFERENCE_NAVY,
               markersize=12, zorder=5)
    right.annotate("shown at left", xy=(drawn_batch, marked), xytext=(-10, -8),
                   textcoords="offset points", ha="right", va="top",
                   fontsize=ANNOTATION_SIZE, color=INK)

    right.set_xscale("log", base=2)
    right.set_xticks([16, 64, 256, 1024, 4096, n_train])
    right.set_xticklabels(["16", "64", "256", "1024", "4096", "all $N$"])
    right.minorticks_off()
    right.set_ylim(0, 126)
    right.set_yticks([0, 30, 60, 90, 120])
    right.set_yticklabels(["0$^\\circ$", "30$^\\circ$", "60$^\\circ$",
                           "90$^\\circ$", "120$^\\circ$"])
    right.set_xlabel("batch size $B$")
    right.set_ylabel("angle to exact gradient", labelpad=10)
    right.grid(False)

    figure.subplots_adjust(left=0.005, right=0.995, top=0.98, bottom=0.18,
                           wspace=0.36)
    left_box = left.get_position(original=True)
    left.set_position([left_box.x0, 0.12, left_box.width, 0.86], which="both")
    save(figure, out_dir, "minibatch_gradient")


if __name__ == "__main__":
    import sources

    use_poster_style()
    figure_minibatch_gradient(sources.OUT_DIR,
                              sources.MINIBATCH_GRADIENT_DATA)
