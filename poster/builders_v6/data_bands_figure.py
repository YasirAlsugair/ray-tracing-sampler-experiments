#!/usr/bin/env python3
"""v6 introduction: a warm posterior fan against a cool optimized fit."""
from __future__ import annotations

from pathlib import Path
import sys

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import sources
from style import lift, save

POSTERIOR_DRAW = "#D99A1D"
POSTERIOR_BAND = "#DAB96B"
FITTED_INDIGO = "#6557A5"
INK = "#242833"
OBSERVATION = "#111318"
GRAY = "#7B8190"
NO_DATA = "#DDE2E9"
GRID_BLUE = "#D9DEE7"

WIDTH = 7.0
HEIGHT = 4.8

RC = {
    "mathtext.fontset": "cm",
    "font.size": 16,
    "axes.labelsize": 17,
    "xtick.labelsize": 14.5,
    "ytick.labelsize": 14.5,
    "axes.edgecolor": GRAY,
    "axes.linewidth": 0.9,
    "axes.labelcolor": INK,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "none",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "none",
}


def _nodata_spans(grid, region):
    nodata = region != "train"
    edges = np.flatnonzero(np.diff(nodata.astype(int)))
    spans, open_at = [], (0 if nodata[0] else None)
    for edge in edges:
        if open_at is None:
            open_at = edge + 1
        else:
            spans.append((open_at, edge))
            open_at = None
    if open_at is not None:
        spans.append((open_at, len(grid) - 1))
    return spans


def figure_data_bands(out_dir):
    data = np.load(sources.HARDER_BNN_POSTER)
    grid = data["grid_raw"]
    mean, sd = data["posterior_mean"], data["posterior_sd"]

    with plt.rc_context(RC):
        figure, axis = plt.subplots(figsize=(WIDTH, HEIGHT))

        for i0, i1 in _nodata_spans(grid, data["region"]):
            axis.axvspan(grid[i0], grid[i1], color=NO_DATA, alpha=0.38, lw=0)
            axis.text(0.5 * (grid[i0] + grid[i1]), -4.18, "NO DATA",
                      ha="center", va="center", fontsize=14, color=GRAY)

        axis.fill_between(grid, mean - 2 * sd, mean + 2 * sd,
                          color=POSTERIOR_BAND, alpha=0.28, lw=0)
        for function in data["posterior_spaghetti"]:
            axis.plot(grid, function, color=POSTERIOR_DRAW, lw=0.65,
                      alpha=0.21, zorder=1)
        axis.plot(grid, data["optimizer_function"], color=FITTED_INDIGO, lw=3.0,
                  solid_capstyle="round", path_effects=lift(4.6), zorder=4)
        axis.plot(grid, data["true_f_raw"], color=INK, lw=2.1, ls=(0, (7, 4)),
                  dash_capstyle="round", zorder=5)
        axis.plot(data["x_raw"], data["y_raw"], "o", color=OBSERVATION,
                  markersize=6.0,
                  markeredgecolor="white", markeredgewidth=1.1, zorder=6)

        axis.set_xlim(-4, 4)
        axis.set_ylim(-4.5, 4.25)
        axis.set_xticks([-4, -2, 0, 2, 4])
        axis.set_yticks([-4, -2, 0, 2, 4])
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$f(x)$")
        axis.grid(True, axis="y", color=GRID_BLUE, lw=0.8, alpha=0.42)
        axis.set_axisbelow(True)

        handles = [
            Line2D([0], [0], color=POSTERIOR_DRAW, lw=2.0),
            Line2D([0], [0], color=FITTED_INDIGO, lw=3.0),
            Line2D([0], [0], color=INK, lw=2.1, ls="--", dash_capstyle="butt"),
            Line2D([0], [0], color=OBSERVATION, lw=0, marker="o", markersize=6.0,
                   markeredgecolor="white", markeredgewidth=1.1),
        ]
        axis.legend(handles, ["Posterior draws", "Adam-optimized network",
                              "Ground truth", "Observations"],
                    loc="upper right", bbox_to_anchor=(0.995, 0.995), fontsize=12,
                    frameon=False,
                    handlelength=1.8, labelspacing=0.32, handletextpad=0.6,
                    borderaxespad=0.0)
        figure.tight_layout()
        save(figure, out_dir, "data_bands")


if __name__ == "__main__":
    figure_data_bands(sources.OUT_DIR)
