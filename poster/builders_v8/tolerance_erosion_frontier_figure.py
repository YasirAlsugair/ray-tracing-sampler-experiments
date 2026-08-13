#!/usr/bin/env python3
"""The three measured results as one strip: noise tolerance against step, the
local exponent under real mini-batches, and cost per effective sample.

Replaces the fig8 + fig9 pair. The dropped panel is fig8's exponent-against-dimension
summary, whose numbers the poster's Conclusion already states in words; the raw
tolerance fits it summarized are kept here as panel (a).

This figure uses two saturated colour families. Method hue carries RT versus HMC;
lightness carries dimension in panel (a), and the remaining panels use the family
anchors at full strength.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from style import INK, MUTED, PANEL, lift, save, use_poster_style

RT_BLUE = "#168AAD"
HMC_MAGENTA = "#B24C83"
REAL_ARM = "#0B596B"
RT_TEXT = "#0A6A84"
FINE_PRINT = "#6F7480"
RT_DIM = {16: "#78B9CB", 81: "#4AA5BD", 256: "#288FAE", 1024: "#0A708F"}
HMC_DIM = {16: "#D59AB9", 81: "#C77EA6", 256: "#B96091", 1024: "#98406F"}

DIMS = (16, 81, 256, 1024)
LS = (16, 32, 64, 128)
TITLE_SIZE = 16.0
AXIS_LABEL_SIZE = 17.0
STANDALONE_MATH_LABEL_SIZE = 22.0


def _tolerance(fits_path):
    fits = json.loads(Path(fits_path).read_text())["fits"]
    out = {}
    for sampler in ("rt", "hmc"):
        rows = []
        for dim in DIMS:
            cells = [fits[sampler][f"D{dim}__L{L}"] for L in LS]
            rows.append({"dim": dim,
                         "dphi": np.array([c["dphi"] for c in cells]),
                         "sigma_c": np.array([c["sigma_c"] for c in cells])})
        out[sampler] = rows
    return out


def _frontier(path):
    out = {"rt": {}, "hmc": {}, "full": {}}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "confirmed" or not row["cost_rows_per_ess"]:
                continue
            cost = float(row["cost_rows_per_ess"])
            if row["batch"]:
                out[row["sampler"]][int(row["batch"])] = cost
            else:
                out["full"][row["sampler"]] = cost
    return out


def _letter(axis, mark, xy=(0.015, 0.985)):
    """Panel letter inside the axes. The caption attaches three different data
    provenances to specific panels, so the letters have to stay; a title row of
    their own costs about 1.3 cm of plot height on the board, so they do not."""
    axis.text(xy[0], xy[1], mark, transform=axis.transAxes, fontsize=TITLE_SIZE,
              fontweight="bold", ha="left", va="top", zorder=7)


def _panel_tolerance(axis, fits_path):
    data = _tolerance(fits_path)
    for sampler, ramp in (("rt", RT_DIM), ("hmc", HMC_DIM)):
        for row in data[sampler]:
            colour = ramp[row["dim"]]
            axis.plot(row["dphi"], row["sigma_c"], "-o", color=colour, linewidth=2.6,
                      markersize=7, path_effects=lift(5.0), zorder=4)
            lift_label = 7 if sampler == "rt" else -8
            text_colour = colour
            axis.annotate(f"$D={row['dim']}$",
                          xy=(row["dphi"][0], row["sigma_c"][0]),
                          xytext=(7, lift_label), textcoords="offset points",
                          fontsize=14, color=text_colour, ha="left", va="center",
                          fontweight="bold")

    # guides sit just under the shallowest line of each family, so parallelism is
    # checked against the data rather than across a decade of empty axis
    span = np.array([0.0112, 0.106])
    for power, anchor, label in ((-1.0, 820.0, r"slope $-1$"),
                                 (-0.5, 30.0, r"slope $-\frac{1}{2}$")):
        guide = anchor * (span / 0.0123) ** power
        axis.plot(span, guide, ls=(0, (7, 5)), color=MUTED, linewidth=1.8, zorder=2)
        axis.annotate(label, xy=(span[0], guide[0]), xytext=(2, -20),
                      textcoords="offset points", fontsize=14.5, color=MUTED)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.0104, 0.175)
    # headroom above the topmost line so the ratio note has a clear corner to sit in
    axis.set_ylim(top=4.2e4)
    axis.set_xticks([0.0123, 0.0245, 0.0491, 0.0982])
    axis.set_xticklabels(["0.012", "0.025", "0.049", "0.098"])
    axis.set_xticks([], minor=True)
    axis.set_xlabel(r"step size  $h$", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel(r"$\sigma_c$", fontsize=STANDALONE_MATH_LABEL_SIZE)
    axis.grid(True, which="major", alpha=0.5)
    _letter(axis, "(a)")


def _panel_erosion(axis, arm_summary):
    summary = json.loads(Path(arm_summary).read_text())
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=2.2)
    entries = {
        entry["arm"]: entry
        for entry in summary["targets"]["superconductor"]["q6_step_exponent"]["entries"]
        if entry["L"] == 128
    }
    for arm, colour, label in (
        ("frozen_symmetrized", RT_BLUE, "idealized gradient noise"),
        ("real_signed", REAL_ARM, "real minibatch noise"),
    ):
        octaves = entries[arm]["matched_window_local_exponents"]
        x = np.arange(1, len(octaves) + 1)
        y = np.array([o["p"] for o in octaves])
        # These are adjacent-pair local exponents, not samples from one fitted
        # line. A light connector shows their progression without implying a
        # global exponent for real minibatch noise.
        axis.plot(x, y, "-o", color=colour, linewidth=2.0, alpha=0.85,
                  markersize=8,
                  path_effects=lift(6.0), zorder=3)
        axis.annotate(label, xy=(x[-1], y[-1]),
                      xytext=(-7, 10 if arm == "frozen_symmetrized" else -6),
                      textcoords="offset points", ha="right",
                      va="bottom" if arm == "frozen_symmetrized" else "top",
                      fontsize=13.5, color=colour, fontweight="bold")

    axis.set_xlim(0.86, 3.14)
    axis.set_xticks([1, 2, 3])
    axis.set_xticklabels([
        r"$0.098\!\to\!0.049$",
        r"$0.049\!\to\!0.025$",
        r"$0.025\!\to\!0.012$",
    ])
    axis.set_xlabel("step-size halving", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel("local exponent  $p$", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylim(0.42, 1.13)
    axis.grid(True, axis="y", alpha=0.4)
    _letter(axis, "(b)")


def _panel_frontier(axis, frontier_path):
    real = _frontier(frontier_path)
    n_train = 17010
    for sampler, colour, label in (("rt", RT_BLUE, "ray tracing"),
                                   ("hmc", HMC_MAGENTA, "minibatch HMC")):
        batches = sorted(real[sampler])
        passes = [real[sampler][b] / n_train for b in batches]
        axis.loglog(batches, passes, "o-", color=colour,
                    path_effects=lift(6.0), zorder=3)
        last = max(batches)
        text_colour = RT_TEXT if sampler == "rt" else colour
        axis.annotate(label, xy=(last, real[sampler][last] / n_train),
                      xytext=(-4, -20 if sampler == "rt" else 14),
                      textcoords="offset points", ha="right", fontsize=14,
                      color=text_colour, fontweight="bold", zorder=6)
    full_cost = real["full"]["rt"] / n_train
    axis.axhline(full_cost, color=INK, linestyle=":", linewidth=2.4)
    axis.text(17, full_cost * 1.12, "full batch", color=INK, fontsize=14)

    axis.set_xlabel(r"batch size  $B$", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel("dataset passes per\neffective sample",
                    fontsize=AXIS_LABEL_SIZE)
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted(real["rt"]))
    axis.set_xticklabels([str(b) for b in sorted(real["rt"])])
    axis.set_yticks([1, 2, 5, 10, 20, 30])
    axis.set_yticklabels(["1", "2", "5", "10", "20", "30"])
    axis.minorticks_off()
    axis.grid(True, which="major", axis="y", alpha=0.4)
    _letter(axis, "(c)")


def figure_tolerance_erosion_frontier(out_dir, fits_path, arm_summary, frontier_path):
    # Sized to render at roughly 1:1 in a full-width column 3 slot. The earlier
    # 19.4 in canvas was being shrunk to 72% there, which is what made the panels
    # read as small however wide the slot was.
    figure = plt.figure(figsize=(14.0, 5.10))
    # The larger first gutter separates the synthetic experiment from the two
    # real-data panels; b and c remain close enough to read as one UCI group.
    a = figure.add_axes([0.065, 0.150, 0.255, 0.725])
    b = figure.add_axes([0.410, 0.150, 0.240, 0.725])
    c = figure.add_axes([0.725, 0.150, 0.260, 0.725])
    _panel_tolerance(a, fits_path)
    _panel_erosion(b, arm_summary)
    _panel_frontier(c, frontier_path)
    figure.text(0.192, 0.965, "Idealized isotropic gradient noise",
                ha="center", va="top", fontsize=16.0, color=INK,
                fontweight="bold")
    figure.text(0.700, 0.965, "UCI Superconductivity dataset",
                ha="center", va="top", fontsize=16.0, color=INK,
                fontweight="bold")
    save(figure, out_dir, "tolerance_erosion_frontier")


if __name__ == "__main__":
    import sources

    use_poster_style()
    figure_tolerance_erosion_frontier(sources.OUT_DIR, sources.DIMENSION_FITS,
                                      sources.ARM_SUMMARY, sources.FRONTIER_REAL)
