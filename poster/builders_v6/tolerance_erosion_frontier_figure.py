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

    # the gap between the families is the exponent difference made visible: it grows
    # by root two with every halving, so quote it at both ends rather than averaging
    ratios = {}
    for row_rt, row_hmc in zip(data["rt"], data["hmc"]):
        if row_rt["dim"] != 1024:
            continue
        ratios = dict(zip(row_rt["dphi"], row_rt["sigma_c"] / row_hmc["sigma_c"]))
    coarse, fine = max(ratios), min(ratios)
    # parked top right: the lines run down to the right, so that corner is the only
    # empty one, and the -1/2 guide label owns the bottom left
    axis.text(0.975, 0.96,
              rf"${ratios[coarse]:.0f}\times$ to ${ratios[fine]:.0f}\times$ more noise"
              "\n"
              rf"tolerated, $\sqrt{{2}}$ per halving",
              transform=axis.transAxes, ha="right", va="top", fontsize=13.5,
              color=INK, linespacing=1.3)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.0104, 0.175)
    # headroom above the topmost line so the ratio note has a clear corner to sit in
    axis.set_ylim(top=4.2e4)
    axis.set_xticks([0.0123, 0.0245, 0.0491, 0.0982])
    axis.set_xticklabels(["0.012", "0.025", "0.049", "0.098"])
    axis.set_xticks([], minor=True)
    axis.set_xlabel(r"per-step turning angle  $\Delta\phi$")
    axis.set_ylabel(r"gradient noise tolerated  $\sigma_c$")
    axis.grid(True, which="major", alpha=0.5)
    _letter(axis, "(a)")


def _panel_erosion(axis, arm_summary):
    summary = json.loads(Path(arm_summary).read_text())
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=2.2)
    styles = {("synthetic", "frozen_symmetrized"): ("o-", RT_BLUE, 1.0),
              ("superconductor", "frozen_symmetrized"): ("s--", RT_BLUE, 0.68),
              ("synthetic", "real_signed"): ("o-", REAL_ARM, 1.0),
              ("superconductor", "real_signed"): ("s--", REAL_ARM, 0.68)}
    names = {"frozen_symmetrized": "idealized", "real_signed": "real mini-batch"}
    order = [("synthetic", "frozen_symmetrized"),
             ("superconductor", "frozen_symmetrized"),
             ("synthetic", "real_signed"),
             ("superconductor", "real_signed")]
    entries = {(target, entry["arm"]): entry
               for target in ("synthetic", "superconductor")
               for entry in summary["targets"][target]["q6_step_exponent"]["entries"]
               if entry["L"] == 128}
    for target, arm in order:
        style, colour, alpha = styles[(target, arm)]
        octaves = entries[(target, arm)]["matched_window_local_exponents"]
        mid = [np.sqrt(o["step_size_coarse"] * o["step_size_fine"]) for o in octaves]
        # only the synthetic arm carries a legend entry: the two targets are told
        # apart by line style, which the note below states once
        label = names[arm] if target == "synthetic" else None
        axis.plot(mid, [o["p"] for o in octaves], style, color=colour, alpha=alpha,
                  path_effects=lift(6.0), zorder=3, label=label)

    axis.set_xscale("log")
    axis.invert_xaxis() # finer steps to the right, so the decay reads downhill
    axis.set_xticks([0.069, 0.0347, 0.0173])
    axis.set_xticklabels(["0.098→0.049", "0.049→0.025", "0.025→0.012"], fontsize=13.5)
    axis.minorticks_off()
    axis.set_xlabel("each halving of the step size")
    axis.set_ylabel("local exponent  $p$  per halving")
    axis.set_ylim(0.42, 1.13)
    axis.grid(True, axis="y", alpha=0.4)
    axis.text(0.97, 0.95, r"ray tracing,  $D = 81$,  $L = 128$", color=FINE_PRINT,
              fontsize=13.5, ha="right", va="top", transform=axis.transAxes)
    axis.legend(loc="lower left", fontsize=14, handlelength=2.0,
                frameon=True, facecolor=PANEL, framealpha=0.93)
    _letter(axis, "(b)")


def _panel_frontier(axis, frontier_path):
    real = _frontier(frontier_path)
    for sampler, colour, label in (("rt", RT_BLUE, "ray tracing"),
                                   ("hmc", HMC_MAGENTA, "stochastic HMC")):
        batches = sorted(real[sampler])
        axis.loglog(batches, [real[sampler][b] for b in batches], "o-", color=colour,
                    path_effects=lift(6.0), zorder=3)
        last = max(batches)
        text_colour = RT_TEXT if sampler == "rt" else colour
        axis.annotate(label, xy=(last, real[sampler][last]),
                      xytext=(-4, -20 if sampler == "rt" else 14),
                      textcoords="offset points", ha="right", fontsize=14,
                      color=text_colour, fontweight="bold", zorder=6)
    full_cost = real["full"]["rt"]
    axis.axhline(full_cost, color=INK, linestyle=":", linewidth=2.4)
    axis.text(17, full_cost * 1.12, "full batch, no noise", color=INK, fontsize=14)
    best_rt, best_hmc = min(real["rt"].values()), min(real["hmc"].values())
    for best, colour, text_colour, text in (
            (best_rt, RT_BLUE, RT_TEXT,
             rf"${full_cost / best_rt:.1f}\times$ cheaper"),
            (best_hmc, HMC_MAGENTA, HMC_MAGENTA,
             rf"${best_hmc / full_cost:.1f}\times$ dearer")):
        axis.annotate("", xy=(16, best), xytext=(16, full_cost),
                      arrowprops=dict(arrowstyle="<->", color=colour, linewidth=2.4))
        axis.text(18, np.sqrt(best * full_cost), text, color=text_colour, fontsize=16.5,
                  va="center", fontweight="bold")

    axis.set_xlabel("batch size")
    axis.set_ylabel("rows per effective sample")
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted(real["rt"]))
    axis.set_xticklabels([str(b) for b in sorted(real["rt"])])
    axis.set_yticks([2e4, 5e4, 1e5, 2e5, 5e5])
    axis.set_yticklabels(["20k", "50k", "100k", "200k", "500k"])
    axis.minorticks_off()
    axis.grid(True, which="major", axis="y", alpha=0.4)
    _letter(axis, "(c)")


def figure_tolerance_erosion_frontier(out_dir, fits_path, arm_summary, frontier_path):
    # Sized to render at roughly 1:1 in a full-width column 3 slot. The earlier
    # 19.4 in canvas was being shrunk to 72% there, which is what made the panels
    # read as small however wide the slot was.
    figure, (a, b, c) = plt.subplots(1, 3, figsize=(14.0, 4.95))
    _panel_tolerance(a, fits_path)
    _panel_erosion(b, arm_summary)
    _panel_frontier(c, frontier_path)
    # Keep every caption line narrower than the axes: bbox_inches="tight" lets a
    # long line set the saved width, which flattens the aspect and costs the board
    # the figure height this canvas was sized for.
    figure.text(0.5, 0.012,
                "(a) isotropic jitter, exact Gaussian target.   "
                "(b) matched-condition arms; solid synthetic, dashed superconductor.   "
                "(c) UCI superconductivity.",
                ha="center", va="bottom", fontsize=13, color=FINE_PRINT)
    figure.subplots_adjust(left=0.065, right=0.99, top=0.975, bottom=0.225, wspace=0.275)
    save(figure, out_dir, "tolerance_erosion_frontier")


if __name__ == "__main__":
    import sources

    use_poster_style()
    figure_tolerance_erosion_frontier(sources.OUT_DIR, sources.DIMENSION_FITS,
                                      sources.ARM_SUMMARY, sources.FRONTIER_REAL)
