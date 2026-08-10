#!/usr/bin/env python3
"""A reduced-element candidate for the poster's three measured-results panels.

This intentionally writes a new artifact and never replaces the current Figure 5.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from style import INK, MUTED, save, use_poster_style
from tolerance_erosion_frontier_figure import (
    HMC_DIM,
    HMC_MAGENTA,
    REAL_ARM,
    RT_BLUE,
    RT_DIM,
    RT_TEXT,
    _frontier,
    _letter,
    _tolerance,
)


def _panel_tolerance_reduced(axis, fits_path):
    data = _tolerance(fits_path)
    for sampler, ramp in (("rt", RT_DIM), ("hmc", HMC_DIM)):
        for row in data[sampler]:
            axis.plot(row["dphi"], row["sigma_c"], "-o",
                      color=ramp[row["dim"]], linewidth=2.25,
                      markersize=5.5, zorder=3)
            lift_label = 5 if sampler == "rt" else -6
            axis.annotate(f"$D={row['dim']}$",
                          xy=(row["dphi"][0], row["sigma_c"][0]),
                          xytext=(6, lift_label), textcoords="offset points",
                          fontsize=11.2, color=ramp[row["dim"]],
                          ha="left", va="center", fontweight="bold")

    # Retain the two visual slope checks, but keep them lighter and more compact
    # than in the current poster figure.
    span = np.array([0.0112, 0.106])
    for power, anchor, label in (
        (-1.0, 820.0, r"slope $=-1$"),
        (-0.5, 30.0, r"slope $=-\frac{1}{2}$"),
    ):
        guide = anchor * (span / 0.0123) ** power
        axis.plot(span, guide, ls=(0, (7, 5)), color=MUTED,
                  linewidth=1.45, zorder=1)
        axis.annotate(label, xy=(span[0], guide[0]), xytext=(2, -24),
                      textcoords="offset points", fontsize=11.5, color=MUTED)

    axis.text(0.97, 0.96,
              r"At $D=1024$, RT tolerates 10 to 28$\times$"
              "\n" r"larger $\sigma_c$ than HMC",
              transform=axis.transAxes, ha="right", va="top",
              fontsize=11.3, color=INK, linespacing=1.15)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.0104, 0.175)
    axis.set_ylim(top=4.2e4)
    axis.set_xticks([0.0123, 0.0245, 0.0491, 0.0982])
    axis.set_xticklabels(["0.012", "0.025", "0.049", "0.098"])
    axis.set_xticks([], minor=True)
    axis.set_xlabel(r"per-step turning angle  $\Delta\phi$")
    axis.set_ylabel(r"gradient noise tolerated  $\sigma_c$")
    axis.grid(False)
    _letter(axis, "(a)")


def _panel_erosion_reduced(axis, arm_summary):
    summary = json.loads(Path(arm_summary).read_text())
    entries = {
        entry["arm"]: entry
        for entry in summary["targets"]["superconductor"]["q6_step_exponent"]["entries"]
        if entry["L"] == 128
    }
    styles = (
        ("frozen_symmetrized", RT_BLUE, "idealized"),
        ("real_signed", REAL_ARM, "real mini-batch"),
    )
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=1.7, zorder=1)
    axis.text(1.04, 1.012, r"$p=1$", color=INK, fontsize=11.5,
              ha="left", va="bottom")

    for arm, colour, label in styles:
        octaves = entries[arm]["matched_window_local_exponents"]
        x = np.arange(1, len(octaves) + 1)
        y = np.array([o["p"] for o in octaves])
        axis.plot(x, y, "-o", color=colour, linewidth=2.6,
                  markersize=7.0, zorder=3)
        axis.annotate(label, xy=(x[-1], y[-1]),
                      xytext=(-7, 9 if arm == "frozen_symmetrized" else -5),
                      textcoords="offset points", ha="right",
                      va="bottom" if arm == "frozen_symmetrized" else "top",
                      fontsize=13.0, color=colour, fontweight="bold")

    axis.set_xlim(0.86, 3.14)
    axis.set_xticks([1, 2, 3])
    axis.set_xticklabels(["1st", "2nd", "3rd"])
    axis.set_xlabel("successive step-size halvings")
    axis.set_ylabel(r"local exponent  $p$")
    axis.set_ylim(0.42, 1.13)
    axis.grid(False)
    _letter(axis, "(b)")


def _panel_frontier_reduced(axis, frontier_path):
    real = _frontier(frontier_path)
    for sampler, colour, text_colour, label in (
        ("rt", RT_BLUE, RT_TEXT, "ray tracing"),
        ("hmc", HMC_MAGENTA, HMC_MAGENTA, "stochastic HMC"),
    ):
        batches = sorted(real[sampler])
        costs = [real[sampler][batch] for batch in batches]
        axis.loglog(batches, costs, "o-", color=colour,
                    linewidth=2.6, markersize=7.0, zorder=3)
        axis.annotate(label, xy=(batches[-1], costs[-1]),
                      xytext=(-4, -18 if sampler == "rt" else 11),
                      textcoords="offset points", ha="right", fontsize=13.0,
                      color=text_colour, fontweight="bold")

    full_cost = real["full"]["rt"]
    axis.axhline(full_cost, color=INK, linestyle=":", linewidth=1.9, zorder=1)
    axis.text(18, full_cost * 1.06, "full batch", color=INK,
              fontsize=11.8, ha="left", va="bottom")

    best_rt = min(real["rt"].values())
    best_hmc = min(real["hmc"].values())
    for best, colour, text_colour, text in (
        (best_rt, RT_BLUE, RT_TEXT,
         rf"{full_cost / best_rt:.1f}$\times$ cheaper"),
        (best_hmc, HMC_MAGENTA, HMC_MAGENTA,
         rf"{best_hmc / full_cost:.1f}$\times$ dearer"),
    ):
        axis.annotate("", xy=(16, best), xytext=(16, full_cost),
                      arrowprops=dict(arrowstyle="<->", color=colour,
                                      linewidth=2.0))
        axis.text(18, np.sqrt(best * full_cost), text,
                  color=text_colour, fontsize=14.0, va="center",
                  fontweight="bold")

    batches = sorted(real["rt"])
    axis.set_xlabel("batch size")
    axis.set_ylabel("rows per effective sample")
    axis.set_xscale("log", base=2)
    axis.set_xticks(batches)
    axis.set_xticklabels([str(batch) for batch in batches])
    axis.set_yticks([2e4, 5e4, 1e5, 2e5, 5e5])
    axis.set_yticklabels(["20k", "50k", "100k", "200k", "500k"])
    axis.minorticks_off()
    axis.grid(False)
    _letter(axis, "(c)")


def figure_candidate(out_dir, fits_path, arm_summary, frontier_path):
    figure, (a, b, c) = plt.subplots(1, 3, figsize=(14.0, 4.95))
    _panel_tolerance_reduced(a, fits_path)
    _panel_erosion_reduced(b, arm_summary)
    _panel_frontier_reduced(c, frontier_path)
    figure.text(
        0.5, 0.012,
        r"(a) Gaussian target.   (b,c) UCI superconductivity.",
        ha="center", va="bottom", fontsize=12.5, color=MUTED,
    )
    # Intended for a full-column (1.00 linewidth) poster placement. The larger
    # gutters keep all three y-labels visually attached to their own panels.
    figure.subplots_adjust(left=0.06, right=0.995, top=0.975,
                           bottom=0.225, wspace=0.40)
    save(figure, out_dir, "tolerance_erosion_frontier_candidate")


if __name__ == "__main__":
    import sources

    use_poster_style()
    figure_candidate(sources.OUT_DIR, sources.DIMENSION_FITS,
                     sources.ARM_SUMMARY, sources.FRONTIER_REAL)
