#!/usr/bin/env python3
"""v6 typical-set figure with a locally balanced blue/orange/teal palette."""
from __future__ import annotations

from pathlib import Path
import sys

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi

import sources
from style import INK, save, use_poster_style

DENSITY = "#2F80C9"
VOLUME = "#F39C12"
MASS = "#0B4F6C"
MASS_FILL = "#CFE2E9"
F_BASE = 12
F_LABEL = 12.5


def figure_typical_set(out_dir):
    use_poster_style()
    plt.rcParams.update({
        "font.size": F_BASE,
        "axes.titlesize": F_LABEL,
        "axes.labelsize": F_LABEL,
        "xtick.labelsize": F_BASE,
        "ytick.labelsize": F_BASE,
        "savefig.bbox": None,
    })

    dimension = 100
    sqrt_dimension = float(np.sqrt(dimension))
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(6.3, 2.38), layout="constrained")
    figure.get_layout_engine().set(h_pad=0.04, w_pad=0.05)

    radius_2d = np.linspace(0.0, 4.2, 800)
    density = np.exp(-0.5 * radius_2d ** 2)
    volume = radius_2d / 4.2
    mass = radius_2d * density
    mass /= mass.max()
    left.plot(radius_2d, density, color=DENSITY, lw=1.7)
    left.plot(radius_2d, volume, color=VOLUME, lw=1.7, ls="--")
    left.plot(radius_2d, mass, color=MASS, lw=2.1)
    left.fill_between(radius_2d, mass, 0.0, color=MASS_FILL, alpha=0.82, lw=0)
    left.text(0.10, 1.07, "density", color=DENSITY, ha="left", va="center")
    left.text(1.35, 1.16, "mass", color=MASS, ha="left", va="center")
    left.text(4.15, 0.52, "volume", color=VOLUME, ha="right", va="center")
    left.set_title(r"$D=2$: density $\times$ volume $=$ mass", loc="left",
                   fontsize=F_LABEL, pad=8)
    left.set_xlim(0.0, 4.2)
    left.set_ylim(0.0, 1.40)
    left.set_xticks([0, 1, 2, 3, 4])
    left.set_yticks([])
    left.spines["left"].set_visible(False)

    radius = np.linspace(0.0, sqrt_dimension + 3.5, 1600)
    for dim, label_x, label_y, align in (
            (1, 0.55, 0.84, "left"), (10, 3.0, 0.70, "center"),
            (100, sqrt_dimension, 0.70, "center")):
        radial_mass = chi.pdf(radius, dim)
        right.fill_between(radius, radial_mass, 0.0,
                           color=MASS_FILL, alpha=0.82, lw=0)
        right.plot(radius, radial_mass, color=MASS, lw=1.7)
        right.text(label_x, label_y, rf"$D = {dim}$", color=INK,
                   ha=align, va="center")

    mean = float(chi.mean(dimension))
    sd = float(chi.std(dimension))
    right.annotate("", xy=(mean - sd, 0.20), xytext=(mean + sd, 0.20),
                   arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2,
                                   shrinkA=0, shrinkB=0))
    right.text(mean, 0.32, r"$\sqrt{2}$", color=INK, ha="center", va="center")
    right.set_title(r"Higher $D$: mass peaks at $r\approx\sqrt{D}$", loc="left",
                    fontsize=F_LABEL, pad=8)
    right.set_xlim(0.0, sqrt_dimension + 3.5)
    right.set_ylim(0.0, 0.95)
    right.set_xticks([0.0, np.sqrt(10.0), sqrt_dimension],
                     labels=["0", r"$\sqrt{10}$", r"$\sqrt{100}$"])
    right.set_yticks([])
    right.spines["left"].set_visible(False)
    figure.supxlabel(r"$r = \|\theta - \theta_{\mathrm{mode}}\|$", fontsize=F_LABEL)
    save(figure, out_dir, "typical_set")


if __name__ == "__main__":
    figure_typical_set(sources.OUT_DIR)
