#!/usr/bin/env python3
"""Poster introduction graphic: both movers in one stratified target density.

Real dynamics on the smooth field (leapfrog and the ray ODE): same start, same
tilt, and HMC's launch speed is set to RT's measured climb so the paths reach
the same target-density level before returning. The figure keeps only the
method names, the density direction, and equal-time dots (HMC's bunch at its
stall, RT's stay even).
"""
from __future__ import annotations

from pathlib import Path
import sys

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from matplotlib.colors import LinearSegmentedColormap

from style import INK, use_poster_style, save

C_HMC = "#B64F78"
C_RT = "#2F7EBB"
C_RT_TEXT = "#225C8A"
C_LABEL = "#4A5160"
# The poster's figures are airy: the medium ramps from just off the panel
# colour to a tint near v6's MEDIUM_HIGH, so the top dissolves into the panel
# and the paths stay the darkest objects in the frame.
MEDIUM = LinearSegmentedColormap.from_list(
    "medium", ["#F3F5F9", "#BCD1E7"])

BASE = 2.46 # ridge offset chosen so the launch sits at u0 = 1.14
WOBBLE_A, WOBBLE_K = 0.18, 1.15 # single calm harmonic: height tracks
# target density closely enough that the matched level meets both visual apexes
DELTA = 0.54 # log-likelihood drop per drawn band; sparse enough at poster scale
TILT_DEG = 6.0 # a shallower launch flattens the scene for the column slot
START = np.array([0.0, -1.50]) # a wobble crest, mid-band
STEP = 0.003
N_DOTS = 13 # equal-time dots per HMC excursion; RT shares the interval

X_LO, X_HI, Y_LO, Y_HI = -2.75, 1.55, -1.64, 0.14
FIG_W, FIG_H = 6.9, 2.9
# Printed at 0.62 of the column, so in-figure text is raised to keep pace with
# the mechanism figure's printed size.
METHOD, NOTE, KEY = 16.5, 15.5, 12.5


def wobble(x):
    # even in x, so the mirrored launches stay exactly equivalent
    return WOBBLE_A * np.cos(WOBBLE_K * x)


def logp(x, y):
    u = y + BASE + wobble(x)
    return -u ** 2 / 2


def grad(p):
    eps = 1e-5
    x, y = p
    fx = (logp(x + eps, y) - logp(x - eps, y)) / (2 * eps)
    fy = (logp(x, y + eps) - logp(x, y - eps)) / (2 * eps)
    return np.array([fx, fy])


F0 = logp(*START)
EDGES = F0 + DELTA / 2 - DELTA * np.arange(0, 14)


def rt_path():
    tilt = np.deg2rad(TILT_DEG)
    u = np.array([-np.sin(tilt), np.cos(tilt)]) # mirrored: field is even in x
    x = START.copy()
    pts = [x.copy()]
    for _ in range(300000):
        x = x + 0.5 * STEP * u
        g = grad(x) # D = 2, so grad log n equals grad log pi
        u = u + STEP * (g - (u @ g) * u)
        u = u / np.linalg.norm(u)
        x = x + 0.5 * STEP * u
        pts.append(x.copy())
        if x[1] < START[1] and len(pts) > 100:
            break
    return np.array(pts)


def hmc_path(speed):
    tilt = np.deg2rad(TILT_DEG)
    v = np.array([np.sin(tilt), np.cos(tilt)]) * speed
    x = START.copy()
    pts = [x.copy()]
    for _ in range(300000):
        v = v + 0.5 * STEP * grad(x)
        x = x + STEP * v
        v = v + 0.5 * STEP * grad(x)
        pts.append(x.copy())
        if x[1] < START[1] and len(pts) > 100:
            break
    return np.array(pts)


def draw_path(ax, pts, colour, dot_stride):
    ax.plot(pts[:, 0], pts[:, 1], color="white", lw=4.3, zorder=3,
            solid_capstyle="round")
    ax.plot(pts[:, 0], pts[:, 1], color=colour, lw=2.7, zorder=4,
            solid_capstyle="round")
    marks = pts[dot_stride::dot_stride]
    ax.plot(marks[:, 0], marks[:, 1], "o", ms=5.2, color=colour, zorder=6,
            markeredgecolor="white", markeredgewidth=1.1)
    seg = pts[-1] - pts[-40]
    tip = pts[-1] + seg / np.linalg.norm(seg) * 0.02
    ax.add_patch(FancyArrowPatch(pts[-1] - seg * 0.4, tip, arrowstyle="-|>",
                                 mutation_scale=12, color=colour, lw=2.0,
                                 zorder=7, shrinkA=0, shrinkB=0))


def figure_intro_dynamics(out_dir):
    rt_pts = rt_path()
    climb = float(F0 - min(logp(*p) for p in rt_pts))
    hmc_pts = hmc_path(np.sqrt(2.0 * climb))
    dot_stride = max(1, len(hmc_pts) // N_DOTS)

    figure, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    mesh_x, mesh_y = np.meshgrid(np.linspace(X_LO, X_HI, 700),
                                 np.linspace(Y_LO, Y_HI, 480))
    f = logp(mesh_x, mesh_y)
    visible = np.sort(EDGES[(EDGES > f.min()) & (EDGES < f.max())])
    fills = np.concatenate([[f.min() - 0.1], visible, [f.max() + 0.1]])
    ax.contourf(mesh_x, mesh_y, f, levels=fills, cmap=MEDIUM, zorder=0)
    ax.contour(mesh_x, mesh_y, f, levels=visible, colors="white",
               linewidths=1.0, linestyles=[(0, (4, 4))], alpha=0.95, zorder=1)
    draw_path(ax, hmc_pts, C_HMC, dot_stride)
    draw_path(ax, rt_pts, C_RT, dot_stride)
    ax.plot(*START, "o", ms=9.0, markerfacecolor="white", markeredgecolor=INK,
            markeredgewidth=1.6, zorder=8)

    ax.text(0.55, -0.60, "HMC", fontsize=METHOD, fontweight="bold",
            color=C_HMC, ha="left", va="center")
    ax.text(-1.45, -0.60, "RT", fontsize=METHOD, fontweight="bold",
            color=C_RT_TEXT, ha="right", va="center")
    # A compact reading key replaces an extra sentence in the poster text.
    key_y = 0.915
    key_x = np.array([0.035, 0.075])
    ax.plot(key_x, [key_y, key_y], transform=ax.transAxes, color=C_LABEL,
            lw=1.2, zorder=9, solid_capstyle="round")
    ax.plot(key_x, [key_y, key_y], "o", transform=ax.transAxes, ms=4.6,
            color=C_LABEL, markeredgecolor="white", markeredgewidth=0.8,
            zorder=10)
    ax.text(0.092, key_y, "equal time between dots", transform=ax.transAxes,
            fontsize=KEY, color=C_LABEL, ha="left", va="center", zorder=10)
    # target density increases downward, into the dense medium
    arrow_x = X_LO + 0.30
    ax.add_patch(FancyArrowPatch((arrow_x, -0.52), (arrow_x, -1.25),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=C_LABEL, lw=1.6, zorder=5,
                                 shrinkA=0, shrinkB=0))
    ax.text(arrow_x - 0.14, -0.885, "higher density", fontsize=NOTE,
            color=C_LABEL, ha="center", va="center", rotation=90)

    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    figure.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    save(figure, out_dir, "intro_dynamics")


if __name__ == "__main__":
    import sources

    use_poster_style()
    figure_intro_dynamics(sources.OUT_DIR)
