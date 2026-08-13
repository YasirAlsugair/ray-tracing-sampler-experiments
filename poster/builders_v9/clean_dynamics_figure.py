#!/usr/bin/env python3
"""Clean-dynamics opener: one correlated Gaussian, HMC rolling and RT bending.

Both panels integrate the real two-dimensional dynamics on the same target,
whose density field is read two ways: a potential for HMC, a refractive medium
for RT. Dots mark equal time intervals, so varying spacing is HMC's speed
trading against position and even spacing is RT's constant speed. No refreshes
are drawn: this is the deterministic flow between refreshes, and it makes no
performance claim.
"""
from __future__ import annotations

from pathlib import Path
import sys

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyArrowPatch

from style import FIELD, save, use_poster_style

# Same figure-local pair as the mechanism figure in this block.
C_HMC = "#B64F78"
C_RT = "#2F7EBB"
C_RT_TEXT = "#225C8A"
C_NOTE = "#3C4250"

A, B = 2.0, 0.7 # target standard deviations; the ridge runs along x
PREC = np.diag([1.0 / A ** 2, 1.0 / B ** 2])

SPAN_X, SPAN_Y = 4.35, 1.45
FIG_W, FIG_H = 10.2, 2.02
NAME, CAPTION = 15.0, 12.0

HMC_START, HMC_V0 = np.array([-2.85, 0.0]), np.array([0.10, 1.05])
HMC_DT, HMC_STEPS, HMC_DOT_EVERY = 0.02, 760, 50
RT_START, RT_ANGLE = np.array([-2.90, 0.45]), -26.0
RT_DS, RT_STEPS, RT_DOT_EVERY = 0.01, 1300, 100


def _halo(width=2.6):
    return [path_effects.withStroke(linewidth=width, foreground="white")]


def _grad(x):
    return -PREC @ x


def _hmc_path(x0, v0, dt, n):
    x, v = x0.astype(float).copy(), v0.astype(float).copy()
    out = [x.copy()]
    for _ in range(n):
        v += 0.5 * dt * _grad(x)
        x += dt * v
        v += 0.5 * dt * _grad(x)
        out.append(x.copy())
    return np.asarray(out)


def _rt_path(x0, angle_deg, ds, n):
    x = x0.astype(float).copy()
    t = np.deg2rad(angle_deg)
    u = np.array([np.cos(t), np.sin(t)])
    out = [x.copy()]
    for _ in range(n):
        x += 0.5 * ds * u
        g = _grad(x) # D = 2, so grad log n equals grad log pi
        u += ds * (g - (u @ g) * u)
        u /= np.linalg.norm(u)
        x += 0.5 * ds * u
        out.append(x.copy())
    return np.asarray(out)


def _field(ax):
    mesh_x, mesh_y = np.meshgrid(np.linspace(-SPAN_X, SPAN_X, 600),
                                 np.linspace(-SPAN_Y, SPAN_Y, 240))
    stacked = np.stack([mesh_x, mesh_y], axis=-1)
    density = np.exp(-0.5 * np.einsum("...i,ij,...j->...", stacked, PREC, stacked))
    root = density ** 0.5 # so the two white contours sit at 1 and 2 sigma
    ax.contourf(mesh_x, mesh_y, root, levels=np.linspace(0, 1, 29), cmap=FIELD,
                zorder=0)
    ax.contour(mesh_x, mesh_y, root, levels=[np.exp(-1.0), np.exp(-0.25)],
               colors="white", linewidths=0.8, alpha=0.75, zorder=1)


def _panel(ax, path, dot_every, colour, name, name_colour, caption):
    _field(ax)
    ax.plot(path[:, 0], path[:, 1], color="white", lw=3.4, zorder=3,
            solid_capstyle="round")
    ax.plot(path[:, 0], path[:, 1], color=colour, lw=1.8, zorder=4,
            solid_capstyle="round")
    dots = path[::dot_every]
    ax.plot(dots[:, 0], dots[:, 1], "o", ms=4.6, color=colour, zorder=6,
            markeredgecolor="white", markeredgewidth=0.9)
    ax.plot(*path[0], "o", ms=8.6, markerfacecolor="white", markeredgecolor=colour,
            markeredgewidth=1.6, zorder=7)
    tip = path[-1] + (path[-1] - path[-12]) * 0.5
    ax.add_patch(FancyArrowPatch(path[-12], tip, arrowstyle="-|>",
                                 mutation_scale=13, color=colour, lw=1.8,
                                 zorder=7, shrinkA=0, shrinkB=0))
    # name and caption live in a band above the field, clear of any trajectory
    ax.text(0.0, 1.10, name, transform=ax.transAxes, ha="left", va="center",
            fontsize=NAME, fontweight="bold", color=name_colour)
    ax.text(1.0, 1.10, caption, transform=ax.transAxes, ha="right", va="center",
            fontsize=CAPTION, color=C_NOTE)
    ax.set_xlim(-SPAN_X, SPAN_X)
    ax.set_ylim(-SPAN_Y, SPAN_Y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def figure_clean_dynamics(out_dir):
    hmc = _hmc_path(HMC_START, HMC_V0, HMC_DT, HMC_STEPS)
    ray = _rt_path(RT_START, RT_ANGLE, RT_DS, RT_STEPS)

    figure, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    _panel(ax_a, hmc, HMC_DOT_EVERY, C_HMC, "HMC", C_HMC,
           "fast in the core, slow at the edges")
    _panel(ax_b, ray, RT_DOT_EVERY, C_RT, "ray tracing", C_RT_TEXT,
           "constant speed, only the direction bends")
    ax_a.text(0.025, 0.09, "dots: equal time intervals", transform=ax_a.transAxes,
              ha="left", va="center", fontsize=CAPTION, color=C_NOTE,
              path_effects=_halo())

    figure.subplots_adjust(left=0.005, right=0.995, top=0.845, bottom=0.01,
                           wspace=0.035)
    save(figure, out_dir, "clean_dynamics")


if __name__ == "__main__":
    import sources

    use_poster_style()
    figure_clean_dynamics(sources.OUT_DIR)
