#!/usr/bin/env python3
"""Poster mechanism graphic: gradient noise heats HMC, but only rotates ray tracing.

Velocity space throughout. Panel (a) is drawn geometry: an exact identity for the one
transverse component shown. Panel (b) is simulated: the real recursions run in D
dimensions and plotted as (exact speed, heading angle in a fixed reference plane).
Both samplers are fed the same noise sequence and matched on per-step turning angle.
"""
from __future__ import annotations

from pathlib import Path
import sys

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Arc, FancyArrowPatch

from style import INK, PANEL, save, use_poster_style

# Figure-local pair: raspberry and cobalt are vivid enough to carry the mechanism
# at poster distance, but neither reads as a warning/success traffic-light colour.
C_HMC = "#B64F78"
C_RT = "#2F7EBB"
C_RT_TEXT = "#225C8A"
# Cool neutrals rather than flat grey: the reference circle should recede behind the
# two coloured paths, and the headings should tie to the poster's block titles.
C_RING = "#AEBBD0"
C_DISC = "#E8EEF6"
C_LABEL = "#4A5160"
C_HEAD = "#002A5C"

DIM = 8
STEPS = 56
SEED = 20260814 # fixed before drawing, never reselected for appearance
MEAN_FINAL_SPEED = 1.62 # sets the noise amplitude, not the outcome
START_DEG = 95.0

# Wide and short, like the typical-set figure in the adjacent column. Direct
# labels replace the old equation captions, letting the geometry fill the slot.
FIG_W, FIG_H = 6.4, 2.70
HEAD, SYM, RING, METHOD = 15.0, 14.5, 11.5, 12.0


def polar(r, deg):
    t = np.deg2rad(deg)
    return np.array([r * np.cos(t), r * np.sin(t)])


def arrow(ax, p0, p1, color, lw=1.85, head=11, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head,
                                 color=color, lw=lw, zorder=z, shrinkA=0,
                                 shrinkB=0, capstyle="round", joinstyle="round"))


def ring_label(ax, r, deg, text, colour):
    """Name a reference circle out on the lower-left diagonal, clear of the captions."""
    ax.text(*polar(r + 0.23, deg), text, ha="center", va="center", fontsize=RING,
            color=colour,
            path_effects=[path_effects.withStroke(linewidth=0.45,
                                                   foreground=colour)])


def endpoint(ax, r, deg, colour, z):
    ax.plot(*polar(r, deg), "o", ms=9.8, color=colour, zorder=z,
            markeredgecolor="white", markeredgewidth=1.1)


def circle(ax, r=1.0, color=C_RING, lw=1.65, ls="-", z=2, alpha=1.0, fill=False):
    t = np.linspace(0, 2 * np.pi, 400)
    if fill:
        ax.fill(r * np.cos(t), r * np.sin(t), color=C_DISC, zorder=0)
    ax.plot(r * np.cos(t), r * np.sin(t), color=color, lw=lw, ls=ls,
            zorder=z, alpha=alpha, solid_capstyle="round")


def setup(ax, label, rmax):
    ax.set_xlim(-1.06 * rmax, 1.06 * rmax)
    ax.set_ylim(-0.88 * rmax, 1.06 * rmax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0, 0.99 * rmax, label, ha="center", va="center", fontsize=HEAD,
            fontweight="bold", color=C_HEAD)

def amplitude():
    """Kick size putting the mean-law final speed at MEAN_FINAL_SPEED."""
    return np.sqrt((MEAN_FINAL_SPEED ** 2 - 1.0) / (STEPS * DIM))


def trajectory(rt):
    """One real trajectory of velocities. Both samplers draw the same noise sequence,
    so the two differ only in how the sampler responds, as the two endpoints do in
    panel (a)."""
    rng = np.random.default_rng(SEED)
    amp = amplitude()
    v = np.zeros(DIM)
    v[0] = 1.0
    out = [v.copy()]
    for _ in range(STEPS):
        z = rng.normal(size=DIM)
        if rt:
            u = v / np.linalg.norm(v)
            v = u + amp * (z - (u @ z) * u)
            v = v / np.linalg.norm(v)
        else:
            v = v + amp * z
        out.append(v.copy())
    return np.array(out)


def reference_plane(states):
    """Plane spanned by v0 and the principal direction of the heading's excursion.

    A plane fixed by an arbitrary basis vector shows an arbitrary slice of the
    heading motion and can understate it to nothing. Taking the leading principal
    direction of the transverse heading displacement is a stated convention that
    shows the excursion the run actually made.
    """
    e1 = states[0] / np.linalg.norm(states[0])
    headings = states / np.linalg.norm(states, axis=1, keepdims=True)
    transverse = headings - np.outer(headings @ e1, e1)
    _, _, vt = np.linalg.svd(transverse, full_matrices=False)
    e2 = vt[0] - (vt[0] @ e1) * e1
    return e1, e2 / np.linalg.norm(e2)


def polar_readout(states, e1, e2):
    """Exact speed, and the heading's angle within the reference plane."""
    radii = np.linalg.norm(states, axis=1)
    angles = START_DEG + np.degrees(np.arctan2(states @ e2, states @ e1))
    return radii, angles


def draw_walk(ax, radii, angles, colour, lw, alpha, z, m=14):
    """Each step is its own translucent segment, so where the walk doubles back the
    overlap darkens. A single polyline would rasterize once and hide the retracing."""
    for i in range(len(radii) - 1):
        t = np.linspace(0, 1, m)
        rr = radii[i] * (1 - t) + radii[i + 1] * t
        aa = np.deg2rad(angles[i] * (1 - t) + angles[i + 1] * t)
        ax.plot(rr * np.cos(aa), rr * np.sin(aa), color=colour, lw=lw, alpha=alpha,
                zorder=z, solid_capstyle="round")


def _panel_one_step(ax, rmax):
    setup(ax, "one step", rmax)
    circle(ax, fill=True)

    th_v = 62.0
    P = polar(1.0, th_v)
    k = 0.92
    tan_cw = np.array([np.sin(np.deg2rad(th_v)), -np.cos(np.deg2rad(th_v))])
    Q = P + k * tan_cw
    rq = np.linalg.norm(Q)
    Pp = Q / rq
    th_q = np.rad2deg(np.arctan2(Q[1], Q[0]))

    circle(ax, r=rq, color=C_HMC, lw=1.45, ls=(0, (5, 5)), z=2, alpha=0.45)

    arrow(ax, (0, 0), P, INK, lw=2.35, head=12, z=6)
    arrow(ax, P, Q, INK, lw=2.15, head=11, z=6)

    # right-angle marker: the drawn noise component is perpendicular to v
    d1 = -(P / np.linalg.norm(P)) * 0.11
    d2 = tan_cw * 0.11
    sq = np.array([P + d1, P + d1 + d2, P + d2])
    ax.plot(sq[:, 0], sq[:, 1], color=INK, lw=0.9, zorder=6)

    arrow(ax, (0, 0), Q, C_HMC, lw=2.85, head=12, z=5)
    ax.plot(*Q, "o", ms=9.4, color=C_HMC, zorder=7)
    ax.add_patch(Arc((0, 0), 2, 2, theta1=th_q, theta2=th_v, lw=3.5,
                     color=C_RT, zorder=6, capstyle="round"))
    ax.plot(*Pp, "o", ms=9.4, color=C_RT, zorder=7)
    ax.plot([Pp[0], Q[0]], [Pp[1], Q[1]], color=C_HMC, lw=2.0, zorder=6)

    ax.text(*polar(0.52, th_v) + np.array([-0.18, 0.02]), "$v$", ha="center",
            va="center", fontsize=SYM, color=INK)
    ax.text(1.02, 0.86, "$P_v\\zeta$", ha="center", va="center", fontsize=SYM,
            color=INK, bbox=dict(facecolor=PANEL, edgecolor="none", pad=1.0))
    ring_label(ax, 1.0, 250.0, "$|v|$", C_LABEL)
    ring_label(ax, rq, 234.0, "$|v'|$", C_HMC)
    ax.annotate("HMC", xy=Q, xytext=(9, 0), textcoords="offset points",
                fontsize=METHOD, fontweight="bold", color=C_HMC,
                ha="left", va="center")
    ax.annotate("RT", xy=P, xytext=(-5, 16), textcoords="offset points",
                fontsize=METHOD, fontweight="bold", color=C_RT_TEXT,
                ha="center", va="center")


def _panel_n_steps(ax, rmax, paths):
    setup(ax, "$N$ steps", rmax)
    circle(ax, fill=True)

    rt_r, rt_a = paths[True]
    hmc_r, hmc_a = paths[False]
    draw_walk(ax, rt_r, rt_a, C_RT, 3.9, 0.62, 4)
    draw_walk(ax, hmc_r, hmc_a, C_HMC, 2.8, 0.72, 6)
    endpoint(ax, rt_r[-1], rt_a[-1], C_RT, 8)
    endpoint(ax, hmc_r[-1], hmc_a[-1], C_HMC, 9)

    outer = hmc_r[-1] # this run's own final speed, as in panel (a)
    circle(ax, r=outer, color=C_HMC, lw=1.45, ls=(0, (5, 5)), z=2, alpha=0.45)
    ax.plot(*polar(1.0, START_DEG), "o", ms=6.8, color=INK, zorder=10,
            markeredgecolor="white", markeredgewidth=0.9)

    ring_label(ax, 1.0, 250.0, "$|v_0|$", C_LABEL)
    ring_label(ax, outer, 234.0, "$|v_N|$", C_HMC)
    ax.annotate("HMC", xy=polar(hmc_r[-1], hmc_a[-1]), xytext=(-10, 10),
                textcoords="offset points", fontsize=METHOD, fontweight="bold",
                color=C_HMC, ha="right", va="bottom")
    ax.annotate("RT", xy=polar(rt_r[-1], rt_a[-1]), xytext=(2, -21),
                textcoords="offset points", fontsize=METHOD, fontweight="bold",
                color=C_RT_TEXT, ha="center", va="center")


def figure_noise_geometry(out_dir):
    states = {rt: trajectory(rt) for rt in (False, True)}
    # one plane for both paths, fixed by the ray-tracing run whose heading is the
    # quantity panel (b) is about
    e1, e2 = reference_plane(states[True])
    paths = {rt: polar_readout(s, e1, e2) for rt, s in states.items()}
    rmax = 1.25 * max(radii.max() for radii, _ in paths.values())

    figure, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    _panel_one_step(ax_a, rmax)
    _panel_n_steps(ax_b, rmax, paths)

    figure.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02, wspace=0.02)
    save(figure, out_dir, "noise_geometry")


if __name__ == "__main__":
    import sources

    use_poster_style()
    figure_noise_geometry(sources.OUT_DIR)
