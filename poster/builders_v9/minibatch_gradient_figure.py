#!/usr/bin/env python3
"""What a mini-batch gradient buys, against what it costs, on the poster's own target.

UCI Superconductivity BLR, D = 81, at one posterior draw, at the N that
studies/11 actually runs. For each batch size many independent row subsets are
drawn at that same point and each stochastic gradient is compared with the exact
one over all N rows.

Left, one batch size drawn out: every arrow's length and its angle to the exact
gradient are the true 81-dimensional values. This is not a projection. A projection
onto two coordinates would drop 79 of the 80 error directions and shorten every
arrow; here the polar coordinates are exact and only the choice of which side of
the plane an arrow falls on is arbitrary.

Right, the same angle against batch size. The reference line matters more than the
curve: in D = 81 a direction carrying no information sits at 90 degrees, so an angle
is only readable against that floor. What governs it is N/B rather than B, so the N
here is the study's, not a smaller stand-in.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] # the repository, for rtbench
for path in (Path(__file__).resolve().parent, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True) # the BLR targets require it
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from style import INK, MUTED, save, use_poster_style  # noqa: E402

# This is a gradient diagnostic rather than a sampler comparison. A warm coral
# makes the stochastic perturbations energetic without borrowing a method hue.
NOISE_CORAL = "#E07A5F"
MEAN_CORAL = "#C9553D"
REFERENCE_NAVY = "#19324D"
SEED = 20260814 # fixed before drawing, never reselected
N_TRAIN = 17010 # studies/11_superconductor_compute_frontier/config.json
BATCHES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 17010)
N_DRAWS = 300
# The left panel is drawn at a deliberately mild ratio. At the N/B the samplers
# actually run, the fan is isotropic and the exact gradient is a fifteenth of a
# typical arrow, so the panel shows nothing; the right panel carries that regime.
DRAWN_OUT = 2048
N_ARROWS = 70
ANNOTATION_SIZE = 18.5
CAPTION_SIZE = 18.5


def _fan(target, theta, batch, key, n_draws):
    """n_draws independent mini-batch gradients at the same theta."""
    keys = jax.random.split(key, n_draws)
    rows = jax.vmap(lambda k: jax.random.choice(
        k, target.n_train, shape=(batch,), replace=False))(keys)
    return np.asarray(jax.vmap(lambda r: target.grad_on_rows(theta, r))(rows))


def _angles(fan, exact):
    """Angle to the exact gradient, in degrees, measured in all D dimensions."""
    cosine = ((fan @ exact)
              / (np.linalg.norm(fan, axis=1) * np.linalg.norm(exact)))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _polar(fan, exact, side):
    """Exact length ratio and exact angle, laid into a plane. Not a projection.

    Radius and angle are the real D-dimensional quantities. Only the sign, which
    side of the plane an arrow falls on, is arbitrary, and it is taken from one
    fixed direction perpendicular to the exact gradient.
    """
    scale = np.linalg.norm(exact)
    radius = np.linalg.norm(fan, axis=1) / scale
    angle = np.radians(_angles(fan, exact))
    angle = np.where(fan @ side >= 0, angle, -angle)
    return radius * np.cos(angle), radius * np.sin(angle)


def figure_minibatch_gradient(out_dir):
    from rtbench.targets.blr import blr_superconductor

    target = blr_superconductor(n=N_TRAIN)
    draw_key, fan_key, arrow_key = jax.random.split(jax.random.PRNGKey(SEED), 3)

    # a posterior draw, not the mode: at the mode the exact gradient vanishes and
    # there is no direction left to measure an angle against
    theta = target.ref_sample(draw_key, 1)[0]
    exact = np.asarray(target.grad(theta))

    probe = np.asarray(jax.random.normal(jax.random.PRNGKey(SEED + 1), (target.dim,)))
    unit = exact / np.linalg.norm(exact)
    side = probe - (probe @ unit) * unit
    side /= np.linalg.norm(side)

    keys = jax.random.split(fan_key, len(BATCHES))
    angles = {b: _angles(_fan(target, theta, b, k, N_DRAWS), exact)
              for b, k in zip(BATCHES, keys)}
    mean = np.array([angles[b].mean() for b in BATCHES])
    low = np.array([np.quantile(angles[b], 0.1) for b in BATCHES])
    high = np.array([np.quantile(angles[b], 0.9) for b in BATCHES])

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.2, 4.05), gridspec_kw={"width_ratios": [1.0, 1.5]})

    # Match the introduction panel's axis geometry without touching this figure's
    # local palette: quiet 0.9-pt spines, outward four-point ticks, horizontal grid.
    for axis in (left, right):
        for spine in axis.spines.values():
            spine.set_linewidth(0.9)
        axis.tick_params(axis="both", which="major", direction="out",
                         length=4.0, width=0.9)

    # ---------------------------------------------------- one batch size drawn out
    fan = _fan(target, theta, DRAWN_OUT, arrow_key, 42)
    x, y = _polar(fan, exact, side)
    reach = float(np.quantile(np.hypot(x, y), 0.86)) * 1.03
    ring = np.linspace(0, 2 * np.pi, 240)
    left.plot(np.cos(ring), np.sin(ring), ls=(0, (5, 4)), color=MUTED, linewidth=1.6)
    # upper left is the sparse side: past 90 degrees there are fewer arrows
    left.annotate("exact gradient", xy=(0.44, 0.515), xytext=(0.02, 0.78),
                  xycoords="axes fraction", textcoords="axes fraction",
                  fontsize=ANNOTATION_SIZE, color=INK, va="bottom",
                  arrowprops=dict(arrowstyle="-", color=INK, linewidth=1.3,
                                  shrinkA=3, shrinkB=3))
    for xi, yi in zip(x, y):
        left.annotate("", xy=(xi, yi), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color=NOISE_CORAL,
                                      alpha=0.24,
                                      linewidth=1.55, shrinkA=0, shrinkB=0))
    left.annotate("", xy=(1.0, 0.0), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color=REFERENCE_NAVY, linewidth=3.6,
                                  shrinkA=0, shrinkB=0, mutation_scale=26))
    left.plot(0, 0, "o", color=REFERENCE_NAVY, markersize=7, zorder=6)
    left.set_xlim(-reach, reach)
    left.set_ylim(-reach, reach)
    left.set_aspect("equal")
    left.set_xticks([])
    left.set_yticks([])
    for spine in ("left", "bottom"):
        left.spines[spine].set_visible(False)
    left.set_xlabel(f"$B={DRAWN_OUT}$; mean error "
                    f"{angles[DRAWN_OUT].mean():.0f}$^\\circ$",
                    fontsize=CAPTION_SIZE, labelpad=12)

    # ------------------------------------------------------- angle against batch
    right.axhline(90.0, color=INK, ls=(0, (6, 4)), linewidth=2.0, zorder=2)
    right.text(BATCHES[-1], 92.0, "uninformative direction",
               fontsize=ANNOTATION_SIZE, color=INK, va="bottom", ha="right")
    right.fill_between(BATCHES, low, high, color=NOISE_CORAL, alpha=0.14, zorder=3)
    right.plot(BATCHES, mean, "-o", color=MEAN_CORAL, linewidth=3.0, markersize=9,
               zorder=4)
    marked = mean[BATCHES.index(DRAWN_OUT)]
    right.plot([DRAWN_OUT], [marked], "o", color=REFERENCE_NAVY,
               markersize=12, zorder=5)
    right.annotate("shown at left", xy=(DRAWN_OUT, marked), xytext=(-10, -8),
                   textcoords="offset points", ha="right", va="top",
                   fontsize=ANNOTATION_SIZE, color=INK)

    right.set_xscale("log", base=2)
    right.set_xticks([16, 64, 256, 1024, 4096, N_TRAIN])
    right.set_xticklabels(["16", "64", "256", "1024", "4096", "full batch"])
    right.minorticks_off()
    right.set_ylim(0, 126)
    right.set_yticks([0, 30, 60, 90, 120])
    right.set_yticklabels(["0$^\\circ$", "30$^\\circ$", "60$^\\circ$", "90$^\\circ$",
                           "120$^\\circ$"])
    right.set_xlabel("batch size $B$")
    right.set_ylabel("angle to the\nexact gradient", labelpad=8)
    right.grid(False)

    figure.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.16,
                           wspace=0.36)
    # Use one vertical frame for both panels. The left fan keeps equal aspect
    # within that frame, while the right chart no longer reaches the crop edge.
    left_box = left.get_position(original=True)
    right_box = right.get_position(original=True)
    panel_bottom, panel_height = 0.16, 0.78
    left.set_position([left_box.x0, panel_bottom, left_box.width, panel_height],
                      which="both")
    right.set_position([right_box.x0, panel_bottom, right_box.width, panel_height],
                       which="both")
    save(figure, out_dir, "minibatch_gradient")


if __name__ == "__main__":
    import sources

    use_poster_style()
    figure_minibatch_gradient(sources.OUT_DIR)
