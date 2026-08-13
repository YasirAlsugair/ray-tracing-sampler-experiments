"""Poster-scale matplotlib defaults: read at two metres, not on a laptop."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK = "#20252D"
MUTED = "#7B8190"
FAINT = "#D9DEE7"
# The poster's block-body colour. Figures save transparent and sit on this, so
# anything that used to mask with white must mask with this instead.
PANEL = "#F4F5F8"
DARK_GOLD = "#8A5F06"

# Defaults for builders that do not declare a figure-local palette. The selected
# v6 figures intentionally use local palettes tuned to their own visual grammar.
COLOR = {
    "rt": "#168AAD",
    "hmc": "#B24C83",
    "sgld": "#2e8b57",
    "gold": "#7a7a7a",
    "ensemble": "#2e8b57",
    "dropout": "#c46aa6",
    # The random walk needs its own tone: reusing HMC aubergine would put two different
    # methods in one colour. Magma at 0.48 is dark enough to stay above the ridge field.
    "random_walk": plt.get_cmap("magma")(0.48),
}

# Sequential ramps for fans of trajectories, so a bundle reads as one method.
RAMP = {"rt": "Blues", "random_walk": "Reds"}

# The ridge field, truncated well short of the dark end so every path drawn on it
# stays darker than anything behind it.
FIELD = LinearSegmentedColormap.from_list(
    "ridge_field", plt.get_cmap("Blues")(np.linspace(0.0, 0.45, 256)))

# Two tints of the same blue for the low- and high-index halves of the medium.
MEDIUM_LOW = "#e8eef6"
MEDIUM_HIGH = "#c2d5e9"

LABEL = {
    "rt": "ray tracing",
    "hmc": "stochastic HMC",
    "sgld": "SGLD",
    "gold": "gold posterior (NUTS)",
    "ensemble": "deep ensemble",
    "dropout": "MC dropout",
}


def use_poster_style() -> None:
    plt.rcParams.update({
        # The poster sets sans text and, via \usefonttheme[onlymath]{serif}, Latin
        # Modern math. Computer Modern is that same design, so figure math matches
        # the equations it sits under.
        "mathtext.fontset": "cm",
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "none",
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 17,
        "legend.frameon": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 3.0,
        "lines.markersize": 10,
        "grid.color": FAINT,
        "grid.linewidth": 0.8,
    })


def lift(width=4.6):
    """Panel-coloured outline behind a line, so crossings stay readable at poster distance."""
    import matplotlib.patheffects as path_effects

    return [path_effects.withStroke(linewidth=width, foreground=PANEL)]


def fading_path(axis, path, color, linewidth=1.9, tail_fraction=0.14, zorder=4):
    """Draw a trajectory whose last stretch fades out, so the endpoint looks deliberate."""
    n_tail = max(int(tail_fraction * len(path)), 2)
    head = path[: len(path) - n_tail + 1]
    axis.plot(head[:, 0], head[:, 1], color=color, linewidth=linewidth, alpha=0.95,
              solid_capstyle="round", zorder=zorder)
    tail = path[len(path) - n_tail:]
    cuts = list(range(0, len(tail), max(len(tail) // 10, 1))) + [len(tail) - 1]
    for start, stop in zip(cuts, cuts[1:]):
        fade = start / max(len(tail) - 1, 1)
        axis.plot(tail[start:stop + 1, 0], tail[start:stop + 1, 1], color=color,
                  linewidth=linewidth * (1.0 - 0.45 * fade),
                  alpha=0.95 * (1.0 - fade) ** 1.15, solid_capstyle="round",
                  zorder=zorder)


def save(figure, out_dir, name, *, transparent=True, pad_inches=None) -> None:
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"transparent": transparent}
    if transparent:
        kwargs.update(facecolor="none", edgecolor="none")
    if pad_inches is not None:
        kwargs["pad_inches"] = pad_inches
    for suffix in ("png", "pdf"):
        figure.savefig(out_dir / f"{name}.{suffix}", **kwargs)
    plt.close(figure)
    print(f"  wrote {name}.png / .pdf")
