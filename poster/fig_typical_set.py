"""Self-contained builder for the typical-set figure.

Produces exactly one file:  artifacts/figures/typical_set.pdf  (overwritten;
suds_poster.tex already includes this filename).

Styled like a statistics-journal figure (STIX serif, thin lines, frameless
legend, panel tags, minimal annotation), not a poster infographic:
  (a) density x volume = mass at D = 2, all three curves rescaled to max 1,
      where the product is visible at one scale and checkable by eye.
  (b) the radial mass of the SAME standard Gaussian at D = 1, 10, 100,
      drawn as true chi_D densities (each has area 1) on a real y-axis,
      so the mass visibly walks out to r ~ sqrt(D) as D grows.

Run:  /Users/yasiralsugair/UofT/empirical/.venv/bin/python fig_typical_set.py

Deterministic (no sampling). numpy + scipy.stats + matplotlib only; mathtext,
no LaTeX. Colors follow the poster palette (blue = density, gold = volume,
navy = mass).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent            # .../UofT/poster_v2
OUT = ROOT / "artifacts" / "figures" / "typical_set.pdf"

# colors mirrored from build_figures.py / the suds_poster.tex preamble
DENS = "#1F6FB4"      # density
VOL = "#D99A1B"       # volume
NAVY = "#002A5C"      # mass / typical set
INK = "#20222B"       # text, spines

F_BASE = 12
F_LABEL = 12.5

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": F_BASE,
    "axes.edgecolor": INK,
    "axes.linewidth": 1.0,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "text.color": INK,
})

D = 100
SQD = float(np.sqrt(D))

fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.3, 3.9), layout="constrained")
fig.get_layout_engine().set(h_pad=0.10, w_pad=0.08)

XLAB = r"$r = \|\theta - \theta_{\mathrm{mode}}\|$"

# ---------------------- (a): density x volume = mass at D = 2, checkable by eye

R1 = 4.2
r2 = np.linspace(0.0, R1, 800)
dens2 = np.exp(-0.5 * r2 ** 2)                 # max 1 at the mode
vol2 = r2 / R1                                 # r^(D-1) = r at D = 2, max 1
mass2 = r2 * np.exp(-0.5 * r2 ** 2)
mass2 = mass2 / mass2.max()                    # the product, max 1

axa.plot(r2, dens2, color=DENS, lw=1.7, label=r"density $p(r)$")
axa.plot(r2, vol2, color=VOL, lw=1.7, ls="--", label=r"volume $r^{\,D-1}$")
axa.plot(r2, mass2, color=NAVY, lw=2.0, label=r"mass $p(r)\,r^{\,D-1}$")
axa.fill_between(r2, mass2, 0.0, color=NAVY, alpha=0.12, lw=0)
axa.legend(frameon=False, loc="upper right", fontsize=F_BASE, handlelength=1.7)

axa.set_title(r"(a)  $D = 2$", loc="left", fontsize=F_LABEL)
axa.set_xlim(0.0, R1)
axa.set_ylim(0.0, 1.40)
axa.set_xticks([0, 1, 2, 3, 4])
axa.set_yticks([])
axa.spines["left"].set_visible(False)
axa.set_xlabel(XLAB, fontsize=F_LABEL)

# ------------------- (b): the same Gaussian's radial mass at D = 1, 10, 100

R_MAX = SQD + 3.5
r = np.linspace(0.0, R_MAX, 1600)

# true chi_D densities: every curve has area 1, nothing rescaled, so the bumps
# stay comparable and the mass simply walks out to r ~ sqrt(D)
for Dk, lx, ly, ha in ((1, 0.65, 0.82, "left"), (10, 3.0, 0.67, "center"),
                       (100, SQD, 0.65, "center")):
    m = chi.pdf(r, Dk)
    axb.fill_between(r, m, 0.0, color=NAVY, alpha=0.10, lw=0)
    axb.plot(r, m, color=NAVY, lw=1.7)
    axb.text(lx, ly, rf"$D = {Dk}$", color=INK, ha=ha, va="center",
             fontsize=F_BASE)

axb.set_title("(b)  the same Gaussian in higher dimensions", loc="left",
              fontsize=F_LABEL)
axb.set_xlim(0.0, R_MAX)
axb.set_ylim(0.0, 0.9)
axb.set_xticks([0.0, np.sqrt(10.0), SQD],
               labels=["0", r"$\sqrt{10}$", r"$\sqrt{100}$"])
axb.set_yticks([0.0, 0.4, 0.8])
axb.set_ylabel(r"mass $p(r)\,r^{\,D-1}$", fontsize=F_LABEL)
axb.set_xlabel(XLAB, fontsize=F_LABEL)

# ------------------------------------------------- save, then verify the output

fig.savefig(OUT)

if not (OUT.exists() and OUT.stat().st_size > 0):
    sys.exit(f"FAILED: {OUT} was not written")

# no text may be clipped at the figure edges, and no two texts may collide
renderer = fig.canvas.get_renderer()
fig_box = fig.get_window_extent(renderer)
texts = [t for t in fig.findobj(Text)
         if t.get_visible() and t.get_text().strip()]
problems = []
for t in texts:
    bb = t.get_window_extent(renderer)
    if (bb.x0 < fig_box.x0 - 0.5 or bb.y0 < fig_box.y0 - 0.5 or
            bb.x1 > fig_box.x1 + 0.5 or bb.y1 > fig_box.y1 + 0.5):
        problems.append(f"clipped at figure edge: {t.get_text()!r}")
# overlap check for hand-placed texts only: matplotlib lays out axis labels,
# titles, and legend internals itself (and reports stale extents for rotated
# ylabels), so only free ax.text artists can collide through my doing
for ax in (axa, axb):
    frees = [t for t in ax.texts if t.get_text().strip()]
    boxes = [t.get_window_extent(renderer) for t in frees]
    leg = ax.get_legend()
    if leg is not None:
        frees.append(leg)
        boxes.append(leg.get_window_extent(renderer))
    for i, (ta, ba) in enumerate(zip(frees, boxes)):
        for tb, bb in zip(frees[i + 1:], boxes[i + 1:]):
            if ba.overlaps(bb):
                problems.append(f"overlap: {ta} / {tb}")
if problems:
    sys.exit("FAILED:\n  " + "\n  ".join(problems))

print(f"wrote {OUT} ({OUT.stat().st_size} bytes); "
      f"all {len(texts)} texts inside the figure, no overlaps")
