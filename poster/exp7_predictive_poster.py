"""Poster panels: the 1-star posterior cloud, annotated for distance reading.

exp7_cloud_poster.pdf: two stars on IDENTICAL axes, the 50 t-chain members'
(mu_m, sigma_m) per star. On shared axes the typical star's cloud collapses
to a speck and the disagreement star's cloud fills the frame: dots close
together = the posterior agrees = trust the prediction. Dashed circles and
two-word labels carry that takeaway; X = the Adam MAP fit (one point, no
cloud).

exp7_marginal_poster.pdf: the SAME two stars seen in y, one panel each:
MAP plug-in (gray dashed) vs the marginalized Student-t chain (gold, thin
curves = 10 of the 50 members). On the typical star the curves agree; on
the disagreement star the mass MAP cannot see is circled.

Star indices are the deterministic picks of
experiments/exp7_predictive_figures.py (typical = median mu-cloud spread at
mid-quartile yerr; disagreement = max spread / mean total scale). Numbers
certified there: test-set NLL MAP -1.733 vs t marginal -1.973.

Run from poster/:  ../.venv/bin/python exp7_predictive_poster.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as tdist, norm
from scipy.special import logsumexp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parent.parent          # the repo root
TAB = ROOT / "results" / "tables"
OUT = Path(__file__).resolve().parent / "artifacts" / "figures"
sys.path.insert(0, str(ROOT / "experiments"))
import exp7_gaia_t as T                                       # noqa: E402

RTGOLD, RTBLUE, SOFTGRAY, INK = "#D99A1B", "#1F6FB4", "#8A8D99", "#222222"
I_TYP, I_DIS = 20562, 3717      # picks from exp7_predictive_figures.py
F = 20                          # >= 24 pt printed at the 0.66 slot

plt.rcParams.update({
    "font.size": F, "axes.titlesize": F + 2,
    "axes.edgecolor": SOFTGRAY, "xtick.color": SOFTGRAY,
    "ytick.color": SOFTGRAY, "axes.labelcolor": INK,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

d = np.load(TAB / "exp7_gaia_pristine.npz")
ty, terr = d["test_y"], d["test_yerr"]
Xt = torch.tensor(d["test_Xs"][[I_TYP, I_DIS]], dtype=torch.float32,
                  device=T.DEV)

members = np.load(TAB / "exp7t_pack_final.npz")["members"]
model = T.make_model()
MU, SIG, NU = [], [], []
with torch.no_grad():
    for m in members:
        T.load_flat(model, m)
        mu, r = model.mu_r(Xt)
        MU.append(mu.cpu().numpy())
        SIG.append(np.exp(T.LNS0 + r.cpu().numpy()))
        NU.append(float(model.nu().cpu()))
MU, SIG, NU = np.array(MU), np.array(SIG), np.array(NU)

mp = np.load(TAB / "exp7_map.npz")
mu_map, sig_map = mp["mu"][[I_TYP, I_DIS]], mp["sig"][[I_TYP, I_DIS]]

# ---- panel pair: the cloud, shared axes ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.55), constrained_layout=True,
                         sharex=True, sharey=True)
titles = ["a typical star", "a star the model disagrees on"]
for k, (ax, title) in enumerate(zip(axes, titles)):
    ax.scatter(MU[:, k], SIG[:, k], s=34, color=RTGOLD, alpha=0.85,
               linewidths=0, zorder=3)
    ax.scatter([mu_map[k]], [sig_map[k]], marker="X", s=170, color=INK,
               zorder=4)
    ax.set_title(title)
    ax.set_xlabel(r"predicted value $\mu_m$")
axes[0].set_ylabel(r"believed scatter $\sigma_m$")
axes[0].set_xlim(-0.08, 0.92)
axes[0].set_ylim(0.0, 0.108)
# circle only the speck; the panel-2 cloud fills its frame, which IS the point
cx, cy = MU[:, 0].mean(), SIG[:, 0].mean()
axes[0].add_patch(Ellipse((cx, cy), 0.11, 0.024, fill=False, ls="--", lw=2.2,
                          edgecolor=INK, zorder=5))
axes[0].annotate("the 50 draws agree", (cx + 0.02, cy + 0.013),
                 xytext=(0.44, 0.060), fontsize=F, color=INK, ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK, lw=1.4))
axes[0].annotate("MAP", (mu_map[0], sig_map[0]), xytext=(-42, -10),
                 textcoords="offset points", fontsize=F - 1, color=INK,
                 ha="right")
axes[1].text(0.03, 0.99, "the draws disagree:\none fit hides this",
             transform=axes[1].transAxes, va="top", ha="left",
             fontsize=F, color=INK)
fig.savefig(OUT / "exp7_cloud_poster.pdf", bbox_inches="tight",
            transparent=True)
fig.savefig(OUT / "exp7_cloud_poster.png", dpi=150, bbox_inches="tight",
            transparent=True)

# ---- companion: the same two stars, seen in y ------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
for k, (ax, title) in enumerate(zip(axes2, titles)):
    yi, ei = ty[[I_TYP, I_DIS][k]], terr[[I_TYP, I_DIS][k]]
    S = np.sqrt(ei ** 2 + SIG[:, k] ** 2)
    s_map_k = float(np.sqrt(ei ** 2 + sig_map[k] ** 2))
    c = MU[:, k].mean()
    half = 5.0 * max(float(S.max()), s_map_k, abs(mu_map[k] - c))
    grid = np.linspace(c - half, c + half, 900)
    lp = tdist.logpdf(grid[None, :], df=NU[:, None], loc=MU[:, k][:, None],
                      scale=S[:, None])
    mix = np.exp(logsumexp(lp, axis=0) - np.log(len(NU)))
    map_pdf = norm.pdf(grid, mu_map[k], s_map_k)
    for row in np.exp(lp)[::5]:
        ax.plot(grid, row, color=RTGOLD, lw=1.0, alpha=0.38, zorder=1)
    ax.plot(grid, map_pdf, color=SOFTGRAY, lw=2.4, ls="--", zorder=2)
    ax.plot(grid, mix, color=RTGOLD, lw=3.0, zorder=3)
    ax.axvline(yi, color=INK, lw=1.4, ls=":", zorder=4)
    if k == 0:
        member_peak = float(np.exp(lp)[::5].max())
        ax.set_ylim(0.0, 1.55 * max(member_peak, float(map_pdf.max())))
    else:
        ax.set_ylim(0.0, 1.30 * float(map_pdf.max()))
    ax.set_title(title)
    ax.set_xlabel(r"$y$ (alpha abundance)")
    ax.set_yticks([])
    if k == 1:
        region = (grid > 0.22) & (grid < 0.95)
        hmax = float(mix[region].max())
        ax.add_patch(Ellipse((0.58, 0.75 * hmax), 0.80, 2.0 * hmax,
                             fill=False, ls="--", lw=2.2, edgecolor=INK,
                             zorder=5))
        ax.text(0.74, 2.3 * hmax, "mass MAP\ncannot see", fontsize=F,
                color=INK, ha="center", va="bottom")
axes2[0].set_ylabel("predictive density")
axes2[0].text(0.03, 0.95, "MAP", transform=axes2[0].transAxes, va="top",
              ha="left", fontsize=F, color=SOFTGRAY)
axes2[0].text(0.03, 0.82, "sampled posterior", transform=axes2[0].transAxes,
              va="top", ha="left", fontsize=F, color=RTGOLD)
axes2[0].text(0.03, 0.69, "dotted: catalog value", transform=axes2[0].transAxes,
              va="top", ha="left", fontsize=F - 3, color=INK)
fig2.savefig(OUT / "exp7_marginal_poster.pdf", bbox_inches="tight",
             transparent=True)
fig2.savefig(OUT / "exp7_marginal_poster.png", dpi=150, bbox_inches="tight",
             transparent=True)

print("saved exp7_cloud_poster + exp7_marginal_poster to", OUT)
print(f"check: nu range [{NU.min():.2f}, {NU.max():.2f}], "
      f"typical spread {MU[:, 0].std():.4f}, "
      f"disagreement spread {MU[:, 1].std():.4f}")
