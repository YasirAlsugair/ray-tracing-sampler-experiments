"""exp7 result figures for the SUDS poster (poster_v2 style contract).

Builds four vector PDFs into artifacts/figures/, every number recomputed
from the committed packs in ../empirical/results/tables (nothing typed in):

    L  exp7_ladder.pdf       the overconfidence ladder (z std per claim)
    S  exp7_structure.pdf    per-group calibration: MAP / ensemble / chain
    T  exp7_temperature.pdf  SGHMC kinetic temperature vs the true T = 1
    Q  exp7_tails.pdf        tail stars under each likelihood's own yardstick

Conventions mirrored from build_figures.py: poster preamble colors, softgray
spines/ticks, ink labels, top/right spines off, tight vector PDFs, in-figure
strings sized to read >= 24 pt at the printed 0.82\linewidth width.

Run from the repo root's poster/ directory:  python build_exp7_figures.py
(needs the repo venv: torch, numpy, scipy, matplotlib)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "artifacts" / "figures"
TAB = ROOT.parent / "results" / "tables"
sys.path.insert(0, str(ROOT.parent / "experiments"))
import exp7_gaia_hetero as H                                  # noqa: E402
import exp7_gaia_t as T                                       # noqa: E402

RTGOLD = "#D99A1B"
RTBLUE = "#1F6FB4"
HMCRED = "#D1495B"
SOFTGRAY = "#8A8D99"
INK = "#20222B"

FIG_W = 7.4
F_TICK, F_LABEL = 15, 16
plt.rcParams.update({
    "font.size": F_TICK, "axes.labelsize": F_LABEL,
    "axes.edgecolor": SOFTGRAY, "xtick.color": SOFTGRAY,
    "ytick.color": SOFTGRAY, "axes.labelcolor": INK,
    "xtick.labelsize": F_TICK, "ytick.labelsize": F_TICK,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "legend.fontsize": F_TICK - 1,
})
EDGES = [0.0, 0.004, 0.006, 0.01, 0.02]


def binstds(z, te):
    return [z[(te >= lo) & (te < hi)].std()
            for lo, hi in zip(EDGES[:-1], EDGES[1:])]


def save(fig, name):
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / name)


# ---- shared data: test split, hetero members, per-model predictions -------
p = np.load(TAB / "exp7_analysis_pack.npz")
y, te = p["test_y"], p["test_yerr"]
sp = np.load(TAB / "exp7_gaia_spectra_payload.npz")
Xs = torch.tensor((sp["coeffs"] - p["norm_mu"]) / p["norm_sd"],
                  dtype=torch.float32, device=H.DEV)

hp = np.load(TAB / "exp7h_pack3.npz")
hm = H.make_model()
mus_h, sig2_h = [], []
with torch.no_grad():
    for st in hp["members"]:
        H.load_flat(hm, st)
        mu, r = hm.mu_r(Xs)
        mus_h.append(mu.cpu().numpy())
        sig2_h.append(np.exp(2.0 * (H.LNS0 + r.cpu().numpy())))
mus_h, sig2_h = np.array(mus_h), np.array(sig2_h)
mu_ch = mus_h.mean(0)
z_ch = (y - mu_ch) / np.sqrt(te**2 + sig2_h.mean(0) + mus_h.var(0))

g = p["member_preds_gated5e6"]
mu10, tau10 = g.mean(0), g.std(0)
s_ = p["member_preds_scatter"]
s_med = float(np.median(p["member_s_scatter"]))
z_scat = (y - s_.mean(0)) / np.sqrt(s_.std(0)**2 + te**2 + s_med**2)

# ---- L: the overconfidence ladder -----------------------------------------
rows = [
    ("err only", ((y - mu10) / te).std(), None, SOFTGRAY),
    (r"err + $\tau$", ((y - mu10) / np.sqrt(tau10**2 + te**2)).std(),
     None, SOFTGRAY),
    ("+ global $s$", z_scat.std(),
     (min(binstds(z_scat, te)), max(binstds(z_scat, te))), RTGOLD),
    (r"+ per-star $\sigma(x)$", z_ch.std(),
     (min(binstds(z_ch, te)), max(binstds(z_ch, te))), RTGOLD),
]
fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
ypos = np.arange(len(rows))[::-1]
for yp, (name, v, rng, col) in zip(ypos, rows):
    ax.barh(yp, v, height=0.58, color=col)
    ax.text((rng[1] if rng else v) + 0.12, yp, f"{v:.2f}",
            va="center", fontsize=F_TICK)
    if rng:
        ax.plot(rng, [yp, yp], color=INK, lw=1.6)
        for xc in rng:
            ax.plot([xc, xc], [yp - 0.14, yp + 0.14], color=INK, lw=1.6)
ax.axvline(1.0, color=INK, ls=":", lw=1.2)
ax.set_yticks(ypos, [r[0] for r in rows])
ax.set_xlabel("z std, held-out stars")
ax.set_xlim(0, 7.4)
save(fig, "exp7_ladder.pdf")

# ---- S: per-group structure ------------------------------------------------
mm = np.load(TAB / "exp7_map.npz")
en = np.load(TAB / "exp7_ensemble.npz")
z_en = (y - en["mus"].mean(0)) / np.sqrt(
    te**2 + en["sig2s"].mean(0) + en["mus"].var(0))

sg_cold = np.load(TAB / "exp7sg_fr3000_pack.npz")
mus_sg, sig2_sg = [], []
with torch.no_grad():
    for st in sg_cold["members"]:
        H.load_flat(hm, st)
        mu, r = hm.mu_r(Xs)
        mus_sg.append(mu.cpu().numpy())
        sig2_sg.append(np.exp(2.0 * (H.LNS0 + r.cpu().numpy())))
mus_sg, sig2_sg = np.array(mus_sg), np.array(sig2_sg)
z_sg = (y - mus_sg.mean(0)) / np.sqrt(
    te**2 + sig2_sg.mean(0) + mus_sg.var(0))

labels = [f"[{lo:g}, {hi:g})" for lo, hi in zip(EDGES[:-1], EDGES[1:])]
fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
xpos = np.arange(len(labels))
for k, (name, z, col) in enumerate((("optimizer (MAP)", mm["z"], SOFTGRAY),
                                    ("deep ensemble", z_en, RTBLUE),
                                    ("SGHMC (tuned)", z_sg, HMCRED),
                                    ("ray tracing (RTS)", z_ch, RTGOLD))):
    ax.bar(xpos + (k - 1.5) * 0.21, binstds(z, te), 0.21, color=col,
           label=name)
ax.axhline(1.0, color=INK, ls=":", lw=1.2)
ax.set_xticks(xpos, labels, fontsize=F_TICK - 1)
ax.set_xlabel("label error bin")
ax.set_ylabel("z std")
ax.set_ylim(0, 1.45)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
          frameon=False, columnspacing=1.2, handlelength=1.4)
save(fig, "exp7_structure.pdf")

# ---- T: the SGHMC thermometer ---------------------------------------------
hot = np.load(TAB / "exp7sg_fr30_pack.npz")
cold = np.load(TAB / "exp7sg_fr3000_pack.npz")
fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
ax.plot(hot["steps"] / 1e6, hot["temp"], color=HMCRED, lw=1.6,
        label="SGHMC, low friction")
ax.plot(cold["steps"] / 1e6, cold["temp"], color=RTBLUE, lw=1.6,
        label="SGHMC, tuned friction")
ax.axhline(1.0, color=INK, ls=":", lw=1.4)
ax.set_yscale("log")
ax.set_ylim(0.78, None)
ax.text(1.0, 0.955, "true posterior temperature", fontsize=F_TICK - 1,
        color=INK, va="top")
ax.set_xlabel("step (millions)")
ax.set_ylabel(r"kinetic temperature $\langle v^2 \rangle$")
ax.legend(loc="upper right", frameon=False)
save(fig, "exp7_temperature.pdf")
q = len(cold["temp"]) * 3 // 4
print(f"   tuned-arm final-quarter temp {cold['temp'][q:].mean():.2f}, "
      f"heated arm {hot['temp'][len(hot['temp'])*3//4:].mean():.2f}")

# ---- Q: tail stars under each likelihood's own yardstick -------------------
tp = np.load(TAB / "exp7t_pack_final.npz")
nu_m = 2.0 + np.exp(1.6 + tp["members"][:, 0])
tmodel = T.make_model()
mus_t, sigs_t = [], []
with torch.no_grad():
    for st in tp["members"]:
        T.load_flat(tmodel, st)
        mu, r = tmodel.mu_r(Xs)
        mus_t.append(mu.cpu().numpy())
        sigs_t.append(np.exp(-3.0 + r.cpu().numpy()))
mus_t, sigs_t = np.array(mus_t), np.array(sigs_t)
pit = np.zeros(len(y))
for m in range(len(mus_t)):
    pit += sstats.t.cdf((y - mus_t[m]) / np.sqrt(te**2 + sigs_t[m]**2),
                        df=nu_m[m])
pit /= len(mus_t)
p4 = 2.0 * sstats.norm.sf(4.0)
n_t = int((2.0 * np.minimum(pit, 1.0 - pit) < p4).sum())
n_gauss = int((np.abs(z_ch) > 4).sum())
chance = p4 * len(y)

fig, ax = plt.subplots(figsize=(FIG_W * 0.62, 3.6))
names = ["Gaussian", "Student-t", "chance"]
vals = [n_gauss, n_t, chance]
cols = [SOFTGRAY, RTGOLD, INK]
bars = ax.bar(names, vals, 0.6, color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.12,
            f"{v:.0f}" if v >= 3 else f"{v:.1f}",
            ha="center", fontsize=F_LABEL)
ax.set_yscale("log")
ax.set_ylim(1, 300)
ax.set_ylabel(r"stars past the $|z| = 4$ tail")
save(fig, "exp7_tails.pdf")
print(f"   tails: Gaussian {n_gauss}, t {n_t} (nu median "
      f"{np.median(nu_m):.2f}), chance {chance:.1f}")
