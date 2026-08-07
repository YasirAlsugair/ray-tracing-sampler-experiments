"""exp7 result figures for the SUDS poster. Standalone, repo-resident.

FOR CHUXUAN: everything this reads is committed in this repo, so after
cloning it you only need a Python with numpy, scipy, torch, and
matplotlib (any recent versions; CPU is fine, a GPU is not needed).
Run from the repo root:

    python experiments/exp7_poster_figures.py

The seven vector PDFs land in poster_figures/ at the top of the repo,
every number recomputed from the committed packs (nothing typed in):

    exp7_ladder.pdf       the overconfidence ladder (z std per claim)
    exp7_structure.pdf    per-group calibration: MAP / ensemble / SGHMC / RTS
    exp7_temperature.pdf  SGHMC kinetic temperature vs the true T = 1
    exp7_tails.pdf        tail stars under each likelihood's own yardstick
    exp7_predictive.pdf   test RMSE + test NLL across the five methods
    exp7_mixing.pdf       autocorrelation times, fast vs slowest observable
    exp7_ood.pdf          OOD flagging at the 11 percent budget (chain of
                          record; the notebook's section 10 used the older
                          gated chain, which flagged far less)

Colors and font sizes follow the poster preamble (RTGOLD etc.); edit the
constants below to restyle. Style: softgray spines/ticks, ink labels,
top/right spines off, tight vector PDFs, in-figure strings sized to read
>= 24 pt at the printed 0.82\linewidth width.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "poster_figures"
FIGS.mkdir(exist_ok=True)
TAB = ROOT / "results" / "tables"
sys.path.insert(0, str(ROOT / "experiments"))
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

# ---- P: predictive performance --------------------------------------------
# Proper scoring: mean negative log predictive density on the held-out
# stars (lower is better), mixture over members for the sampled methods.
from scipy.special import logsumexp

DARKGOLD = "#8A5F06"
LOG2PI = np.log(2 * np.pi)


def gauss_mix_nll(mus_m, var_m):
    """Mean NLL of a mixture of Gaussians over members. var_m includes err."""
    lp = (-0.5 * (y[None, :] - mus_m) ** 2 / var_m
          - 0.5 * np.log(var_m) - 0.5 * LOG2PI)
    return float(-np.mean(logsumexp(lp, axis=0) - np.log(len(mus_m))))


mm_var = te**2 + mm["sig"] ** 2
nll_map = float(np.mean(0.5 * (y - mm["mu"]) ** 2 / mm_var
                        + 0.5 * np.log(mm_var) + 0.5 * LOG2PI))
nll_en = gauss_mix_nll(en["mus"], te[None, :] ** 2 + en["sig2s"])
nll_sg = gauss_mix_nll(mus_sg, te[None, :] ** 2 + sig2_sg)
nll_ch = gauss_mix_nll(mus_h, te[None, :] ** 2 + sig2_h)
lp_t = np.stack([sstats.t.logpdf(y, df=nu_m[k], loc=mus_t[k],
                                 scale=np.sqrt(te**2 + sigs_t[k]**2))
                 for k in range(len(mus_t))])
nll_t = float(-np.mean(logsumexp(lp_t, axis=0) - np.log(len(mus_t))))

perf = [
    ("optimizer (MAP)", float(mm["rmse"]), nll_map, SOFTGRAY),
    ("deep ensemble", float(np.sqrt(((en["mus"].mean(0) - y)**2).mean())),
     nll_en, RTBLUE),
    ("SGHMC (tuned)", float(np.sqrt(((mus_sg.mean(0) - y)**2).mean())),
     nll_sg, HMCRED),
    ("RTS, Gaussian", float(np.sqrt(((mu_ch - y)**2).mean())), nll_ch, RTGOLD),
    ("RTS, Student-t", float(np.sqrt(((mus_t.mean(0) - y)**2).mean())),
     nll_t, DARKGOLD),
]
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.4))
ypp = np.arange(len(perf))[::-1]
spans = []
for col_i in (1, 2):
    vs = [r[col_i] for r in perf]
    pad = (max(vs) - min(vs)) * 0.18
    spans.append((min(vs) - pad, max(vs) + pad * 2.2))
for ax, col_i, lab, (lo, hi) in ((axes[0], 1, "test RMSE", spans[0]),
                                 (axes[1], 2, "test NLL (lower is better)",
                                  spans[1])):
    vals = [r[col_i] for r in perf]
    ax.barh(ypp, np.array(vals) - lo, left=lo, height=0.58,
            color=[r[3] for r in perf])
    for yp, v in zip(ypp, vals):
        ax.text(v + (hi - lo) * 0.02, yp, f"{v:.4f}" if col_i == 1
                else f"{v:.3f}", va="center", fontsize=F_TICK - 2)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(lab)
axes[0].set_yticks(ypp, [r[0] for r in perf], fontsize=F_TICK - 1)
axes[1].set_yticks([])
fig.tight_layout()
save(fig, "exp7_predictive.pdf")
for name, rmse, nll, _ in perf:
    print(f"   {name:>16}: RMSE {rmse:.4f}  NLL {nll:.3f}")

# ---- M: mixing, integrated autocorrelation of the exact misfit -------------
def iat_steps(series, cadence=250):
    """Sokal-windowed integrated autocorrelation time, in sampler steps."""
    x = np.asarray(series, float)
    x = x - x.mean()
    n = len(x)
    acf = np.correlate(x, x, "full")[n - 1:] / (x.var() * n)
    tau = 1.0
    for w in range(1, n // 3):
        tau = 1.0 + 2.0 * np.sum(acf[1:w + 1])
        if w >= 5 * tau:
            break
    return tau * cadence, acf


# Stationary stretches only: the RT chain's batch-4096 span before the
# NaN episode; SGHMC after its burn-in descent; the t chain's final half.
# The fast observable (misfit) rewards jitter, so the honest mixing
# number is the SLOWEST observable: the weighted norm. The RT taus there
# are window-limited estimates (span only ~3 tau); SGHMC's weighted norm
# still trends at 2M steps (drift 3.4x scatter), so its slow-direction
# mixing time is a hard lower bound, not an estimate.
tpk = np.load(TAB / "exp7t_pack_final.npz")
rows_m = [
    ("RTS, Gaussian", hp["misfit"][48_000:62_800],
     hp["wnorm"][48_000:62_800], RTGOLD, None),
    ("RTS, Student-t", tpk["misfit"][len(tpk["misfit"]) // 2:],
     tpk["wnorm"][len(tpk["wnorm"]) // 2:], DARKGOLD, None),
    ("SGHMC (tuned)", cold["misfit"][2_000:], None, HMCRED, 2e6),
]
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.2))
ymm = np.arange(len(rows_m))[::-1]
for yp, (name, fast, slow, col, bound) in zip(ymm, rows_m):
    tf, _ = iat_steps(fast)
    axes[0].barh(yp, tf, height=0.55, color=col)
    axes[0].text(tf * 1.25, yp, f"{tf/1e3:.1f}k" if tf < 1e4
                 else f"{tf/1e3:.0f}k", va="center", fontsize=F_TICK - 2)
    if slow is not None:
        ts, _ = iat_steps(slow)
        axes[1].barh(yp, ts, height=0.55, color=col)
        axes[1].text(ts * 1.22, yp, f"~{ts/1e6:.1f}M", va="center",
                     fontsize=F_TICK - 2)
    else:
        axes[1].barh(yp, bound, height=0.55, color=col, alpha=0.45,
                     hatch="//", edgecolor=col)
        axes[1].text(bound * 1.22, yp, "> 2M, never\nequilibrated",
                     va="center", fontsize=F_TICK - 3)
for ax, lab in ((axes[0], "misfit (fast)"),
                (axes[1], "weighted norm (slowest)")):
    ax.set_xscale("log")
    ax.set_title(lab, fontsize=F_LABEL - 1)
axes[0].set_xlim(2e2, 3e5)
axes[1].set_xlim(2e5, 6e7)
axes[0].set_yticks(ymm, [r[0] for r in rows_m], fontsize=F_TICK - 1)
axes[1].set_yticks([])
fig.suptitle("autocorrelation time, in gradient steps (same batch for all)",
             fontsize=F_TICK, y=1.02)
fig.tight_layout()
save(fig, "exp7_mixing.pdf")

# ---- O: out-of-distribution flagging --------------------------------------
# The exp6 fake-images question with real astronomy: the pristine cut
# rejected 376k stars, and three classes of them are genuine OOD inputs.
# The alarm is member disagreement tau, thresholded so it flags 11 percent
# of pristine held-out stars (the exp6 budget); each OOD group is then
# measured against that frozen threshold. Computed on the CHAIN OF RECORD
# (the per-star hetero chain); the notebook's section 10 ran the same test
# on the older gated chain.
ood = np.load(TAB / "exp7_ood_payload.npz")
tau_test = mus_h.std(0)
thresh = np.quantile(tau_test, 0.89)

ood_rows = [("pristine test", float((tau_test > thresh).mean()), SOFTGRAY)]
with torch.no_grad():
    for gname in ("dwarfs", "hot", "flagged"):
        Xo = torch.tensor(ood[f"Xs_{gname}"], dtype=torch.float32,
                          device=H.DEV)
        mus_o = []
        for st in hp["members"]:
            H.load_flat(hm, st)
            mu, _ = hm.mu_r(Xo)
            mus_o.append(mu.cpu().numpy())
        tau_o = np.array(mus_o).std(0)
        ood_rows.append((gname, float((tau_o > thresh).mean()), RTGOLD))

fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
yoo = np.arange(len(ood_rows))[::-1]
for yp, (name, frac, col) in zip(yoo, ood_rows):
    ax.barh(yp, frac, height=0.58, color=col)
    ax.text(frac + 0.015, yp, f"{frac:.2f}", va="center", fontsize=F_TICK)
ax.axvline(0.11, color=INK, ls=":", lw=1.2)
ax.set_yticks(yoo, [r[0] for r in ood_rows])
ax.set_xlabel("fraction flagged at the 11 percent budget")
ax.set_xlim(0, 1.0)
save(fig, "exp7_ood.pdf")
print("   OOD flagged:", ", ".join(f"{n} {f:.2f}" for n, f, _ in ood_rows))
