"""SUDS-poster variant of empirical/experiments/exp7_poster_figures.py.

Style-only fork for suds_poster_v4.tex column 3: FIG_W 9.6, fonts 20/21,
shorter aspects for the structure/members panels, poster palette for the
imposters panel, structure legend replaced by the block's colored text key,
short ylabels that fit the 3 in canvas. ALL NUMBERS IDENTICAL to the canon
script; only style constants and legend/label lines differ.

Run from the repo root's poster/ directory:  python exp7_gaia_panels_suds.py
Then copy exp7_structure/exp7_imposters/exp7_members.pdf from
artifacts/exp7_suds_figs/ into artifacts/figures/.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent          # the repo root
FIGS = Path(__file__).resolve().parent / "artifacts" / "exp7_suds_figs"
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

# Method palette (Okabe-Ito), the single source of truth for
# method-comparison figures; import/copy this dict rather than redefining.
# Gray recedes (non-Bayesian baseline), the two blues are one family
# (approximate-Bayesian baselines, light to dark), vermillion is the only
# warm hue: RTS is our method and must pop.
# SUDS-poster variant: the imposters panel follows the poster palette
# (gold = ray tracing, red = SGHMC, blue = ensemble), matching section S.
COLORS = {
    "point (MAP)": "#8A8D99",
    "deep ensemble": "#1F6FB4",
    "SGHMC (tuned)": "#D1495B",
    "ray tracing (RTS)": "#D99A1B",
}


def _darken(hex_color, factor=0.75):
    """The bar-edge rule: the face color with each RGB channel x factor."""
    import matplotlib.colors as mcolors
    return tuple(c * factor for c in mcolors.to_rgb(hex_color))

FIG_W = 9.6
F_TICK, F_LABEL = 20, 21
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

labels = [f"[{lo:g},{hi:g})" for lo, hi in zip(EDGES[:-1], EDGES[1:])]
fig, ax = plt.subplots(figsize=(FIG_W, 2.9))
xpos = np.arange(len(labels))
for k, (name, z, col) in enumerate((("point (MAP)", mm["z"], SOFTGRAY),
                                    ("deep ensemble", z_en, RTBLUE),
                                    ("SGHMC (tuned)", z_sg, HMCRED),
                                    ("ray tracing (RTS)", z_ch, RTGOLD))):
    ax.bar(xpos + (k - 1.5) * 0.21, binstds(z, te), 0.21, color=col,
           label=name)
ax.axhline(1.0, color=INK, ls=":", lw=1.2)
ax.set_xticks(xpos, labels, fontsize=F_TICK - 4)
ax.set_xlabel("label error bin")
ax.set_ylabel("z std")
ax.set_ylim(0, 1.45)
# SUDS variant: the four methods are keyed by a colored text line in the
# poster block, shared with the imposters panel below; no in-figure legend.
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

# ---- O: out-of-distribution flagging, old chain vs chain of record --------
# The exp6 fake-images question with real astronomy: three classes of stars
# the pristine cut rejected are genuine OOD inputs. The alarm is member
# disagreement tau, thresholded to flag 11 percent of pristine held-out
# stars (the exp6 budget), each chain against its own threshold. The gated
# first-likelihood chain's spread was collapsed by its over-soft gate; the
# converged per-star chain's spread is genuine, and the same honesty that
# calibrates the labels is what powers the alarm.
import exp7_gaia_driver as D

ood = np.load(TAB / "exp7_ood_payload.npz")
gm = np.load(TAB / "exp7_gated5e6_members.npz")["members"]
old_model = D.make_model()

def group_fracs(members, model, mu_of, tau_test):
    th = np.quantile(tau_test, 0.89)
    fr = {"pristine test": float((tau_test > th).mean())}
    with torch.no_grad():
        for gname in ("dwarfs", "hot", "flagged"):
            Xo = torch.tensor(ood[f"Xs_{gname}"], dtype=torch.float32,
                              device=H.DEV)
            mus_o = []
            for st in members:
                H.load_flat(model, st)
                mus_o.append(mu_of(model, Xo).cpu().numpy())
            fr[gname] = float((np.array(mus_o).std(0) > th).mean())
    return fr

fr_old = group_fracs(gm, old_model,
                     lambda m, X: m(X).squeeze(-1), tau10)
fr_new = group_fracs(hp["members"], hm,
                     lambda m, X: m.mu_r(X)[0], mus_h.std(0))

names = ["pristine test", "dwarfs", "hot", "flagged"]
fig, ax = plt.subplots(figsize=(FIG_W, 3.8))
yoo = np.arange(len(names))[::-1]
for off, fr, col, lab in ((0.19, fr_old, SOFTGRAY,
                           "first chain (spread collapsed by the gate)"),
                          (-0.19, fr_new, RTGOLD,
                           "chain of record (honest spread)")):
    for yp, n in zip(yoo, names):
        ax.barh(yp + off, fr[n], height=0.36, color=col,
                label=lab if yp == yoo[0] else None)
        ax.text(fr[n] + 0.015, yp + off, f"{fr[n]:.2f}", va="center",
                fontsize=F_TICK - 2)
ax.axvline(0.11, color=INK, ls=":", lw=1.2)
ax.set_yticks(yoo, names)
ax.set_xlabel("fraction flagged at the 11 percent budget")
ax.set_xlim(0, 1.12)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False,
          fontsize=F_TICK - 2)
save(fig, "exp7_ood.pdf")
print("   OOD old ", {k: round(v, 2) for k, v in fr_old.items()})
print("   OOD new ", {k: round(v, 2) for k, v in fr_new.items()})

# ---- R: reliability (coverage) diagram ------------------------------------
# The regression version of a reliability diagram: for each nominal central
# probability p, the fraction of held-out stars whose label falls inside the
# method's central p-interval, via each predictive's exact PIT (mixtures
# handled properly, shapes included). Diagonal = honest; below = over-
# confident; above = over-padded.
def gauss_mix_pit(mus_m, var_m):
    return np.mean(sstats.norm.cdf((y[None, :] - mus_m) / np.sqrt(var_m)),
                   axis=0)

pit_map = sstats.norm.cdf((y - mm["mu"]) / np.sqrt(te**2 + mm["sig"]**2))
pit_en = gauss_mix_pit(en["mus"], te[None, :]**2 + en["sig2s"])
pit_sg = gauss_mix_pit(mus_sg, te[None, :]**2 + sig2_sg)
pit_ch = gauss_mix_pit(mus_h, te[None, :]**2 + sig2_h)
pit_tt = pit          # the t-mixture PIT from the tails section

grid = np.linspace(0.05, 0.99, 40)
fig, ax = plt.subplots(figsize=(FIG_W * 0.72, 4.6))
for pit_v, lab, col in ((pit_map, "optimizer (MAP)", SOFTGRAY),
                        (pit_en, "deep ensemble", RTBLUE),
                        (pit_sg, "SGHMC (tuned)", HMCRED),
                        (pit_ch, "RTS, Gaussian", RTGOLD),
                        (pit_tt, "RTS, Student-t", DARKGOLD)):
    cov = [(np.abs(pit_v - 0.5) < p_ / 2).mean() for p_ in grid]
    ax.plot(grid, cov, color=col, lw=2.0, label=lab)
ax.plot([0, 1], [0, 1], color=INK, ls="--", lw=1.0)
ax.set_xlabel("nominal coverage")
ax.set_ylabel("empirical coverage, test set")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.legend(frameon=False, loc="upper left", fontsize=F_TICK - 2)
ax.text(0.62, 0.30, "below the line:\noverconfident", fontsize=F_TICK - 2,
        color=SOFTGRAY)
save(fig, "exp7_reliability.pdf")
for pit_v, lab in ((pit_map, "MAP"), (pit_en, "ensemble"), (pit_sg, "SGHMC"),
                   (pit_ch, "RTS-Gauss"), (pit_tt, "RTS-t")):
    dev = max(abs((np.abs(pit_v - 0.5) < p_ / 2).mean() - p_) for p_ in grid)
    print(f"   {lab:>10}: max |coverage - nominal| = {dev:.3f}")

# ---- C: OOD ROC curves, the literature's currency --------------------------
# AUC-ROC of the tau alarm per OOD group (Izmailov et al. report OOD this
# way), with the 11-percent-budget operating point marked on each curve.
# Chain of record throughout.
tau_test_r = mus_h.std(0)
thresh_r = np.quantile(tau_test_r, 0.89)


def roc(pos, neg):
    scores = np.concatenate([pos, neg])
    is_pos = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores)
    tp = np.cumsum(is_pos[order]) / len(pos)
    fp = np.cumsum(1 - is_pos[order]) / len(neg)
    auc = float(np.trapezoid(tp, fp))
    return fp, tp, auc


ood_taus = {}
with torch.no_grad():
    for gname in ("dwarfs", "hot", "flagged"):
        Xo = torch.tensor(ood[f"Xs_{gname}"], dtype=torch.float32,
                          device=H.DEV)
        mus_o = []
        for st in hp["members"]:
            H.load_flat(hm, st)
            mu, _ = hm.mu_r(Xo)
            mus_o.append(mu.cpu().numpy())
        ood_taus[gname] = np.array(mus_o).std(0)

# SGHMC overlay: its alarm is indistinguishable (AUC within 0.006), the
# honest exhibit that OOD detection tests spread, not sampler quality
tau_test_sg = mus_sg.std(0)
ood_taus_sg = {}
with torch.no_grad():
    for gname in ("dwarfs", "hot", "flagged"):
        Xo = torch.tensor(ood[f"Xs_{gname}"], dtype=torch.float32,
                          device=H.DEV)
        mus_o = []
        for st in sg_cold["members"]:
            H.load_flat(hm, st)
            mus_o.append(hm.mu_r(Xo)[0].cpu().numpy())
        ood_taus_sg[gname] = np.array(mus_o).std(0)

fig, ax = plt.subplots(figsize=(FIG_W * 0.62, 4.2))
for gname, col in (("hot", DARKGOLD), ("dwarfs", RTGOLD),
                   ("flagged", SOFTGRAY)):
    fp, tp, auc = roc(ood_taus[gname], tau_test_r)
    ax.plot(fp, tp, color=col, lw=2.0, label=f"{gname}  AUC {auc:.2f}")
    fp2, tp2, _ = roc(ood_taus_sg[gname], tau_test_sg)
    ax.plot(fp2, tp2, color=col, lw=1.4, ls="--")
    op = float((ood_taus[gname] > thresh_r).mean())
    ax.plot([0.11], [op], "o", color=col, ms=8, mec=INK, mew=0.8)
ax.plot([0, 1], [0, 1], ls=":", color=INK, lw=1.2)
ax.text(0.42, 0.06, "solid: RTS\ndashed: SGHMC (same alarm)",
        fontsize=F_TICK - 2, color=INK)
ax.set_xlabel("false alarms (pristine flagged)")
ax.set_ylabel("caught (OOD flagged)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc="center right", frameon=False)
save(fig, "exp7_ood_roc.pdf")

# ---- K: test NLL vs number of members averaged -----------------------------
# The exp6-style curve: how much the predictive improves as members are
# averaged. Per-member log-densities are precomputed once; a K-member
# mixture is then a logsumexp over any subset. Bands: std over 25 random
# subsets. The MAP line is the memberless floor.
def member_logpdfs_gauss(mus_m, var_m):
    return (-0.5 * (y[None, :] - mus_m) ** 2 / var_m
            - 0.5 * np.log(var_m) - 0.5 * LOG2PI)


lp_sets = [
    ("RTS, Gaussian", member_logpdfs_gauss(mus_h, te[None, :]**2 + sig2_h),
     RTGOLD),
    ("RTS, Student-t", np.stack([
        sstats.t.logpdf(y, df=nu_m[k], loc=mus_t[k],
                        scale=np.sqrt(te**2 + sigs_t[k]**2))
        for k in range(len(mus_t))]), DARKGOLD),
    ("SGHMC (tuned)", member_logpdfs_gauss(mus_sg, te[None, :]**2 + sig2_sg),
     HMCRED),
    ("deep ensemble", member_logpdfs_gauss(en["mus"],
                                           te[None, :]**2 + en["sig2s"]),
     RTBLUE),
]
rng_k = np.random.default_rng(7)
fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
for name, lp, col in lp_sets:
    M = len(lp)
    ks = [k for k in (1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50) if k <= M]
    mean_k, lo_k, hi_k = [], [], []
    for k in ks:
        vals = []
        for _ in range(25):
            sub = rng_k.choice(M, size=k, replace=False)
            vals.append(-np.mean(logsumexp(lp[sub], axis=0) - np.log(k)))
        vals = np.array(vals)
        mean_k.append(vals.mean())
        lo_k.append(vals.mean() - vals.std())
        hi_k.append(vals.mean() + vals.std())
    ax.plot(ks, mean_k, "-o", color=col, lw=2.0, ms=5, label=name)
    # SUDS variant: no subset-spread bands, lines only
ax.axhline(nll_map, color=SOFTGRAY, ls="--", lw=1.4)
ax.text(1.02, nll_map + 0.004, "point estimate (MAP)", fontsize=F_TICK - 2,
        color=SOFTGRAY, va="bottom")
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
ax.set_xlabel("number of members averaged")
ax.set_ylabel("test NLL (nats)")
ax.legend(frameon=False, loc="upper right")
save(fig, "exp7_members.pdf")

# ---- F: imposter detection, one bar per method (the exp6 slide format) -----
# Each method uses the uncertainty signal it actually has, thresholded at
# its OWN 89th percentile on pristine test stars (shared 11 percent
# budget): MAP its learned sigma(x) (a single network has no member
# spread), ensemble/SGHMC/RTS their member disagreement tau.
mw = np.load(TAB / "exp7_map_weights.npz")
ew = np.load(TAB / "exp7_ensemble_weights.npz")


def sig_of(model, X):
    with torch.no_grad():
        _, r = model.mu_r(X)
        return np.exp(-3.0 + r.cpu().numpy())


def tau_of(states, X):
    mus = []
    with torch.no_grad():
        for st in states:
            H.load_flat(hm, st)
            mus.append(hm.mu_r(X)[0].cpu().numpy())
    return np.array(mus).std(0)


H.load_flat(hm, mw["state"])
map_test = sig_of(hm, Xs)
ens_test = tau_of(ew["states"], Xs)

methods_f = [
    ("point (MAP)", map_test,
     lambda X: (H.load_flat(hm, mw["state"]), sig_of(hm, X))[1]),
    ("deep ensemble", ens_test, lambda X: tau_of(ew["states"], X)),
    ("SGHMC (tuned)", tau_test_sg,
     lambda X: tau_of(sg_cold["members"], X)),
    ("ray tracing (RTS)", tau_test_r,
     lambda X: tau_of(hp["members"], X)),
]
groups_f = ("dwarfs", "hot", "flagged")
FTEXT = "#333333"
fig, ax = plt.subplots(figsize=(FIG_W, 3.3))
xg = np.arange(len(groups_f))
ax.grid(axis="y", color="black", alpha=0.12, lw=0.5, zorder=0)
for k, (name, test_scores, alarm) in enumerate(methods_f):
    th = np.quantile(test_scores, 0.89)
    fracs = []
    for gname in groups_f:
        Xo = torch.tensor(ood[f"Xs_{gname}"], dtype=torch.float32,
                          device=H.DEV)
        fracs.append(100.0 * float((alarm(Xo) > th).mean()))
    ax.bar(xg + (k - 1.5) * 0.21, fracs, 0.21, color=COLORS[name],
           edgecolor=_darken(COLORS[name]), linewidth=0.6, zorder=3,
           label=name)
    print(f"   F {name:>18}: " + " ".join(f"{g}={f:.0f}%"
          for g, f in zip(groups_f, fracs)))
ax.axhline(11.0, color="#666666", ls=(0, (4, 3)), lw=0.9, zorder=4,
           label="flag rate on pristine test (11%)")
ax.set_xticks(xg, ["dwarfs", "hot stars", "flagged by APOGEE"])
ax.set_ylabel("% flagged")
ax.set_ylim(0, 108)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(FTEXT)
ax.tick_params(colors=FTEXT)
ax.yaxis.label.set_color(FTEXT)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_color(FTEXT)
# SUDS variant: the four method colors are keyed by the structure panel
# directly above this one, so only the budget line needs a legend entry.
handles_f, labels_f = ax.get_legend_handles_labels()
i11 = labels_f.index("flag rate on pristine test (11%)")
leg = ax.legend([handles_f[i11]], [labels_f[i11]],
                loc="upper left", bbox_to_anchor=(0.0, 1.10),
                frameon=False, fontsize=F_TICK - 3, handlelength=1.2)
for text in leg.get_texts():
    text.set_color(FTEXT)
save(fig, "exp7_imposters.pdf")


# ---- V: convergence certificate, the two-panel format from the exp6 deck ---
# All three chains, each from its own recorded start. The Gaussian chain
# of record (pack3) runs from its warm start; its panels stop at
# snapshot 62,600 (15.65M steps): the recorded non-finite-window
# episode's leading misfit jitter starts at snapshot 62,622 (the full
# excursion at 62,824), the same
# era section M excludes and whose tail drags wnorm down (why
# final_state was ruled unusable for seeding). State the cut in the
# caption. The Student-t and SGHMC chains both warm-started FROM that
# final state (wnorm 18.8k): the t chain climbs back and levels near
# 23.5k in 8.5M steps; SGHMC is still climbing when its 2M-step budget
# ends, the no-certificate verdict made visible. wnorm is prior-scaled
# (||theta/sigma_prior||^2), so the prior shell is exactly D; the
# posterior settles near 2.1x the shell (the data pulls the weights
# beyond the prior scale). Misfit is each chain's own likelihood, so
# the t level is not comparable in absolute terms to the Gaussian ones.
from scipy.ndimage import median_filter

D_h = hp["members"].shape[1]
CUT_V = 62_600
m_g, w_g = hp["misfit"][:CUT_V], hp["wnorm"][:CUT_V]
assert m_g[-2_000:].max() < -185_000, "episode leaked past the cut"
w_settle = float(np.median(w_g[-12_000:]))
m_settle = float(np.median(m_g[-12_000:]))
chains_v = [
    ("RTS, Gaussian", m_g, w_g, RTGOLD, 1.4),
    ("RTS, Student-t", tpk["misfit"], tpk["wnorm"], DARKGOLD, 1.2),
    ("SGHMC (tuned)", cold["misfit"], cold["wnorm"], HMCRED, 1.2),
]
for name, m_c, w_c, _, _ in chains_v:
    print(f"   V {name:>14}: wnorm last-20% median "
          f"{np.median(w_c[-len(w_c)//5:]):,.0f}, misfit "
          f"{np.median(m_c[-len(m_c)//5:]):,.0f}")
print(f"   V Gaussian settle {w_settle/D_h:.2f}x shell D={D_h:,}")

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.2))
for name, m_c, w_c, col, lw in chains_v:
    x_c = np.arange(len(m_c)) * 250 / 1e6
    axes[0].plot(x_c, w_c / 1e3, color=col, lw=lw, label=name)
    axes[1].plot(x_c, m_c / 1e3, color=col, lw=0.5, alpha=0.22)
    axes[1].plot(x_c, median_filter(m_c, size=401, mode="nearest") / 1e3,
                 color=col, lw=1.6, label=name)
axes[0].axhline(D_h / 1e3, color="#666666", ls=(0, (4, 3)), lw=0.9,
                label=f"prior shell, D = {D_h:,}")
axes[0].annotate("SGHMC still rising\nat 2M steps", xy=(2.05, 22.7),
                 xytext=(3.6, 26.3), fontsize=F_TICK - 4, color="#666666",
                 va="top", arrowprops=dict(arrowstyle="-", color="#666666",
                                           lw=0.8))
axes[0].set_ylabel(r"$\|\theta/\sigma_{\mathrm{prior}}\|^2$  ($10^3$)")
axes[0].set_ylim(10, 27)
axes[0].legend(loc="lower right", frameon=True, facecolor="white",
               edgecolor=SOFTGRAY, framealpha=1.0, fontsize=F_TICK - 4,
               handlelength=1.0, borderpad=0.3, handletextpad=0.5)
axes[0].set_title("Squared weight norm", fontsize=F_LABEL - 1)

axes[1].set_ylim(-214.5, -168)
axes[1].set_ylabel(r"$10^3$ nats")
axes[1].legend(loc="upper right", frameon=True, facecolor="white",
               edgecolor=SOFTGRAY, framealpha=1.0, fontsize=F_TICK - 4,
               handlelength=1.0, borderpad=0.3, handletextpad=0.5)
axes[1].set_title(r"Training misfit, $-\sum \ln L$", fontsize=F_LABEL - 1)

for ax in axes:
    ax.set_xlabel("minibatch steps (millions)")
fig.tight_layout()
save(fig, "exp7_convergence.pdf")
