"""Matplotlib figures for the exp6 results (the five-step MNIST plan).

Outputs (results/figures/):
  exp6_fig1_march.png       MLP ln-posterior trace + likelihood/prior decomposition
  exp6_fig2_cliff.png       acceptance vs step size, the razor cliff, both models
  exp6_fig3_predictive.png  NLL and ECE per predictive, the window contrast

Colors are the project palette snapped to darker steps that pass the categorical
checks on a white surface (lightness band, chroma floor, CVD separation, 3:1
contrast): gold=RT chain, blue=likelihood/ensemble, purple=point, teal=prior.
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp6_simple_mnist_train as base

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOLD, BLUE, PURPLE, TEAL = "#C0871F", "#6E8BE8", "#7A5EA8", "#0F9678"
INK, MUTED = "#2b2b33", "#7a7a85"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})

DEV = base.DEV
TAB = base.TAB
FIG = base.ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
D_MLP = 50890


def load_chain(name):
    z = np.load(TAB / f"exp6_rt_chain_{name}.npz")
    return z["samples"], np.asarray(z["ln_post"])


# ---------------------------------------------------------------- figure 1
def fig1_march():
    samples, lnp = load_chain("mlp")
    n = len(lnp)

    model = base.MLP().to(DEV)
    model.load_state_dict(torch.load(base.CKPT / "exp6_mlp.pt", map_location=base.DEV))
    specs = [(nm, p.shape, p.numel()) for nm, p in model.named_parameters()]
    tr_loader, _, _ = base.loaders()
    xs, ys = [], []
    for x, y in torch.utils.data.DataLoader(tr_loader.dataset, batch_size=10000):
        xs.append(x.to(DEV)); ys.append(y.to(DEV))

    def decompose(theta):
        pd, i = {}, 0
        for nm, sh, sz in specs:
            pd[nm] = theta[i:i + sz].view(sh); i += sz
        with torch.no_grad():
            ce = sum(F.cross_entropy(torch.func.functional_call(model, pd, (xb,)),
                                     yb, reduction="sum").item()
                     for xb, yb in zip(xs, ys))
        return ce, float((theta ** 2).sum())

    ks = np.linspace(0, n - 1, 41).astype(int)
    ce_v, pr_v = [], []
    for k in ks:
        ce, sq = decompose(torch.tensor(samples[k], device=DEV))
        ce_v.append(ce); pr_v.append(sq / 2)
    ce_v, pr_v = np.array(ce_v), np.array(pr_v)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.14})

    ax1.plot(lnp, color=GOLD, lw=1.3)
    peak = int(np.argmax(lnp))
    ax1.annotate(f"mode region\n(peak {lnp[peak]:,.0f} at trajectory {peak})",
                 xy=(peak, lnp[peak]), xytext=(2600, -4800),
                 fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax1.annotate("descent: entropy pulling the chain\nout to the prior shell",
                 xy=(12000, lnp[12000]), xytext=(12800, -11500),
                 fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    target = -(ce_v[-1] + D_MLP / 2)
    ax1.axhline(target, color=MUTED, ls="--", lw=1)
    ax1.text(300, target + 700,
             f"predicted typical set: -(sum CE + D/2) = {target:,.0f}",
             fontsize=9, color=MUTED)
    ax1.set_ylim(target - 2500, 0)
    ax1.set_ylabel("ln posterior (nats)")
    ax1.set_title("MLP chain: 20,000 exact-Metropolis trajectories, still marching "
                  "(97% acceptance)")

    ax2.plot(ks, pr_v, color=TEAL, lw=1.8, marker="o", ms=2.6, label="prior cost")
    ax2.plot(ks, ce_v, color=BLUE, lw=1.8, marker="o", ms=2.6, label="data misfit")
    ax2.axhline(D_MLP / 2, color=MUTED, ls="--", lw=1)
    ax2.text(300, D_MLP / 2 + 700, f"prior shell: D/2 = {D_MLP // 2:,} nats",
             fontsize=9, color=MUTED)
    ax2.annotate(f"prior cost  ½||θ||²\n{pr_v[0]:,.0f} → {pr_v[-1]:,.0f}",
                 xy=(ks[26], pr_v[26]), xytext=(9200, 8200), fontsize=9, color=TEAL,
                 arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.8))
    ax2.annotate(f"data misfit  sum CE\n{ce_v[0]:,.0f} → {ce_v[-1]:,.0f} "
                 "(fits train better than Adam)",
                 xy=(ks[10], ce_v[10] + 300), xytext=(5800, 3200), fontsize=9, color=BLUE,
                 arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax2.set_ylim(-800, 29500)
    ax2.set_xlabel("trajectory")
    ax2.set_ylabel("nats")
    ax2.set_title("The descent decomposed: the prior term does all the marching",
                  fontsize=10)

    fig.savefig(FIG / "exp6_fig1_march.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "exp6_fig1_march.png")


# ---------------------------------------------------------------- figure 2
def fig2_cliff():
    # measured acceptance points: (dt, acc) per model, L = 30 trajectories
    pts = {"mlp": [], "cnn": []}
    pilot = (TAB / "exp6_pilot.log").read_text()
    for m in re.finditer(r"\[(mlp|cnn)\] L= 30 dt=(\S+)\s+acc=([\d.]+)", pilot):
        pts[m.group(1)].append((float(m.group(2)), float(m.group(3))))
    auto = (TAB / "exp6_auto.log").read_text()
    for m in re.finditer(r"\[tune:(mlp|cnn)\] dt=(\S+) acc=([\d.]+)", auto):
        pts[m.group(1)].append((float(m.group(2)), float(m.group(3))))
    # first tuner pass (log overwritten by the rerun; values recorded in the log
    # of 2026-07-09): the far side of the MLP cliff
    pts["mlp"] += [(4e-4, 0.58), (5e-4, 0.0), (6e-4, 0.0)]

    chosen = {"mlp": (3.5e-4, 0.97), "cnn": (1.5e-4, 0.99)}
    colors = {"mlp": GOLD, "cnn": BLUE}

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for name in ("mlp", "cnn"):
        arr = np.array(sorted(pts[name]))
        dts = np.unique(arr[:, 0])
        mean = [arr[arr[:, 0] == d, 1].mean() for d in dts]
        ax.plot(dts, mean, color=colors[name], lw=1.8, zorder=2)
        ax.scatter(arr[:, 0], arr[:, 1], s=26, color=colors[name], zorder=3,
                   edgecolors="white", linewidths=0.8)
        cd, ca = chosen[name]
        ax.scatter([cd], [ca], marker="*", s=180, color=colors[name], zorder=4,
                   edgecolors="white", linewidths=0.8)
    ax.annotate("MLP chosen: dt=3.5e-4\n(85% in tuning, 97% realized)",
                xy=chosen["mlp"], xytext=(4.6e-4, 0.72), fontsize=9, color=GOLD,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.8))
    ax.annotate("CNN chosen: dt=1.5e-4\n(95% in tuning, 99% realized)",
                xy=chosen["cnn"], xytext=(2.6e-5, 0.62), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.axhline(0.8, color=MUTED, ls=":", lw=1)
    ax.text(1.05e-5, 0.815, "80% tuning target", fontsize=9, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("step size dt (log scale)")
    ax.set_ylabel("acceptance (higher is better, target 80%)")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("The acceptance cliff: a factor of 3 in step size separates "
                 "~100% from 0%")
    fig.savefig(FIG / "exp6_fig2_cliff.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "exp6_fig2_cliff.png")


# ---------------------------------------------------------------- figure 3
def fig3_predictive():
    _, te_loader, _ = base.loaders()

    def ece(probs, labels, bins=15):
        conf, pred = probs.max(1), probs.argmax(1)
        correct = (pred == labels).astype(float)
        edges = np.linspace(0, 1, bins + 1)
        out = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                out += m.mean() * abs(correct[m].mean() - conf[m].mean())
        return out

    def chain_predictive(name, lo, hi, s_pred=50):
        samples, _ = load_chain(name)
        model = base.ARCH[name]().to(DEV) if hasattr(base, "ARCH") else \
            {"mlp": base.MLP, "cnn": base.CNN}[name]().to(DEV)
        specs = [(nm, p.shape, p.numel()) for nm, p in model.named_parameters()]
        probs = []
        for k in np.linspace(lo, hi - 1, s_pred).astype(int):
            theta = torch.tensor(samples[k], device=DEV)
            pd, i = {}, 0
            for nm, sh, sz in specs:
                pd[nm] = theta[i:i + sz].view(sh); i += sz
            with torch.no_grad():
                p = [F.softmax(torch.func.functional_call(model, pd, (x.to(DEV),)), 1)
                     .cpu().numpy() for x, _ in te_loader]
            probs.append(np.concatenate(p))
        return np.stack(probs)

    variants = {}
    for name, n_chain in (("mlp", 20000), ("cnn", 600)):
        z_pt = np.load(TAB / f"exp6_point_{name}.npz")
        z_en = np.load(TAB / f"exp6_ensemble_{name}.npz")
        labels = z_pt["labels"]
        rows = [("point estimate", z_pt["probs"][None], PURPLE, ""),
                ("deep ensemble (10)", z_en["probs"], BLUE, ""),
                ("RT chain, final quarter", chain_predictive(name, 3 * n_chain // 4,
                                                             n_chain), GOLD, "")]
        if name == "mlp":
            rows.append(("RT chain, first 2k traj", chain_predictive(name, 0, 2000),
                         GOLD, "//"))
        variants[name] = (labels, rows)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.4),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.32})
    for col, name in enumerate(("mlp", "cnn")):
        labels, rows = variants[name]
        names = [r[0] for r in rows]
        nll = [-np.log(r[1].mean(0)[np.arange(len(labels)), labels] + 1e-12).mean()
               for r in rows]
        ec = [ece(r[1].mean(0), labels) for r in rows]
        for row, (vals, metric) in enumerate(((nll, "test NLL"), (ec, "ECE"))):
            ax = axes[row][col]
            # top-align rows across columns so bars line up with the shared labels
            y = 3 - np.arange(len(rows))
            for yi, v, r in zip(y, vals, rows):
                ax.barh(yi, v, height=0.62, color=r[2], hatch=r[3],
                        edgecolor="white", linewidth=0.8)
                ax.text(v + max(vals) * 0.02, yi, f"{v:.4f}", va="center",
                        fontsize=8.5, color=INK)
            ax.set_ylim(-0.55, 3.55)
            ax.set_yticks(y)
            ax.set_yticklabels(names if col == 0 else [""] * len(rows), fontsize=9)
            ax.set_xlim(0, max(vals) * 1.22)
            ax.set_xlabel(f"{metric} (lower is better)", fontsize=9)
            ax.grid(axis="y", visible=False)
            if row == 0:
                ax.set_title(name.upper(), fontsize=10)
    fig.suptitle("Same chain, two windows, two stories: the predictive depends on "
                 "where the transient was stopped", fontsize=11, y=1.0)
    fig.savefig(FIG / "exp6_fig3_predictive.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "exp6_fig3_predictive.png")


if __name__ == "__main__":
    fig1_march()
    fig2_cliff()
    fig3_predictive()
