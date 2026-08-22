"""exp7: per-star posterior predictive, Josh's whiteboard task.

Two figures from the committed campaign packs, no new sampling:

fig A (exp7_one_star_scatter): for three chosen test stars, the 50 t-chain
members' (mu_m(x), sigma_m(x)) colored by nu_m, with the Gaussian-target MAP
prediction marked. The cloud is the posterior's disagreement about that star
(epistemic); sigma_m is the scatter each member believes (aleatoric).

fig B (exp7_map_vs_marginal): for the same stars, the MAP plug-in density
against the marginalized predictives,

    p(y | x, D) ~= (1/M) sum_m  t_{nu_m}(y; mu_m(x), s_m(x)),
    s_m^2 = yerr^2 + sigma_m(x)^2,

for both chains (Gaussian = nu -> infinity). Mixtures are averaged in density
space via logsumexp, never in log space.

Star picks: typical (median mu-cloud spread, mid-quartile yerr), largest
catalog error, largest model disagreement (max spread / mean total scale).

Cross-checks printed at the end: the test-set mixture NLLs must reproduce the
certified campaign numbers, t -1.973 and Gaussian -1.941; MAP plug-in lands
at -1.733.

Run:  cd empirical && .venv/bin/python experiments/exp7_predictive_figures.py
Outputs: plots/exp7_predictive/{exp7_one_star_scatter,exp7_map_vs_marginal}.{pdf,png}
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

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"
OUT = ROOT / "plots" / "exp7_predictive"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "experiments"))
import exp7_gaia_t as T                                       # noqa: E402
import exp7_gaia_hetero as H                                  # noqa: E402

RTGOLD, RTBLUE, SOFTGRAY, INK = "#D99A1B", "#1F6FB4", "#8A8D99", "#222222"
M_SHOWN = 10          # faint member curves drawn per panel (of 50)

plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": SOFTGRAY, "xtick.color": SOFTGRAY,
    "ytick.color": SOFTGRAY, "axes.labelcolor": INK,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def forward_members(members, module, Xt):
    model = module.make_model()
    mus, sigs, nus = [], [], []
    with torch.no_grad():
        for m in members:
            module.load_flat(model, m)
            mu, r = model.mu_r(Xt)
            mus.append(mu.cpu().numpy())
            sigs.append(np.exp(module.LNS0 + r.cpu().numpy()))
            nus.append(float(model.nu().cpu()) if hasattr(model, "nu")
                       else np.inf)
    return np.array(mus), np.array(sigs), np.array(nus)


def main():
    d = np.load(TAB / "exp7_gaia_pristine.npz")
    ty, terr = d["test_y"], d["test_yerr"]
    Xt = torch.tensor(d["test_Xs"], dtype=torch.float32, device=T.DEV)

    MU_t, SIG_t, NU_t = forward_members(
        np.load(TAB / "exp7t_pack_final.npz")["members"], T, Xt)
    MU_h, SIG_h, _ = forward_members(
        np.load(TAB / "exp7h_pack3.npz")["members"], H, Xt)
    mp = np.load(TAB / "exp7_map.npz")
    mu_map, sig_map = mp["mu"], mp["sig"]

    S_t = np.sqrt(terr[None, :] ** 2 + SIG_t ** 2)
    S_h = np.sqrt(terr[None, :] ** 2 + SIG_h ** 2)
    s_map = np.sqrt(terr ** 2 + sig_map ** 2)

    lp_t = tdist.logpdf(ty[None, :], df=NU_t[:, None], loc=MU_t, scale=S_t)
    lp_h = norm.logpdf(ty[None, :], loc=MU_h, scale=S_h)
    M = len(NU_t)
    nll_t = -(logsumexp(lp_t, axis=0) - np.log(M)).mean()
    nll_h = -(logsumexp(lp_h, axis=0) - np.log(M)).mean()
    nll_map = -norm.logpdf(ty, loc=mu_map, scale=s_map).mean()
    assert abs(nll_t - (-1.973)) < 0.01 and abs(nll_h - (-1.941)) < 0.01, \
        (nll_t, nll_h)

    spread = MU_t.std(axis=0)
    ratio = spread / S_t.mean(axis=0)
    mid = (terr > np.quantile(terr, .25)) & (terr < np.quantile(terr, .75))
    cand = np.where(mid)[0]
    i_typ = int(cand[np.argmin(np.abs(spread[cand] - np.median(spread)))])
    i_err = int(np.argmax(terr))
    i_epi = int(np.argmax(ratio))
    if i_epi == i_err:
        i_epi = int(np.argsort(ratio)[-2])
    picks = [(i_typ, "typical star"), (i_err, "largest catalog error"),
             (i_epi, "largest model disagreement")]

    print(f"test NLL  MAP plug-in {nll_map:+.3f}   Gaussian marginal "
          f"{nll_h:+.3f}   Student-t marginal {nll_t:+.3f}")
    print(f"mu-cloud spread median {np.median(spread):.4f}, "
          f"spread/scale ratio median {np.median(ratio):.3f}, "
          f"p99 {np.quantile(ratio, .99):.3f}")
    for i, name in picks:
        print(f"  {name}: star {i}, yerr {terr[i]:.4f}, "
              f"spread {spread[i]:.4f}, ratio {ratio[i]:.3f}")

    # ---- fig A -------------------------------------------------------------
    figA, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, (i, name) in zip(axes, picks):
        ax.scatter(MU_t[:, i], SIG_t[:, i], s=26, color=RTGOLD, alpha=0.85,
                   linewidths=0, zorder=3)
        ax.scatter([mu_map[i]], [sig_map[i]], marker="X", s=110, color=INK,
                   zorder=4)
        ax.axvline(ty[i], color=SOFTGRAY, lw=1.0, ls=":", zorder=1)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(r"$\mu_m(x)$")
    axes[0].set_ylabel(r"$\sigma_m(x)$")
    axes[0].annotate("MAP", (mu_map[i_typ], sig_map[i_typ]), xytext=(-7, -3),
                     textcoords="offset points", ha="right", fontsize=11,
                     color=INK)
    axes[0].text(ty[i_typ], 0.97, " catalog $y$", color=SOFTGRAY, fontsize=10,
                 ha="left", va="top",
                 transform=axes[0].get_xaxis_transform())
    for ext in ("pdf", "png"):
        figA.savefig(OUT / f"exp7_one_star_scatter.{ext}", dpi=150,
                     bbox_inches="tight")

    # ---- fig B -------------------------------------------------------------
    figB, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, (i, name) in zip(axes, picks):
        c = MU_t[:, i].mean()
        half = 5.0 * max(S_t[:, i].max(), s_map[i], abs(mu_map[i] - c))
        grid = np.linspace(c - half, c + half, 800)
        lp_mem = tdist.logpdf(grid[None, :], df=NU_t[:, None],
                              loc=MU_t[:, i][:, None],
                              scale=S_t[:, i][:, None])
        mix_t = np.exp(logsumexp(lp_mem, axis=0) - np.log(M))
        lp_gm = norm.logpdf(grid[None, :], loc=MU_h[:, i][:, None],
                            scale=S_h[:, i][:, None])
        mix_h = np.exp(logsumexp(lp_gm, axis=0) - np.log(M))
        for row in np.exp(lp_mem)[:: M // M_SHOWN]:
            ax.plot(grid, row, color=RTGOLD, lw=0.6, alpha=0.15, zorder=1)
        ax.plot(grid, norm.pdf(grid, mu_map[i], s_map[i]), color=SOFTGRAY,
                lw=1.6, ls="--", zorder=2)
        ax.plot(grid, mix_h, color=RTBLUE, lw=1.6, zorder=3)
        ax.plot(grid, mix_t, color=RTGOLD, lw=2.2, zorder=4)
        ax.axvline(ty[i], color=INK, lw=1.0, ls=":", zorder=5)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(r"$y$ (alpha abundance)")
        ax.set_yticks([])
    axes[0].set_ylabel("predictive density")
    for k, (txt, col) in enumerate([("MAP plug-in", SOFTGRAY),
                                    ("Gaussian marginal", RTBLUE),
                                    ("Student-t marginal", RTGOLD)]):
        axes[0].text(0.03, 0.94 - 0.10 * k, txt, color=col, fontsize=11,
                     transform=axes[0].transAxes)
    for ext in ("pdf", "png"):
        figB.savefig(OUT / f"exp7_map_vs_marginal.{ext}", dpi=150,
                     bbox_inches="tight")

    print("saved to", OUT)
    print(
        "\ncaption A: One held-out star pushed through the 50 posterior "
        "draws of the Student-t chain;\neach dot is one draw's predicted "
        f"center and scatter for that star, X marks the Adam MAP fit,\n"
        "the dotted line is the catalog value. nu varies only "
        f"{NU_t.min():.2f} to {NU_t.max():.2f} across draws and is omitted."
        "\n\ncaption B: Predictive densities for the same stars: MAP plug-in "
        "(gray, dashed), marginalized\nGaussian chain (blue), marginalized "
        f"Student-t chain (gold; thin curves show {M_SHOWN} of its 50 "
        f"members).\nDotted line: catalog value. Test-set NLL over "
        f"{len(ty):,} stars: MAP {nll_map:.3f}, Gaussian marginal "
        f"{nll_h:.3f},\nStudent-t marginal {nll_t:.3f}.")


if __name__ == "__main__":
    main()
