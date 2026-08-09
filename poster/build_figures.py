"""Figure builder for the SUDS poster (poster_v2 reconstruction).

Builds ONLY the two redesigned figures:
    E  artifacts/figures/typical_set.pdf        (col 1, "Where the probability lives")
    S  artifacts/figures/shell_diagnostic.pdf   (col 3, "The shell, measured")

The other figures in artifacts/figures/ (fig0_posterior_draws_harder.pdf,
fig3_scaling_demonstration.pdf, hmc_vs_rt_noise.png) and the logos in figures/
were extracted from the cloud-session build of this poster (see
artifacts/figures/README.md) and are deliberately NOT regenerated here.

Shell-diagnostic data: real chains from the repo's validated stochastic-gradient
harness (explainer/sg_experiment.py + explainer/physics.snell_kick, the same code
behind the sigma_c / fig3 measurements). The loop bodies are copied verbatim below
with position recording added; a bit-identity assertion against the untouched
functions guarantees the copies drift-proof. No Metropolis test, matching the
paper's Eq. 34 stochastic-gradient experiment. Cached in artifacts/shell_chains.npz.

Run:  /Users/yasiralsugair/UofT/empirical/.venv/bin/python build_figures.py [ES]

Style mirrors poster/make_poster_figs.py (softgray spines/ticks, ink labels,
top/right spines off, bbox_inches="tight" vector PDFs). Font sizes are larger
than that script's: these figures print at 0.82 * 37.8 cm = 31 cm = 12.2 in,
so every in-figure string uses >= 24 * width_in / 12.2 pt to read at >= 24 pt
on the printed poster.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi, chi2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent          # .../UofT/poster_v2
FIGS = ROOT / "artifacts" / "figures"
CHAINS = ROOT / "artifacts" / "shell_chains.npz"

sys.path.insert(0, str(ROOT.parent / "explainer"))   # physics.py, sg_experiment.py
from physics import snell_kick                        # noqa: E402
import sg_experiment                                  # noqa: E402

# colors mirrored EXACTLY from the suds_poster.tex preamble
RTGOLD = "#D99A1B"
RTBLUE = "#1F6FB4"
HMCRED = "#D1495B"
SOFTGRAY = "#8A8D99"
INK = "#20222B"
DARKGOLD = "#8A5F06"   # dark-gold-on-gold text, as in make_poster_figs

# printed width of a 0.82\linewidth figure: 0.82 * 0.31 * 121.92 cm = 30.99 cm = 12.20 in
PRINT_W_IN = 12.20
FIG_W = 7.4                       # inches; house full-width convention
MAG = PRINT_W_IN / FIG_W          # ~1.65x when printed
F_TICK = 15                       # -> ~24.7 pt printed
F_LABEL = 16                      # -> ~26.4 pt printed
F_ANNO = 15.5                     # -> ~25.6 pt printed

plt.rcParams.update({
    "font.size": F_TICK,
    "axes.edgecolor": SOFTGRAY,
    "axes.labelcolor": INK,
    "xtick.color": SOFTGRAY,
    "ytick.color": SOFTGRAY,
    "xtick.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})


def save(fig, name, print_w_in=PRINT_W_IN, ticks_pt=F_TICK, labels_pt=F_LABEL):
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    from pypdf import PdfReader
    box = PdfReader(FIGS / name).pages[0].mediabox
    w_in = float(box.width) / 72.0
    mag = print_w_in / w_in
    print(f"wrote {FIGS / name}  ({w_in:.2f} in wide; printed magnification "
          f"{mag:.2f}x -> ticks {ticks_pt * mag:.1f} pt, labels {labels_pt * mag:.1f} pt)")


# ---------------------------------------------------------------- typical set

def _t_pdf(t, D):
    """Density of t = |theta|^2 / D for theta ~ N(0, I_D); integrates to 1 in t."""
    return D * chi2.pdf(D * t, df=D)


def fig_typical_set():
    """Log-space decomposition at D = 100: log p(r), (D-1) log r, and their
    pointwise sum, all exact, no per-curve offsets or rescaling, so the reader
    can verify blue + gold = dark by eye against the zero line. Below, the
    normalised radial mass (chi_D) shows how sharp the shell really is once
    the log compression is undone."""
    Dd = 100
    r = np.linspace(1.0, 16.0, 1200)
    log_dens = -0.5 * r ** 2
    log_vol = (Dd - 1) * np.log(r)
    log_mass = log_vol + log_dens
    r_peak = float(np.sqrt(Dd - 1))          # true mode of chi_D

    # this figure prints at 0.70\linewidth = 26.5 cm = 10.42 in (0.82 overflows
    # column 1), so its fonts are larger than the module defaults to keep
    # >= 24 pt printed: size >= 24 * width_in / 10.42
    ft, fl, fa = 19, 20, 19.5
    fig, (ax, axs) = plt.subplots(
        2, 1, sharex=True, figsize=(8.4, 3.2),
        gridspec_kw=dict(height_ratios=[4, 1], hspace=0.08))

    # zero reference line: what makes blue + gold = dark checkable by eye
    ax.axhline(0.0, color=SOFTGRAY, lw=1.0, zorder=1)
    for a in (ax, axs):
        a.axvline(r_peak, color=SOFTGRAY, lw=1.6, ls=(0, (2, 2)), zorder=1)

    ax.plot(r, log_dens, color=RTBLUE, lw=3.2, zorder=3)
    ax.plot(r, log_vol, color=RTGOLD, lw=3.2, zorder=3)
    ax.plot(r, log_mass, color=INK, lw=3.8, zorder=4)

    # dots where the guide crosses the curves: at the peak, gold + blue = dark
    for y, c in ((-0.5 * r_peak ** 2, RTBLUE),
                 ((Dd - 1) * np.log(r_peak), RTGOLD),
                 ((Dd - 1) * np.log(r_peak) - 0.5 * r_peak ** 2, INK)):
        ax.plot([r_peak], [y], marker="o", ms=7.5, color=c, zorder=5)

    # direct labels, no legend box
    lab = dict(fontsize=fa, fontweight="bold")
    ax.text(3.05, 195, r"volume $r^{D-1}$", color=RTGOLD, ha="center",
            va="center", **lab)
    ax.text(12.5, 205, r"mass $p(r)\,r^{D-1}$", color=INK, ha="center",
            va="center", **lab)
    ax.text(5.6, -85, r"density $p(r)$", color=RTBLUE, ha="left",
            va="center", **lab)
    ax.text(9.6, 300, r"$\|\theta\| \approx \sqrt{D}$", color=INK, ha="right",
            va="top", fontsize=fa)

    ax.set_ylim(-140, 305)
    ax.set_yticks([-100, 0, 100, 200, 300])
    ax.tick_params(labelsize=ft)
    ax.set_ylabel("log (unnormalised)", fontsize=fl, y=0.40)

    # lower strip: the shell itself, in linear space
    mass = chi.pdf(r, Dd)
    axs.fill_between(r, mass, 0.0, color=RTGOLD, alpha=0.30, lw=0)
    axs.plot(r, mass, color=RTGOLD, lw=2.0)
    axs.text(1.35, 0.52 * float(mass.max()), "mass, linear scale",
             color=DARKGOLD, ha="left", va="center", fontsize=fa)
    axs.set_ylim(0.0, 1.15 * float(mass.max()))
    axs.set_yticks([])
    axs.spines["left"].set_visible(False)
    axs.set_xlim(1.0, 16.0)
    axs.set_xticks([1, 5, 10, 15])
    axs.tick_params(labelsize=ft)
    axs.set_xlabel(r"$r = \|\theta\|$", fontsize=fl)
    save(fig, "typical_set_local.pdf", print_w_in=0.70 * 37.8 / 2.54,
         ticks_pt=ft, labels_pt=fl)


# ------------------------------------------------- shell diagnostic (real runs)

D = 100
N_STEPS = 60000
BURN = 10000
N_CHAINS = 4
SIGMA = 20.0
DS_INPUT = 0.03                      # radius_traces / s1_radius.npz configuration
DS = float(np.sqrt(D) * DS_INPUT)    # RT arc-length step 0.3
DT = DS_INPUT                        # HMC leapfrog step
HMC_M = 30                           # full momentum refresh period (as in s1_radius.npz)
RT_F = 0.03                          # RT partial-refresh rate
SEED = {"hmc_exact": 91000, "hmc_sto": 92000, "rt_sto": 93000}


def _run_hmc_positions(x0, n_steps, dt, M, sigma, rng):
    """run_hmc_sto loop body, verbatim, with positions recorded."""
    Dd = x0.shape[0]
    x = np.array(x0, float)
    p = rng.standard_normal(Dd)
    out = np.empty(n_steps + 1)
    xs = np.empty((n_steps + 1, Dd))
    out[0] = -0.5 * float(x @ x)
    xs[0] = x
    for i in range(1, n_steps + 1):
        if (i - 1) % M == 0:
            p = rng.standard_normal(Dd)
        p = p + 0.5 * dt * sg_experiment._grad_lnL_sto(x, sigma, rng)
        x = x + dt * p
        p = p + 0.5 * dt * sg_experiment._grad_lnL_sto(x, sigma, rng)
        out[i] = -0.5 * float(x @ x)
        xs[i] = x
    return xs, out


def _run_rts_positions(x0, n_steps, ds, f, sigma, rng):
    """run_rts_sto loop body, verbatim, with positions recorded."""
    Dd = x0.shape[0]
    x = np.array(x0, float)
    v = rng.standard_normal(Dd)
    a = float(np.exp(-abs(f)))
    b = float(np.sqrt(1.0 - a * a))
    out = np.empty(n_steps + 1)
    xs = np.empty((n_steps + 1, Dd))
    out[0] = -0.5 * float(x @ x)
    xs[0] = x
    for i in range(1, n_steps + 1):
        nv = float(np.linalg.norm(v)); x = x + 0.5 * ds * (v / nv)
        g = sg_experiment._grad_lnL_sto(x, sigma, rng)
        v = snell_kick(v, g / (Dd - 1.0), ds, Dd)[0]
        nv = float(np.linalg.norm(v)); x = x + 0.5 * ds * (v / nv)
        v = a * v + b * rng.standard_normal(Dd)
        out[i] = -0.5 * float(x @ x)
        xs[i] = x
    return xs, out


def split_rhat(chains):
    """Split R-hat (BDA3 / Gelman et al.) per coordinate. chains: (M, N, D)."""
    M, N, Dd = chains.shape
    half = N // 2
    seq = chains[:, : 2 * half].reshape(M * 2, half, Dd)
    W = seq.var(axis=1, ddof=1).mean(axis=0)
    B_over_n = seq.mean(axis=1).var(axis=0, ddof=1)
    var_hat = (half - 1) / half * W + B_over_n
    return np.sqrt(var_hat / W)


def run_shell_chains():
    if CHAINS.exists():
        return dict(np.load(CHAINS, allow_pickle=True))

    def chains_for(kind, sigma):
        r2, lnls, pos = [], [], []
        for c in range(N_CHAINS):
            rng = np.random.default_rng(SEED[kind] + c)
            x0 = rng.standard_normal(D)
            if kind.startswith("hmc"):
                xs, lnl = _run_hmc_positions(x0, N_STEPS, DT, HMC_M, sigma, rng)
            else:
                xs, lnl = _run_rts_positions(x0, N_STEPS, DS, RT_F, sigma, rng)
            # bit-identity: the untouched harness, same seed, must give this lnL trace
            rng2 = np.random.default_rng(SEED[kind] + c)
            x02 = rng2.standard_normal(D)
            if kind.startswith("hmc"):
                ref = sg_experiment.run_hmc_sto(x02, N_STEPS, DT, HMC_M, sigma, rng2)
            else:
                ref = sg_experiment.run_rts_sto(x02, N_STEPS, DS, RT_F, sigma, rng2)
            assert np.array_equal(lnl, ref), f"driver diverged from harness: {kind} c={c}"
            r2.append((xs ** 2).sum(axis=1) / D)
            lnls.append(lnl)
            pos.append(xs)
        return np.array(r2), np.array(lnls), np.array(pos)

    out = {}
    for kind, sigma in (("hmc_exact", 0.0), ("hmc_sto", SIGMA), ("rt_sto", SIGMA)):
        r2, lnl, pos = chains_for(kind, sigma)
        post = pos[:, BURN + 1:, :]
        rh = split_rhat(post)
        out[f"{kind}_r2"] = r2
        out[f"{kind}_rhat_max"] = np.array(float(rh.max()))
        out[f"{kind}_rhat_med"] = np.array(float(np.median(rh)))
        out[f"{kind}_mean_r2"] = np.array(float(r2[:, BURN + 1:].mean()))
        print(f"{kind:10s} sigma={sigma:4.1f}  post-warmup <r2/D> = "
              f"{out[f'{kind}_mean_r2']:.3f}  split-Rhat max = {rh.max():.3f} "
              f"median = {np.median(rh):.3f}")
    out["provenance"] = np.array(json.dumps(dict(
        harness="explainer/sg_experiment.py run_hmc_sto/run_rts_sto loop bodies, "
                "verbatim copy with position recording; kick = explainer/physics.snell_kick; "
                "lnL traces asserted bit-identical to the untouched functions",
        target="standard Gaussian N(0, I_D)", D=D, n_steps=N_STEPS, burn=BURN,
        n_chains=N_CHAINS, sigma=SIGMA, dt=DT, hmc_M=HMC_M, ds=DS, rt_f=RT_F,
        seeds={k: [v + c for c in range(N_CHAINS)] for k, v in SEED.items()},
        no_metropolis="matches the paper's Eq. 34 stochastic-gradient experiment",
    )))
    np.savez_compressed(CHAINS, **out)
    print("cached ->", CHAINS)
    return out


def fig_shell_diagnostic():
    d = run_shell_chains()
    post = {k: d[f"{k}_r2"][:, BURN + 1:].ravel() for k in ("hmc_exact", "hmc_sto", "rt_sto")}
    rhat = float(d["hmc_sto_rhat_max"])
    m_sto = float(d["hmc_sto_mean_r2"])

    lo = 0.55
    hi = float(np.quantile(post["hmc_sto"], 0.999)) + 0.35
    t = np.linspace(lo, hi, 1500)
    ref = _t_pdf(t, D)

    binw = 0.075

    def own_bins(x):
        # fixed bin width, but only over this dataset's own support: no
        # zero-density outline running along the x-axis elsewhere
        a = max(lo, float(x.min()) - 0.5 * binw)
        b = min(hi, float(x.max()) + 0.5 * binw)
        return np.arange(a, b + binw, binw)

    fig, ax = plt.subplots(figsize=(FIG_W, 3.4))
    ax.fill_between(t, ref, 0.0, color=SOFTGRAY, alpha=0.42, lw=0, zorder=1)
    ax.hist(post["rt_sto"], bins=own_bins(post["rt_sto"]), density=True,
            histtype="stepfilled", color=RTBLUE, alpha=0.38, lw=0, zorder=2)
    ax.hist(post["rt_sto"], bins=own_bins(post["rt_sto"]), density=True,
            histtype="step", color=RTBLUE, lw=2.2, zorder=4)
    ax.hist(post["hmc_sto"], bins=own_bins(post["hmc_sto"]), density=True,
            histtype="stepfilled", color=HMCRED, alpha=0.38, lw=0, zorder=2)
    ax.hist(post["hmc_sto"], bins=own_bins(post["hmc_sto"]), density=True,
            histtype="step", color=HMCRED, lw=2.2, zorder=3)
    ax.hist(post["hmc_exact"], bins=own_bins(post["hmc_exact"]), density=True,
            histtype="step", color=HMCRED, lw=2.0, ls=(0, (4, 2)), zorder=5)

    peak = float(ref.max())
    bump_top = float(np.histogram(post["hmc_sto"], bins=own_bins(post["hmc_sto"]),
                                  density=True)[0].max())
    lab = dict(fontsize=F_ANNO, fontweight="bold")
    ax.text(1.42, 0.995 * peak, "the shell", color=INK, ha="left", va="center",
            fontsize=F_ANNO)
    ax.text(1.42, 0.875 * peak, r"$\chi^2_D\,/\,D$", color=SOFTGRAY, ha="left",
            va="center", fontsize=F_ANNO)
    # swatch leaders touching the on-shell curves they name
    ax.plot([1.16, 1.38], [0.735 * peak] * 2, color=HMCRED, lw=2.0,
            ls=(0, (4, 2)), zorder=6, clip_on=False)
    ax.text(1.42, 0.735 * peak, "HMC, exact gradient", color=HMCRED, ha="left",
            va="center", **lab)
    ax.plot([1.16, 1.38], [0.615 * peak] * 2, color=RTBLUE, lw=2.2, zorder=6)
    ax.text(1.42, 0.615 * peak, r"ray tracing, $\sigma = 20$", color=RTBLUE,
            ha="left", va="center", **lab)
    ax.text(m_sto + 0.07, bump_top + 0.10 * peak, r"stochastic HMC, $\sigma = 20$",
            color=HMCRED, ha="center", va="bottom", **lab)
    ax.text(m_sto, 0.42 * bump_top, rf"split $\hat{{R}} = {rhat:.2f}$", color=INK,
            ha="center", va="center", fontsize=F_ANNO)

    ax.set_xlim(lo, hi)
    ax.set_ylim(0.0, 1.16 * peak)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=F_TICK)
    ax.set_xlabel(r"$\|\theta\|^2 \, / \, D$", fontsize=F_LABEL)
    fig.tight_layout()
    save(fig, "shell_diagnostic.pdf")





# ======================================================================
# v4 figures (suds_poster_v4.tex). Slots and printed widths:
#   B  fig0_data_bands.pdf         col 1 @ 0.60 -> 22.7 cm =  8.93 in
#   G  fig2_noise_geometry.pdf     col 2 @ 0.50 -> 18.9 cm =  7.44 in
#   T  fig8_tolerance_measured.pdf col 2 @ 0.87 -> 32.9 cm = 12.95 in
#   R  fig10_real_posterior.pdf    col 3 @ 0.92 -> 34.8 cm = 13.69 in
# Every curve is real data already in this repo tree: poster/poster_data.npz
# (sampled toy posterior + Adam fit), artifacts/shell_chains.npz (cached
# sigma=20 chains, provenance inside), poster/figures/sigma_c_data.npz
# (deviation_sweep cache, D=100, 6 chains/point), and the committed exp7
# packs in empirical/results/tables. Nothing is synthesized or smoothed;
# traces are thinned by plain slicing for plot weight only.

POSTER1 = ROOT.parent / "poster"
EMP_TAB = ROOT.parent / "empirical" / "results" / "tables"


def fig_data_bands():
    """fig0: 120 real posterior draws vs the single Adam fit (poster_data.npz)."""
    d = np.load(POSTER1 / "poster_data.npz")
    xg = d["grid_raw"]
    mu, sd = d["posterior_mean"], d["posterior_sd"]
    ft, fl, fa = 19, 20, 19.5           # >= 24 pt at the 1.28x slot magnification

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    nodata = d["region"] != "train"
    edges = np.flatnonzero(np.diff(nodata.astype(int)))
    spans, open_at = [], (0 if nodata[0] else None)
    for e in edges:
        if open_at is None:
            open_at = e + 1
        else:
            spans.append((open_at, e)); open_at = None
    if open_at is not None:
        spans.append((open_at, len(xg) - 1))
    for i0, i1 in spans:
        ax.axvspan(xg[i0], xg[i1], color=SOFTGRAY, alpha=0.10, lw=0)

    ax.fill_between(xg, mu - 2 * sd, mu + 2 * sd, color=RTGOLD, alpha=0.14, lw=0)
    for f in d["posterior_spaghetti"]:
        ax.plot(xg, f, color=RTGOLD, lw=0.55, alpha=0.20, zorder=1)
    ax.plot(xg, d["optimizer_function"], color=RTBLUE, lw=3.0, zorder=3)
    ax.plot(xg, d["true_f_raw"], color=INK, lw=2.0, ls=(0, (4, 2)), zorder=2)
    ax.scatter(d["x_raw"], d["y_raw"], s=26, color=INK, zorder=4)

    lab = dict(fontsize=fa, fontweight="bold")
    ax.text(-3.9, 3.6, "posterior draws", color=DARKGOLD, ha="left", **lab)
    ax.text(-3.9, 2.9, "one Adam fit", color=RTBLUE, ha="left", **lab)
    ax.text(-3.9, 2.2, "truth", color=INK, ha="left", fontsize=fa)
    ax.annotate("no data here", xy=(0.0, -3.9), ha="center", fontsize=fa,
                color=SOFTGRAY)
    ax.set_xlim(-4, 4); ax.set_ylim(-4.4, 4.3)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.tick_params(labelsize=ft)
    ax.set_xlabel("input", fontsize=fl)
    ax.set_ylabel("output", fontsize=fl)
    fig.tight_layout()
    save(fig, "fig0_data_bands_local.pdf", print_w_in=8.93, ticks_pt=ft, labels_pt=fl)


def fig_noise_geometry():
    """fig2: the cached sigma=20 chains. Stochastic HMC heats off the shell;
    ray tracing under the SAME injected noise holds it. All 4 chains per
    sampler, thinned by slicing (every 25th recorded step, no smoothing)."""
    d = dict(np.load(CHAINS, allow_pickle=True))
    ft, fa = 19, 19
    thin = slice(None, None, 25)
    steps = np.arange(60001)[thin] / 1e3

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.axhline(1.0, color=SOFTGRAY, lw=1.6, ls=(0, (2, 2)), zorder=1)
    for c in range(4):
        ax.plot(steps, d["hmc_sto_r2"][c][thin], color=HMCRED, lw=0.9,
                alpha=0.75, zorder=3)
        ax.plot(steps, d["rt_sto_r2"][c][thin], color=RTGOLD, lw=0.9,
                alpha=0.75, zorder=4)
    rhat = float(d["hmc_sto_rhat_max"])
    top = float(d["hmc_sto_r2"].max())
    lab = dict(fontsize=fa, fontweight="bold")
    ax.text(30, top + 1.05, "stochastic HMC", color=HMCRED, ha="center", **lab)
    ax.text(30, top + 0.42, rf"split $\hat{{R}} = {rhat:.2f}$: no alarm",
            color=HMCRED, ha="center", fontsize=fa - 2)
    ax.text(30, 0.22, "ray tracing, same noise", color=DARKGOLD,
            ha="center", **lab)
    ax.text(65.5, 1.0, "the\nshell", color=SOFTGRAY, ha="center",
            va="center", fontsize=fa - 2)
    ax.set_xlim(0, 70)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_ylim(0.0, top + 1.75)
    ax.tick_params(labelsize=ft)
    ax.set_xlabel("step (thousands)", fontsize=ft + 1)
    ax.set_ylabel(r"$\|\theta\|^2 / D$", fontsize=ft + 1)
    fig.tight_layout()
    save(fig, "fig2_noise_geometry_local.pdf", print_w_in=0.56 * 37.8 / 2.54,
         ticks_pt=ft, labels_pt=ft + 1)


def fig_tolerance_measured():
    """fig8 slot: the D=100 deviation sweep (sigma_c_data.npz), 6 chains per
    point, both samplers at the same input step. Knee = first noise level
    whose mean error exceeds 3x the no-noise floor, log-interpolated; the
    annotated ratio is THIS run's, nothing quoted from the paper."""
    d = dict(np.load(POSTER1 / "figures" / "sigma_c_data.npz"))
    s2 = d["sigmas"][1:] ** 2
    floor = max(d["rt_mean"][0], d["hmc_mean"][0])
    show = lambda v: np.maximum(v, floor)

    def knee(mean):
        i = np.nonzero(mean > 3 * floor)[0][0]
        f = (np.log(3 * floor) - np.log(mean[i - 1])) \
            / (np.log(mean[i]) - np.log(mean[i - 1]))
        return float(np.exp((1 - f) * np.log(s2[i - 1]) + f * np.log(s2[i])))

    k_h, k_r = knee(d["hmc_mean"][1:]), knee(d["rt_mean"][1:])
    ratio = k_r / k_h
    print(f"   fig8 knees (3x floor): HMC {k_h:.3g}, RT {k_r:.3g}, "
          f"ratio {ratio:.1f}x")

    ft8, fl8, fa8 = 17, 18, 17.5
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    ax.axhline(floor, color=SOFTGRAY, lw=1.2, ls=":")
    for mean, lo, hi, color, ls in (
            (d["hmc_mean"][1:], d["hmc_min"][1:], d["hmc_max"][1:], HMCRED,
             (0, (4, 2))),
            (d["rt_mean"][1:], d["rt_min"][1:], d["rt_max"][1:], RTGOLD, "-")):
        ax.fill_between(s2, show(lo), show(hi), color=color, alpha=0.16, lw=0)
        ax.loglog(s2, show(mean), ls=ls, color=color, lw=2.6, marker="o", ms=6)
    for k, color in ((k_h, HMCRED), (k_r, RTGOLD)):
        ax.axvline(k, color=color, lw=1.4, ls=":")
    top = 2.4 * float(d["hmc_max"].max())
    ax.annotate("", xy=(k_r, floor * 0.70), xytext=(k_h, floor * 0.70),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.text(3.0, floor * 0.44,
            rf"$\approx{ratio:.0f}\times$ more tolerated noise",
            fontsize=fa8, color=INK, ha="center", va="center",
            fontweight="bold")
    lab = dict(fontsize=fa8, fontweight="bold")
    ax.text(s2[0], 3.1, "HMC", color=HMCRED, ha="left", **lab)
    ax.text(s2[0], floor * 0.50, "ray tracing", color=DARKGOLD,
            ha="left", **lab)
    ax.text(s2[0], floor * 1.14, "no-noise floor", color=SOFTGRAY,
            ha="left", fontsize=fa8 - 2)
    ax.set_xlim(s2[0] * 0.8, s2[-1] * 1.6)
    ax.set_ylim(floor * 0.30, top)
    ax.tick_params(labelsize=ft8)
    ax.set_xlabel(r"gradient-noise variance $\sigma^2$", fontsize=fl8)
    ax.set_ylabel("error in sampled\n" r"$\ln L$ level (nats)", fontsize=fl8)
    fig.tight_layout()
    save(fig, "fig8_tolerance_measured.pdf", print_w_in=12.95, ticks_pt=ft8,
         labels_pt=fl8)


def fig_real_posterior():
    """fig10: prior-scaled weight-norm traces of the three exp7 Gaia chains
    (committed packs). The Gaussian chain of record stops at snapshot 62,600,
    before its recorded non-finite-window episode (same cut as the exp7
    analysis scripts); the Student-t and SGHMC chains warm-start from that
    chain's final state. Snapshot cadence 250 steps; thinned by slicing."""
    hp = np.load(EMP_TAB / "exp7h_pack3.npz")
    tp = np.load(EMP_TAB / "exp7t_pack_final.npz")
    sg = np.load(EMP_TAB / "exp7sg_fr3000_pack.npz")
    D7 = hp["members"].shape[1]
    w_g = hp["wnorm"][:62_600]
    assert hp["misfit"][:62_600][-2000:].max() < -185_000, "episode leaked"

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for w, col, lw in ((w_g, RTGOLD, 2.0), (tp["wnorm"], DARKGOLD, 1.8),
                       (sg["wnorm"], HMCRED, 1.8)):
        x = np.arange(len(w)) * 250 / 1e6
        thin = slice(None, None, max(1, len(w) // 2500))
        ax.plot(x[thin], w[thin] / 1e3, color=col, lw=lw)
    ax.axhline(D7 / 1e3, color=SOFTGRAY, lw=1.6, ls=(0, (4, 2)))

    lab = dict(fontsize=F_ANNO, fontweight="bold")
    ax.text(15.5, 22.9, "ray tracing, Gaussian likelihood", color=DARKGOLD,
            ha="right", va="top", **lab)
    ax.annotate("ray tracing, Student-t", xy=(5.6, 23.0), xytext=(10.2, 17.8),
                fontsize=F_ANNO, color=DARKGOLD, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=DARKGOLD, lw=1.2,
                                shrinkB=3))
    ax.annotate("SGHMC: still rising at its\n2M-step budget", xy=(2.08, 22.9),
                xytext=(0.1, 26.4), fontsize=F_ANNO, color=HMCRED,
                fontweight="bold", va="top",
                arrowprops=dict(arrowstyle="-", color=HMCRED, lw=1.2,
                                shrinkB=4))
    ax.text(15.4, 12.3, "prior shell, " + rf"$D$ = {D7:,}", color=SOFTGRAY,
            ha="right", fontsize=F_ANNO - 1)
    ax.set_xlim(-0.3, 16)
    ax.set_ylim(10, 27)
    ax.tick_params(labelsize=F_TICK)
    ax.set_xlabel("minibatch steps (millions)", fontsize=F_LABEL)
    ax.set_ylabel(r"$\|\theta/\sigma_{\mathrm{prior}}\|^2$  ($10^3$)",
                  fontsize=F_LABEL)
    fig.tight_layout()
    save(fig, "fig10_real_posterior.pdf", print_w_in=13.69)


if __name__ == "__main__":
    # Default builds only fig10: the other letters build the *_local
    # alternates. The poster's fig0/fig2/fig8/fig9/typical_set are the cloud
    # figures_v4 snapshot copied into artifacts/figures/, not built here.
    only = sys.argv[1] if len(sys.argv) > 1 else "R"
    if "E" in only:
        fig_typical_set()
    if "S" in only:
        fig_shell_diagnostic()
    if "B" in only:
        fig_data_bands()
    if "G" in only:
        fig_noise_geometry()
    if "T" in only:
        fig_tolerance_measured()
    if "R" in only:
        fig_real_posterior()
