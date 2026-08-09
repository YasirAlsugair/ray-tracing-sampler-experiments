"""SUDS-poster variant of figures/shell_measured.py: fonts scaled for a
0.68-column slot (in-file ~23-24 pt), wide-short canvas, label stack
respaced. Numbers and data flow identical to the original.

Run:  ~/.venvs/rts/bin/python poster_v2/shell_measured_suds.py \
          --out poster_v2/artifacts/figures/shell_measured_poster
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import stats

import arviz as az

# ---------------------------------------------------------------- constants

D: int = 100
N_CHAINS: int = 4
N_DRAWS: int = 60_000

# Poster palette. Linestyle, not color, carries exact-vs-stochastic (crimson dashed =
# exact gradients, crimson solid = stochastic), so the figure survives grayscale.
CRIMSON: str = "#C0405A"  # HMC family
STEEL: str = "#3B76B0"    # ray tracing
SHELL_GRAY: str = "#C9CDD4"  # analytic reference / confidence band
GRAY_TEXT: str = "#8A8F98"   # direct labels on gray elements
NEUTRAL_EDGE: str = "#B0B6C0"  # inset border, reference outline

BINS: NDArray[np.float64] = np.linspace(0.55, 4.75, 91)
DASH_EXACT: tuple[int, tuple[int, int]] = (0, (4, 2))

BAND_TRIALS: int = 1_000
BAND_LEVEL: float = 0.95
ECDF_GRID: NDArray[np.float64] = np.linspace(0.0, 1.0, 101)

SAMPLERS: tuple[str, str, str] = ("hmc_exact", "rt_sto", "hmc_sto")
LABELS: dict[str, str] = {
    "hmc_exact": "HMC, exact gradients",
    "rt_sto": "ray tracing, σ = 20",
    "hmc_sto": "stochastic-gradient HMC, σ = 20",
}
# Longest alias wins, so "rt_sto_r2" matches rt_sto, not hmc_sto via bare "sto".
ALIASES: dict[str, tuple[str, ...]] = {
    "hmc_exact": ("hmc_exact", "exact_hmc", "exact"),
    "rt_sto": ("rt_sto", "rts", "ray", "rt"),
    "hmc_sto": ("hmc_sto", "sto_hmc", "sghmc", "stochastic", "sto"),
}

FIG_DIR: Path = Path(__file__).resolve().parent
REPO: Path = FIG_DIR.parent

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Fira Sans", "Source Sans 3", "TeX Gyre Heros", "DejaVu Sans"],
        "font.size": 23,
        "xtick.labelsize": 23,
        "axes.labelsize": 24,
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "pdf.fonttype": 42,
    }
)

# ---------------------------------------------------------------- data loading


def _to_radii(arr: NDArray[np.floating]) -> NDArray[np.float64]:
    """Reduce a saved chain array to radii r = ||theta||^2 / D, shape (chains, draws)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3:  # (chains, draws, D) parameter draws
        a = np.sum(a**2, axis=-1) / a.shape[-1]
    if a.ndim != 2:
        raise ValueError(f"expected 2D or 3D chain array, got shape {a.shape}")
    if np.nanmedian(a) > 10.0:  # ||theta||^2 was saved without the /D normalization
        a = a / D
    if a.shape[1] % 2 == 1:  # initial state recorded alongside the draws
        a = a[:, 1:]
    return a


def _match_sampler(name: str) -> str | None:
    """Match an array key or file stem to a sampler; longest alias wins."""
    low = name.lower()
    best: tuple[int, str] | None = None
    for sampler, aliases in ALIASES.items():
        for alias in aliases:
            if alias in low and (best is None or len(alias) > best[0]):
                best = (len(alias), sampler)
    return best[1] if best else None


def load_radii(data_dir: Path | None) -> tuple[dict[str, NDArray[np.float64]], str]:
    """Scan for saved chains; return (radii per sampler, provenance flag "chains")."""
    if data_dir is not None:
        roots = [data_dir]
    else:
        roots = [
            REPO / "results",
            REPO / "experiments",
            REPO / "poster_v2" / "artifacts",
            REPO / "poster" / "figures",
        ]
    radii: dict[str, NDArray[np.float64]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.np[yz]")):
            if path.suffix == ".npy":
                sampler = _match_sampler(path.stem)
                if sampler and sampler not in radii:
                    radii[sampler] = _to_radii(np.load(path))
                continue
            with np.load(path) as archive:
                for key in archive.files:
                    arr = archive[key]
                    if arr.ndim not in (2, 3):
                        continue
                    sampler = _match_sampler(key)
                    if sampler and sampler not in radii:
                        radii[sampler] = _to_radii(arr)
    missing = [s for s in SAMPLERS if s not in radii]
    if missing:
        raise FileNotFoundError(
            f"no saved chains found for {missing} under {[str(r) for r in roots]}; "
            "run with --synthetic to iterate on layout"
        )
    return {s: radii[s] for s in SAMPLERS}, "chains"


def synthetic_radii(seed: int) -> tuple[dict[str, NDArray[np.float64]], str]:
    """Stand-in radii from the right equilibria; provenance flag "synthetic"."""
    rng = np.random.default_rng(seed)
    shape = (N_CHAINS, N_DRAWS)
    exact = rng.chisquare(D, shape) / D
    rts = rng.chisquare(D, shape) / D + 0.03  # same shape, shifted to mean 1.03
    # Broad right-skewed bump centered at 2.9; small per-chain offsets stand in for the
    # between-chain wobble of the real stochastic chains (puts split-Rhat near 1.01).
    k = 30.0
    sto = rng.gamma(k, 2.9 / k, shape) + np.array([-0.09, -0.03, 0.03, 0.09])[:, None]
    return {"hmc_exact": exact, "rt_sto": rts, "hmc_sto": sto}, "synthetic"


# ---------------------------------------------------------------- statistics


def rank_normalized_rhat(radii: NDArray[np.float64]) -> float:
    """Rank-normalized split-Rhat (Vehtari et al. 2021) on (chains, draws) radii."""
    try:
        return float(az.rhat(radii, method="rank"))
    except TypeError:  # arviz without the method keyword
        return float(az.rhat(radii))


def bulk_ess(radii: NDArray[np.float64]) -> float:
    """Bulk effective sample size on (chains, draws) radii."""
    try:
        return float(az.ess(radii, method="bulk"))
    except TypeError:
        return float(az.ess(radii))


def shell_pdf(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Density of r = X/D for X ~ chi^2_D (change of variables)."""
    return D * stats.chi2(df=D).pdf(r * D)


def simultaneous_band(
    n: int, grid: NDArray[np.float64], rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """95% simultaneous confidence band for ECDF(u) - u under uniformity.

    The installed ArviZ (1.x) has no plot_ecdf, so this implements Sailynoja,
    Buerkner & Vehtari (2022) directly: simulate BAND_TRIALS iid-uniform samples of
    size n and binary-search the pointwise level gamma until BAND_LEVEL of the
    simulated ECDFs lie entirely inside the Binomial(n, t) quantile envelope.
    """
    sims = np.sort(rng.random((BAND_TRIALS, n)), axis=1)
    ecdfs = np.stack([np.searchsorted(row, grid, side="right") for row in sims]) / n
    lo_g, hi_g = 1e-4, 0.5
    for _ in range(40):
        gamma = 0.5 * (lo_g + hi_g)
        lo = stats.binom.ppf(gamma / 2, n, grid) / n
        hi = stats.binom.ppf(1 - gamma / 2, n, grid) / n
        coverage = np.mean(np.all((ecdfs >= lo) & (ecdfs <= hi), axis=1))
        if coverage >= BAND_LEVEL:
            lo_g = gamma
        else:
            hi_g = gamma
    gamma = lo_g
    lo = stats.binom.ppf(gamma / 2, n, grid) / n - grid
    hi = stats.binom.ppf(1 - gamma / 2, n, grid) / n - grid
    return lo, hi


# ---------------------------------------------------------------- panels


def _stairs_trimmed(ax: Axes, dens: NDArray[np.float64], edges: NDArray[np.float64], **kw) -> None:
    """Step outline restricted to the histogram's visible support, so empty and
    sub-pixel edge bins do not draw a colored line along y=0 across the axis."""
    nz = np.nonzero(dens > 0.004 * dens.max())[0]
    if nz.size:
        ax.stairs(dens[nz[0] : nz[-1] + 1], edges[nz[0] : nz[-1] + 2], **kw)


def _draw_shell_cluster(ax: Axes, dens: dict[str, NDArray[np.float64]], grid: NDArray[np.float64]) -> None:
    """The analytic reference plus the two on-shell histograms (shared by main + inset)."""
    ax.fill_between(grid, shell_pdf(grid), 0, color=SHELL_GRAY, alpha=0.75, lw=0, zorder=1)
    ax.plot(grid, shell_pdf(grid), color="#A9AFBA", lw=1.3, zorder=2.6)
    _stairs_trimmed(ax, dens["rt_sto"], BINS, fill=True, color=STEEL, alpha=0.35, lw=0, zorder=2)
    _stairs_trimmed(ax, dens["rt_sto"], BINS, baseline=None, color=STEEL, lw=1.8, zorder=3)
    _stairs_trimmed(
        ax, dens["hmc_exact"], BINS, baseline=None, color=CRIMSON, lw=1.8, ls=DASH_EXACT, zorder=4
    )


def _add_inset(ax: Axes, radii: dict[str, NDArray[np.float64]], dens: dict[str, NDArray[np.float64]]) -> None:
    """Zoom on the left cluster: the 1.00 vs 1.03 mean shift, no connector lines."""
    axins = ax.inset_axes([0.40, 0.55, 0.24, 0.38])
    _draw_shell_cluster(axins, dens, np.linspace(0.85, 1.25, 400))
    m_exact = float(radii["hmc_exact"].mean())
    m_rts = float(radii["rt_sto"].mean())
    axins.axvline(m_exact, color=CRIMSON, lw=0.9, ls=":", zorder=5)
    axins.axvline(m_rts, color=STEEL, lw=0.9, ls=":", zorder=5)
    axins.text(m_exact - 0.012, 3.02, f"{m_exact:.2f}", color=CRIMSON, fontsize=7, ha="right")
    axins.text(m_rts + 0.012, 3.02, f"{m_rts:.2f}", color=STEEL, fontsize=7, ha="left")
    axins.set_xlim(0.85, 1.25)
    axins.set_ylim(0, 3.45)
    axins.set_xticks([0.9, 1.0, 1.1, 1.2])
    axins.set_yticks([])
    axins.tick_params(labelsize=7, length=2, color=NEUTRAL_EDGE)
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(NEUTRAL_EDGE)
        spine.set_linewidth(0.8)


def draw_main(
    ax: Axes,
    radii: dict[str, NDArray[np.float64]],
    rhat_sto: float,
    inset: bool,
) -> None:
    dens = {s: np.histogram(radii[s].ravel(), bins=BINS, density=True)[0] for s in SAMPLERS}
    _draw_shell_cluster(ax, dens, np.linspace(0.5, 4.9, 800))
    _stairs_trimmed(ax, dens["hmc_sto"], BINS, fill=True, color=CRIMSON, alpha=0.35, lw=0, zorder=2)
    _stairs_trimmed(ax, dens["hmc_sto"], BINS, baseline=None, color=CRIMSON, lw=1.8, zorder=3)

    # The three left-cluster curves are nearly coincident, so their labels stack in the
    # empty space off the cluster's right flank, outermost curve first.
    ax.text(1.28, 2.55, LABELS["rt_sto"], color=STEEL, fontsize=24, zorder=5)
    ax.text(1.28, 1.98, LABELS["hmc_exact"], color=CRIMSON, fontsize=24, zorder=5)
    ax.text(1.28, 1.41, "the shell  $\\chi^2_D/D$", color=GRAY_TEXT, fontsize=23, zorder=5)

    peak = float(dens["hmc_sto"].max())
    mode = float(0.5 * (BINS[:-1] + BINS[1:])[int(np.argmax(dens["hmc_sto"]))])
    # Two lines so the label clears the right frame edge from its flank anchor.
    ax.annotate(
        LABELS["hmc_sto"].replace(", ", ",\n"), xy=(3.45, 0.55 * peak), xytext=(6, 8),
        textcoords="offset points", ha="left", va="bottom", color=CRIMSON,
        fontsize=24, zorder=5,
    )
    ax.text(
        mode, 0.22 * peak, f"split $\\hat{{R}}$ = {rhat_sto:.2f}",
        color="#8F2F44", fontsize=22, ha="center", zorder=5,
    )

    ax.set_xlim(0.52, 4.85)
    ax.margins(y=0.05)
    ax.set_ylim(bottom=0)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel(r"$\|\theta\|^2/D$")
    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)

    if inset:
        _add_inset(ax, radii, dens)


def draw_ecdf(
    ax: Axes, radii: dict[str, NDArray[np.float64]], rng: np.random.Generator
) -> None:
    cdf = stats.chi2(df=D).cdf
    # Thin each sampler by its own bulk ESS so the retained draws are approximately
    # independent, which is what the simultaneous band assumes.
    thinned: dict[str, NDArray[np.float64]] = {}
    for s in SAMPLERS:
        step = max(1, int(np.floor(radii[s].size / bulk_ess(radii[s]))))
        thinned[s] = radii[s][:, ::step].ravel()
        print(f"ecdf thinning  {LABELS[s]:<34s} every {step:>4d}th draw -> n = {thinned[s].size}")

    # One band at the smallest thinned size: the widest (most conservative) of the
    # three fair bands, so any curve outside it is outside its own fair band too.
    n_band = min(v.size for v in thinned.values())
    lo, hi = simultaneous_band(n_band, ECDF_GRID, rng)
    ax.fill_between(ECDF_GRID, lo, hi, color=SHELL_GRAY, alpha=0.55, lw=0, zorder=1)
    ax.axhline(0, color=GRAY_TEXT, lw=0.8, zorder=2)

    styles = {
        "hmc_exact": (CRIMSON, DASH_EXACT),
        "rt_sto": (STEEL, "-"),
        "hmc_sto": (CRIMSON, "-"),
    }
    diffs: dict[str, NDArray[np.float64]] = {}
    for s in SAMPLERS:
        pit = np.sort(cdf(thinned[s] * D))
        diffs[s] = np.searchsorted(pit, ECDF_GRID, side="right") / pit.size - ECDF_GRID
        color, ls = styles[s]
        ax.plot(ECDF_GRID, diffs[s], color=color, ls=ls, lw=1.8, zorder=3)
        print(f"max |ECDF - uniform|  {LABELS[s]:<34s} {np.abs(diffs[s]).max():.3f}")

    # Stochastic HMC tracks -u and exits through the floor; annotate at the clip point.
    below = np.nonzero(diffs["hmc_sto"] < -0.12)[0]
    u_clip = float(ECDF_GRID[below[0]]) if below.size else 0.12
    ax.text(
        u_clip + 0.02, -0.106, "→ −1: all mass off the shell",
        color=CRIMSON, fontsize=8.5, ha="left", zorder=5,
    )

    ax.text(0.15, 0.075, "95% simultaneous band", color=GRAY_TEXT, fontsize=9, zorder=5)
    ax.text(0.70, 0.062, LABELS["hmc_exact"], color=CRIMSON, fontsize=9, zorder=5)
    ax.text(0.78, -0.100, "ray tracing", color=STEEL, fontsize=9, va="center", zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 0.12)
    ax.set_yticks([-0.1, 0, 0.1])
    ax.set_xlabel(r"PIT under the shell,  $F_{\chi^2_D}(\|\theta\|^2)$")
    ax.set_ylabel("ECDF $-$ uniform")


# ---------------------------------------------------------------- driver


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=None, help="directory with saved chains")
    parser.add_argument("--synthetic", action="store_true", help="use watermarked stand-in radii")
    parser.add_argument("--inset", action="store_true", help="zoom inset on the 1.00 vs 1.03 shift")
    parser.add_argument("--ecdf", action="store_true", help="add the delta-ECDF panel")
    parser.add_argument("--seed", type=int, default=0, help="rng seed (synthetic data, band)")
    parser.add_argument("--out", type=Path, default=FIG_DIR / "shell_measured", help="output stem")
    parser.add_argument("--transparent", action="store_true", help="transparent background")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    radii, provenance = synthetic_radii(args.seed) if args.synthetic else load_radii(args.data_dir)

    rhats = {s: rank_normalized_rhat(radii[s]) for s in SAMPLERS}
    esses = {s: bulk_ess(radii[s]) for s in SAMPLERS}
    for s in SAMPLERS:
        print(f"{LABELS[s]:<34s} split-Rhat = {rhats[s]:.4f}   bulk ESS = {esses[s]:7.0f}   "
              f"mean r = {radii[s].mean():.3f}")
    print(
        f"caption ({provenance}): On a D=100 standard Gaussian ({N_CHAINS} chains x "
        f"{N_DRAWS:,} draws per sampler), exact-gradient HMC and ray tracing at σ = 20 "
        f"concentrate on the shell (mean ‖θ‖²/D = {radii['hmc_exact'].mean():.2f} and "
        f"{radii['rt_sto'].mean():.2f}), while stochastic-gradient HMC at the same noise "
        f"equilibrates at {radii['hmc_sto'].mean():.2f}, three times off the shell, with "
        f"rank-normalized split R-hat = {rhats['hmc_sto']:.2f} (bulk ESS = "
        f"{esses['hmc_sto']:.0f}): the chains agree with each other, not with the truth."
    )

    if args.ecdf:
        fig, (ax_main, ax_ecdf) = plt.subplots(
            2, 1, figsize=(7.2, 5.6), height_ratios=[1.15, 1], constrained_layout=True
        )
        draw_main(ax_main, radii, rhats["hmc_sto"], inset=args.inset)
        draw_ecdf(ax_ecdf, radii, rng)
    else:
        fig, ax_main = plt.subplots(figsize=(10.5, 3.1), constrained_layout=True)
        draw_main(ax_main, radii, rhats["hmc_sto"], inset=args.inset)

    if provenance == "synthetic":
        fig.text(
            0.99, 0.01, "SYNTHETIC — layout only",
            ha="right", fontsize=7, color="#C0405A", alpha=0.6,
        )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), transparent=args.transparent)
    fig.savefig(out.with_suffix(".png"), dpi=600, transparent=args.transparent)
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
