"""Experiment 3 -- Mode coverage.

On multimodal targets we measure mode-occupancy fractions and inter-mode switch
rate against the analytic truth, then run Ricardo's test: vary the walker
initialisation (zero-mean Gaussian with increasing variance s) and sweep the mode
separation, asking whether ray tracing reaches the far modes it otherwise misses.

Targets: 3-mode Gaussian mixture (equal weights -> truth occupancy 1/3 each) and,
as the honest boundary case, disjoint L=0 islands where the released sampler has
no re-emission and is expected to stay trapped.

Step size is tuned once per method on the sep=6 mixture and reused across cells
(the within-mode scale sigma=1 is fixed, so the tuned step is separation-stable).

Outputs: results/tables/exp3_*.csv, results/figures/exp3_*.png
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rts.config  # noqa
import numpy as np
import jax
from rts import targets, chains, metrics
from plots import style
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "results", "figures")
TAB = os.path.join(HERE, "..", "results", "tables")

L_LEAP = 20
M = 48
N_STEPS = 2500
BURN = 800
SEED = 23
S_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SEP_GRID = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]


def init_from_gaussian(key, M, D, s):
    return np.asarray(jax.random.normal(key, (M, D))) * s


def coverage_stats(X, centers, home=0):
    cov_visit, n_modes = metrics.far_mode_coverage(X, centers, home=home)
    occ = metrics.mode_occupancy(X, centers)
    far_occ = float(1.0 - occ[home])          # mass outside the home mode
    sw, _ = metrics.switch_rate(X, centers)
    return {"coverage": cov_visit, "n_modes_mean": n_modes, "far_occ": far_occ,
            "switch_rate": sw, "occ": occ}


def main():
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    key = jax.random.PRNGKey(SEED)

    # tune once per method on sep=6
    base = targets.gaussian_mixture(2, sep=6.0, K=3)
    tuned = {}
    for kind in ("rt", "hmc"):
        key, kt = jax.random.split(key)
        tuned[kind] = chains.tune_step_size(kind, kt, base, n_leapfrog=L_LEAP,
                                            target_accept=0.7)["step_size"]
    print("tuned step sizes:", tuned)

    # ---------- (1) occupancy + switch vs truth, init from truth ----------
    occ_rows = {}
    for kind in ("rt", "hmc"):
        key, k_init, k_run = jax.random.split(key, 3)
        inits = base.sample(k_init, M)  # spread across all 3 modes
        res = chains.run_many(kind, k_run, inits, base, step_size=tuned[kind],
                              n_steps=N_STEPS, n_leapfrog=L_LEAP)
        X = chains.chain_steps_walkers_dims(res)[BURN:]
        st = coverage_stats(X, base.centers)
        occ_rows[kind] = st
        print(f"[occ] {kind:3s} occ={np.round(st['occ'],3)} switch={st['switch_rate']:.4f}")

    # ---------- (2) init-variance sweep, RT vs HMC ----------
    sweep_s = {"rt": [], "hmc": []}
    for kind in ("rt", "hmc"):
        for s in S_GRID:
            key, k_init, k_run = jax.random.split(key, 3)
            inits = init_from_gaussian(k_init, M, base.D, s)
            res = chains.run_many(kind, k_run, inits, base, step_size=tuned[kind],
                                  n_steps=N_STEPS, n_leapfrog=L_LEAP)
            X = chains.chain_steps_walkers_dims(res)[BURN:]
            st = coverage_stats(X, base.centers)
            st["s"] = s
            sweep_s[kind].append(st)
            print(f"[init s={s:5.1f}] {kind:3s} coverage={st['coverage']:.2f} far_occ={st['far_occ']:.2f}")

    # ---------- (3) separation sweep, init at home (s=0.5) ----------
    sweep_sep = {"rt": [], "hmc": []}
    for kind in ("rt", "hmc"):
        for sep in SEP_GRID:
            tgt = targets.gaussian_mixture(2, sep=sep, K=3)
            key, k_init, k_run = jax.random.split(key, 3)
            inits = init_from_gaussian(k_init, M, tgt.D, 0.5)  # all near home
            res = chains.run_many(kind, k_run, inits, tgt, step_size=tuned[kind],
                                  n_steps=N_STEPS, n_leapfrog=L_LEAP)
            X = chains.chain_steps_walkers_dims(res)[BURN:]
            st = coverage_stats(X, tgt.centers)
            st["sep"] = sep
            sweep_sep[kind].append(st)
            print(f"[sep={sep:4.1f}] {kind:3s} coverage={st['coverage']:.2f} switch={st['switch_rate']:.4f}")

    # ---------- (4) heatmap coverage over (s x sep), RT ----------
    heat = np.zeros((len(S_GRID), len(SEP_GRID)))
    for i, s in enumerate(S_GRID):
        for j, sep in enumerate(SEP_GRID):
            tgt = targets.gaussian_mixture(2, sep=sep, K=3)
            key, k_init, k_run = jax.random.split(key, 3)
            inits = init_from_gaussian(k_init, M, tgt.D, s)
            res = chains.run_many("rt", k_run, inits, tgt, step_size=tuned["rt"],
                                  n_steps=2000, n_leapfrog=L_LEAP)
            X = chains.chain_steps_walkers_dims(res)[BURN:]
            heat[i, j] = metrics.far_mode_coverage(X, tgt.centers)[0]

    # ---------- (5) disjoint islands, init at home ----------
    isl = targets.disjoint_islands(2, sep=12.0, K=3, R=3.0)
    island_rows = {}
    for kind in ("rt", "hmc"):
        key, kt, k_init, k_run = jax.random.split(key, 4)
        ss = chains.tune_step_size(kind, kt, isl, n_leapfrog=L_LEAP, target_accept=0.7)["step_size"]
        inits = isl.sample(k_init, M) * 0 + isl.centers[0]  # all start in home ball
        inits = inits + np.asarray(jax.random.normal(k_init, (M, isl.D))) * 0.3
        res = chains.run_many(kind, k_run, inits, isl, step_size=ss, n_steps=N_STEPS, n_leapfrog=L_LEAP)
        X = chains.chain_steps_walkers_dims(res)[BURN:]
        occ = metrics.mode_occupancy(X, isl.centers)
        sw, _ = metrics.switch_rate(X, isl.centers)
        island_rows[kind] = {"occ": occ, "switch_rate": sw, "step_size": ss}
        print(f"[island] {kind:3s} occ={np.round(occ,3)} switch={sw:.4f}")

    # ===================== FIGURES =====================
    # Fig 1: occupancy bars + switch
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    K = base.centers.shape[0]
    x = np.arange(K)
    a1.bar(x - 0.25, occ_rows["rt"]["occ"], 0.25, color=style.RT, label="Ray tracing")
    a1.bar(x, occ_rows["hmc"]["occ"], 0.25, color=style.HMC, label="HMC")
    a1.bar(x + 0.25, base.analytic_occupancy, 0.25, color=style.TRUTH, label="truth")
    a1.set_xticks(x); a1.set_xticklabels([f"mode {i}" for i in range(K)])
    a1.set_ylabel("occupancy fraction"); a1.set_title("Mode occupancy vs truth (sep=6)")
    a1.legend(fontsize=8)
    a2.bar(["Ray tracing", "HMC"], [occ_rows["rt"]["switch_rate"], occ_rows["hmc"]["switch_rate"]],
           color=[style.RT, style.HMC])
    a2.set_ylabel("inter-mode switch rate"); a2.set_title("How often a walker changes mode")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp3_occupancy.png")); plt.close(fig)

    # Fig 2: init-variance sweep + separation sweep
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for kind in ("rt", "hmc"):
        b1.plot(S_GRID, [r["far_occ"] for r in sweep_s[kind]], "o-",
                color=style.COLOR[kind], label=style.LABEL[kind])
    b1.axhline(2/3, color=style.TRUTH, ls="--", label="truth (2/3 in far modes)")
    b1.set_xscale("log", base=2); b1.set_xlabel("init std s   (walkers ~ N(0, s^2 I))")
    b1.set_ylabel("steady-state mass in far modes"); b1.set_title("Ricardo's test: init variance vs far-mode mass (sep=6)")
    b1.legend(fontsize=8)
    for kind in ("rt", "hmc"):
        b2.plot(SEP_GRID, [r["coverage"] for r in sweep_sep[kind]], "o-",
                color=style.COLOR[kind], label=style.LABEL[kind])
    b2.set_xlabel("mode separation"); b2.set_ylabel("fraction of walkers reaching a far mode")
    b2.set_title("Starting at home: how far can a ray reach? (init s=0.5)")
    b2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp3_sweeps.png")); plt.close(fig)

    # Fig 3: heatmap + islands
    fig, (c1, c2) = plt.subplots(1, 2, figsize=(12, 4.3))
    im = c1.imshow(heat, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    c1.set_xticks(range(len(SEP_GRID))); c1.set_xticklabels([f"{s:g}" for s in SEP_GRID])
    c1.set_yticks(range(len(S_GRID))); c1.set_yticklabels([f"{s:g}" for s in S_GRID])
    c1.set_xlabel("mode separation"); c1.set_ylabel("init std s")
    c1.set_title("Ray tracing far-mode coverage")
    fig.colorbar(im, ax=c1, label="fraction reaching a far mode")
    Ki = isl.centers.shape[0]
    xi = np.arange(Ki)
    c2.bar(xi - 0.2, island_rows["rt"]["occ"], 0.4, color=style.RT, label="Ray tracing")
    c2.bar(xi + 0.2, island_rows["hmc"]["occ"], 0.4, color=style.HMC, label="HMC")
    c2.set_xticks(xi); c2.set_xticklabels([f"island {i}" for i in range(Ki)])
    c2.set_ylabel("occupancy"); c2.set_title("Disjoint L=0 islands: both trapped at home\n(released sampler has no re-emission)")
    c2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp3_heatmap_islands.png")); plt.close(fig)

    # tables
    out = {
        "occupancy": {k: {"occ": list(map(float, occ_rows[k]["occ"])),
                          "switch_rate": occ_rows[k]["switch_rate"]} for k in occ_rows},
        "init_sweep": {k: [{kk: (float(v) if not isinstance(v, np.ndarray) else list(map(float, v)))
                            for kk, v in r.items()} for r in sweep_s[k]] for k in sweep_s},
        "sep_sweep": {k: [{kk: (float(v) if not isinstance(v, np.ndarray) else list(map(float, v)))
                           for kk, v in r.items()} for r in sweep_sep[k]] for k in sweep_sep},
        "heatmap": heat.tolist(), "S_GRID": S_GRID, "SEP_GRID": SEP_GRID,
        "islands": {k: {"occ": list(map(float, island_rows[k]["occ"])),
                        "switch_rate": island_rows[k]["switch_rate"]} for k in island_rows},
    }
    with open(os.path.join(TAB, "exp3_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nExp3 done.")


if __name__ == "__main__":
    main()
