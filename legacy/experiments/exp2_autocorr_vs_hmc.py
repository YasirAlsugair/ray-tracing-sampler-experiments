"""Experiment 2 -- Autocorrelation and ESS: ray tracing vs HMC.

For each target and each method we tune the step size to ~0.7 acceptance at every
trajectory length in a small grid, then report the BEST configuration per method
(by ESS per gradient evaluation) plus the full grid. Tuning both methods and
normalising by gradient work is what makes the comparison fair: a single fixed
config can park HMC on a trajectory-length resonance and flatter RT unfairly.

Outputs: results/tables/exp2_*.csv, results/figures/exp2_*.png
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

BURN = 500            # discard transient
N_STEPS = 3000
M = 48                # walkers
L_GRID = [10, 30]     # leapfrog steps (short vs long trajectory)
SEED = 7

TARGETS = [
    ("gaussian D=2", lambda: targets.gaussian(2)),
    ("gaussian D=10", lambda: targets.gaussian(10)),
    ("gaussian D=10 cond=10", lambda: targets.gaussian(10, cond=10.0)),
    ("rosenbrock D=2", lambda: targets.rosenbrock(2, b=20.0)),
    ("mixture D=2 sep=4", lambda: targets.gaussian_mixture(2, sep=4.0, K=2)),
]


def eval_config(kind, key, tgt, L, ss):
    k_init, k_run = jax.random.split(key)
    inits = tgt.sample(k_init, M) if tgt.sample is not None else \
        np.asarray(jax.random.normal(k_init, (M, tgt.D)))
    res = chains.run_many(kind, k_run, inits, tgt, step_size=ss,
                          n_steps=N_STEPS, n_leapfrog=L)
    X = chains.chain_steps_walkers_dims(res)[BURN:]
    summ = metrics.ess_summary(X)
    eff = metrics.efficiency(summ["ess_worst_coord"], res["n_grad_evals"], res["seconds"])
    return {
        "kind": kind, "L": L, "step_size": ss,
        "accept": float(res["accept"].mean()),
        "tau_worst": summ["tau_worst"], "tau_pca1": summ["tau_pca1"],
        "ess_worst": summ["ess_worst_coord"], "ess_pca1": summ["ess_pca1"],
        "ess_per_grad": eff["ess_per_grad_eval"], "ess_per_sec": eff["ess_per_sec"],
        "seconds": res["seconds"], "n_grad_evals": res["n_grad_evals"],
    }


def main():
    key = jax.random.PRNGKey(SEED)
    rows = []
    best_per = {}   # (target, kind) -> best row by ess_per_grad
    for tname, tfn in TARGETS:
        tgt = tfn()
        for kind in ("rt", "hmc"):
            for L in L_GRID:
                key, kt, ke = jax.random.split(key, 3)
                tune = chains.tune_step_size(kind, kt, tgt, n_leapfrog=L,
                                             target_accept=0.7)
                row = eval_config(kind, ke, tgt, L, tune["step_size"])
                row["target"] = tname
                rows.append(row)
                k = (tname, kind)
                if k not in best_per or row["ess_per_grad"] > best_per[k]["ess_per_grad"]:
                    best_per[k] = row
                print(f"{tname:24s} {kind:3s} L={L:3d} ss={tune['step_size']:.3f} "
                      f"acc={row['accept']:.2f} tau_worst={row['tau_worst']:7.1f} "
                      f"ess/grad={row['ess_per_grad']:.2e}")

    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    # full grid CSV
    cols = ["target", "kind", "L", "step_size", "accept", "tau_worst", "tau_pca1",
            "ess_worst", "ess_pca1", "ess_per_grad", "ess_per_sec", "seconds"]
    with open(os.path.join(TAB, "exp2_grid.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")

    # --- Figure A: best ESS/grad-eval, RT vs HMC, per target ---
    tnames = [t[0] for t in TARGETS]
    rt_vals = [best_per[(t, "rt")]["ess_per_grad"] for t in tnames]
    hmc_vals = [best_per[(t, "hmc")]["ess_per_grad"] for t in tnames]
    x = np.arange(len(tnames))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, rt_vals, 0.4, color=style.RT, label="Ray tracing")
    ax.bar(x + 0.2, hmc_vals, 0.4, color=style.HMC, label="HMC")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(tnames, rotation=20, ha="right")
    ax.set_ylabel("ESS per gradient evaluation\n(best tuned config, higher is better)")
    ax.set_title("Sampling efficiency: ray tracing vs HMC")
    for xi, (rv, hv) in enumerate(zip(rt_vals, hmc_vals)):
        ax.text(xi - 0.2, rv, f"{rv:.1e}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + 0.2, hv, f"{hv:.1e}", ha="center", va="bottom", fontsize=7)
        ax.text(xi, max(rv, hv) * 2.2, f"{rv/hv:.1f}x", ha="center", fontsize=9,
                color=style.TRUTH, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp2_ess_per_grad.png"))
    plt.close(fig)

    # --- Figure B: autocorrelation functions on gaussian D=10 ---
    tgt = targets.gaussian(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind in ("rt", "hmc"):
        br = best_per[("gaussian D=10", kind)]
        key, kt, ke = jax.random.split(key, 3)
        tune = chains.tune_step_size(kind, kt, tgt, n_leapfrog=br["L"], target_accept=0.7)
        k_init, k_run = jax.random.split(ke)
        inits = tgt.sample(k_init, M)
        res = chains.run_many(kind, k_run, inits, tgt, step_size=tune["step_size"],
                              n_steps=N_STEPS, n_leapfrog=br["L"])
        X = chains.chain_steps_walkers_dims(res)[BURN:]  # (T, M, D)
        # ACF of coordinate 0 averaged over walkers
        acf = np.mean([metrics.autocorr_func_1d(X[:, w, 0]) for w in range(M)], axis=0)
        ax.plot(acf[:120], color=style.COLOR[kind], lw=2, label=f"{style.LABEL[kind]} (tau={br['tau_worst']:.1f})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag (MCMC steps)")
    ax.set_ylabel("autocorrelation, coordinate 0")
    ax.set_title("Autocorrelation on a 10-D Gaussian")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp2_acf_gaussian10.png"))
    plt.close(fig)

    with open(os.path.join(TAB, "exp2_best.json"), "w") as f:
        json.dump({f"{t}|{k}": best_per[(t, k)] for t in tnames for k in ("rt", "hmc")},
                  f, indent=2)
    print("\nExp2 done. Figures + tables written.")


if __name__ == "__main__":
    main()
