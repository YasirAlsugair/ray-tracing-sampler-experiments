"""Experiment 4 -- Learning the proposal (research-direction prototype).

The simplest honest version of "learn W / the emission rate to target mode
coverage and autocorrelation". We parameterise two knobs the released code
exposes (via our fork):
  c = tempering exponent of a weight W = L^c  (bends the ray along L^(1-c)),
  r = refresh_rate of the Ornstein-Uhlenbeck momentum refresh (the emission-like knob).
and we LEARN (c, r) on a 2-mode target by maximising

  J(c, r) = 2 * mode_balance + log10(ESS_slowmode + 1)

which rewards exactly the two axes Ricardo named: mode coverage (balance) and low
autocorrelation (ESS of the slow mode = the mode indicator). We map J on a grid
(the landscape) and run CMA-ES to show the knob being learned. Correctness is
safe: tempering-W invariance was demonstrated in Experiment 1 and refresh leaves
the target unchanged; we re-confirm the learned config recovers ~0.5/0.5 mass.

Outputs: results/tables/exp4_*.json, results/figures/exp4_learn.png
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rts.config  # noqa
import numpy as np
import jax
import cma
from rts import targets, chains, metrics
from plots import style
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "results", "figures")
TAB = os.path.join(HERE, "..", "results", "tables")

L_LEAP = 20
SEP = 8.0    # hard enough that baseline RT from a home start is mode-stuck
TGT = None
STEP = 1.0
EVAL_KEY = jax.random.PRNGKey(99)   # fixed -> deterministic objective surface
C_GRID = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9])
R_GRID = np.array([0.0, 0.25, 0.75, 1.5, 3.0])


def make_logW(c):
    if c <= 0:
        return None
    lp = TGT.log_prob
    return lambda x: c * lp(x)


def evaluate(c, r, key, M=40, n_steps=1500, burn=400):
    logW = make_logW(c)
    k_init, k_run = jax.random.split(key)
    inits = np.asarray(jax.random.normal(k_init, (M, TGT.D))) * 0.5  # all near home
    res = chains.run_many("rt", k_run, inits, TGT, step_size=STEP, n_steps=n_steps,
                          n_leapfrog=L_LEAP, refresh_rate=float(r), log_W_fn=logW)
    X = chains.chain_steps_walkers_dims(res)[burn:]
    occ = metrics.mode_occupancy(X, TGT.centers)
    balance = float(1 - abs(occ[0] - occ[1]))
    sw, _ = metrics.switch_rate(X, TGT.centers)
    summ = metrics.ess_summary(X)
    J = 2 * balance + np.log10(summ["ess_pca1"] + 1)
    return {"c": float(c), "r": float(r), "J": float(J), "balance": balance,
            "switch_rate": sw, "ess_pca1": summ["ess_pca1"],
            "occ0": float(occ[0]), "occ1": float(occ[1])}


def main():
    global TGT, STEP
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    TGT = targets.gaussian_mixture(2, sep=SEP, K=2)
    STEP = chains.tune_step_size("rt", jax.random.PRNGKey(1), TGT, n_leapfrog=L_LEAP,
                                 target_accept=0.7)["step_size"]
    print("tuned step size:", STEP)

    # ---- landscape grid ----
    J_grid = np.zeros((len(C_GRID), len(R_GRID)))
    for i, c in enumerate(C_GRID):
        for j, r in enumerate(R_GRID):
            J_grid[i, j] = evaluate(c, r, EVAL_KEY)["J"]
        print(f"grid c={c:.1f} done")

    # ---- CMA-ES learning ----
    es = cma.CMAEvolutionStrategy([0.3, 0.5], 0.3, {
        "bounds": [[0.0, 0.0], [0.9, 3.0]], "popsize": 6, "maxiter": 10,
        "verbose": -9, "seed": 1})
    traj = []
    while not es.stop():
        sols = es.ask()
        fits = []
        for s in sols:
            res = evaluate(s[0], s[1], EVAL_KEY)
            fits.append(-res["J"])
        es.tell(sols, fits)
        bx = es.best.x
        traj.append((float(bx[0]), float(bx[1]), float(-es.best.f)))
    learned_c, learned_r = float(es.best.x[0]), float(es.best.x[1])
    print(f"learned: c={learned_c:.3f} r={learned_r:.3f} J={-es.best.f:.3f}")

    # ---- baseline vs learned, longer chains for clean before/after ----
    base_eval = evaluate(0.0, 0.0, EVAL_KEY, M=64, n_steps=4000, burn=800)
    learn_eval = evaluate(learned_c, learned_r, EVAL_KEY, M=64, n_steps=4000, burn=800)
    print("baseline:", {k: round(base_eval[k], 3) for k in ("balance", "switch_rate", "ess_pca1")})
    print("learned :", {k: round(learn_eval[k], 3) for k in ("balance", "switch_rate", "ess_pca1")})

    # sample clouds for scatter
    def cloud(c, r):
        logW = make_logW(c)
        k_init, k_run = jax.random.split(jax.random.PRNGKey(7))
        inits = np.asarray(jax.random.normal(k_init, (64, TGT.D))) * 0.5
        res = chains.run_many("rt", k_run, inits, TGT, step_size=STEP, n_steps=4000,
                              n_leapfrog=L_LEAP, refresh_rate=float(r), log_W_fn=logW)
        X = chains.chain_steps_walkers_dims(res)[800:]
        return X.reshape(-1, TGT.D)
    base_cloud = cloud(0.0, 0.0)
    learn_cloud = cloud(learned_c, learned_r)

    # ===================== FIGURE =====================
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.5))
    # (a) landscape + CMA trajectory
    im = a.imshow(J_grid, origin="lower", aspect="auto", cmap="viridis",
                  extent=[R_GRID[0], R_GRID[-1], C_GRID[0], C_GRID[-1]])
    tr = np.array(traj)
    a.plot(tr[:, 1], tr[:, 0], "-o", color="white", ms=4, lw=1.5, label="CMA-ES best path")
    a.plot(learned_r, learned_c, "*", color=style.RT, ms=20, markeredgecolor="k", label="learned")
    a.plot(0, 0, "s", color=style.HMC, ms=9, markeredgecolor="k", label="baseline W=1, r=0")
    a.set_xlabel("refresh rate r"); a.set_ylabel("tempering exponent c   (W = L^c)")
    a.set_title("Objective landscape J(c, r)\n(coverage + slow-mode ESS)")
    a.legend(fontsize=7, loc="upper right")
    fig.colorbar(im, ax=a, label="J (higher better)")
    # (b) before/after scatter
    b.scatter(base_cloud[:, 0], base_cloud[:, 1], s=2, alpha=0.25, color=style.HMC,
              label=f"baseline (balance {base_eval['balance']:.2f})")
    b.scatter(learn_cloud[:, 0] + 0.0, learn_cloud[:, 1], s=2, alpha=0.12, color=style.RT,
              label=f"learned (balance {learn_eval['balance']:.2f})")
    for k in range(2):
        b.plot(TGT.centers[k, 0], TGT.centers[k, 1], "k+", ms=12)
    b.set_xlabel("x_0"); b.set_ylabel("x_1"); b.set_title("Samples: baseline vs learned proposal")
    b.legend(fontsize=8, markerscale=4)
    # (c) metric bars
    labels = ["mode balance", "switch rate x10", "log10 ESS(slow)"]
    base_vals = [base_eval["balance"], base_eval["switch_rate"] * 10, np.log10(base_eval["ess_pca1"] + 1)]
    learn_vals = [learn_eval["balance"], learn_eval["switch_rate"] * 10, np.log10(learn_eval["ess_pca1"] + 1)]
    x = np.arange(3)
    c.bar(x - 0.2, base_vals, 0.4, color=style.HMC, label="baseline W=1, r=0")
    c.bar(x + 0.2, learn_vals, 0.4, color=style.RT, label=f"learned c={learned_c:.2f}, r={learned_r:.2f}")
    c.set_xticks(x); c.set_xticklabels(labels, rotation=12)
    c.set_title("What learning the proposal buys"); c.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp4_learn.png")); plt.close(fig)

    with open(os.path.join(TAB, "exp4_summary.json"), "w") as f:
        json.dump({"tuned_step": STEP, "learned_c": learned_c, "learned_r": learned_r,
                   "baseline": base_eval, "learned": learn_eval,
                   "trajectory": traj, "C_GRID": C_GRID.tolist(), "R_GRID": R_GRID.tolist(),
                   "J_grid": J_grid.tolist()}, f, indent=2)
    print("\nExp4 done.")


if __name__ == "__main__":
    main()
