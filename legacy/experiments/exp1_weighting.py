"""Experiment 1 -- Effect of the weighting function W.

Upstream hard-codes W=1. Our fork exposes log_W_fn, bending the ray along
grad(log L - log W). The paper says the stationary distribution stays propto L
for any symmetric W; W only changes the PATH (efficiency / mode mixing).

Two parts:
  (A) HONESTY GATE on a unimodal Gaussian: does the sampled distribution stay the
      same target under every W? We check moments (robust to autocorrelation) and
      a thinned KS statistic vs the analytic marginal.
  (B) WHAT CHANGES on a 2-mode mixture: acceptance, mode switch rate, occupancy
      balance, and ESS per gradient as a function of W.

W family: W = L^c, i.e. log_W = c * log L. The medium then bends along L^(1-c):
  c=0    -> W=1, upstream;                 c=0.5 -> temper tau=2 (bend along L^0.5)
  c=0.75 -> temper tau=4 (bend along L^0.25); c=1 -> W=L = Gibbs (no bending, RW).

Outputs: results/tables/exp1_*.csv, results/figures/exp1_*.png
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rts.config  # noqa
import numpy as np
import jax
from scipy import stats
from rts import targets, chains, metrics
from plots import style
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "..", "results", "figures")
TAB = os.path.join(HERE, "..", "results", "tables")

# W = L^c, bending follows L^(1-c): c<0 sharpens the bend, c>0 weakens it.
# We avoid the exact Gibbs limit c=1 (W=L): there the bending gradient is
# identically zero, so the released kick hits unit_g = 0/0 and acceptance
# collapses -- a numerical singularity of the code, reported separately, not a
# violation of the target invariance.
W_CHOICES = [
    ("sharpen c=-0.5", -0.5),
    ("W=1 (baseline)", 0.0),
    ("temper c=0.5", 0.5),
    ("near-Gibbs c=0.9", 0.9),
]
L_LEAP = 20
SEED = 11


def make_logW(tgt, c):
    if c == 0.0:
        return None
    lp = tgt.log_prob
    return lambda x: c * lp(x)


def run_W(tgt, key, c, step_size, n_steps, M, burn):
    # Controlled comparison: step size is FIXED across W (tuned once on W=1), so
    # the only thing varying is W. Acceptance is then a reported consequence of W,
    # not held constant -- this isolates W's effect on the path from step-size
    # retuning, which otherwise confounds the comparison.
    logW = make_logW(tgt, c)
    k_init, k_run = jax.random.split(key)
    inits = tgt.sample(k_init, M)
    res = chains.run_many("rt", k_run, inits, tgt, step_size=step_size,
                          n_steps=n_steps, n_leapfrog=L_LEAP, log_W_fn=logW)
    X = chains.chain_steps_walkers_dims(res)[burn:]   # (T, M, D)
    return res, X, step_size


def main():
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    key = jax.random.PRNGKey(SEED)

    # ----- Part A: honesty gate on a Gaussian -----
    gate = targets.gaussian(2)
    key, kt = jax.random.split(key)
    gate_step = chains.tune_step_size("rt", kt, gate, n_leapfrog=L_LEAP, target_accept=0.7)["step_size"]
    gate_rows = []
    gate_samples = {}
    for name, c in W_CHOICES:
        key, k = jax.random.split(key)
        res, X, ss = run_W(gate, k, c, gate_step, n_steps=4000, M=64, burn=1000)
        flat = X.reshape(-1, gate.D)
        # thin to ~independent draws for KS
        thinned = X[::25].reshape(-1, gate.D)
        ks0 = stats.kstest(thinned[:, 0], "norm")
        gate_samples[name] = flat[:, 0]
        gate_rows.append({
            "W": name, "c": c, "step_size": ss, "accept": float(res["accept"].mean()),
            "mean0": float(flat[:, 0].mean()), "var0": float(flat[:, 0].var()),
            "mean1": float(flat[:, 1].mean()), "var1": float(flat[:, 1].var()),
            "ks_stat": float(ks0.statistic), "ks_p": float(ks0.pvalue),
        })
        print(f"[gate] {name:16s} acc={res['accept'].mean():.2f} "
              f"mean0={flat[:,0].mean():+.3f} var0={flat[:,0].var():.3f} "
              f"KS={ks0.statistic:.3f} p={ks0.pvalue:.2f}")

    # ----- Part B: what changes on a 2-mode mixture -----
    mix = targets.gaussian_mixture(2, sep=5.0, K=2)
    key, kt = jax.random.split(key)
    mix_step = chains.tune_step_size("rt", kt, mix, n_leapfrog=L_LEAP, target_accept=0.7)["step_size"]
    mix_rows = []
    for name, c in W_CHOICES:
        key, k = jax.random.split(key)
        res, X, ss = run_W(mix, k, c, mix_step, n_steps=5000, M=64, burn=1000)
        occ = metrics.mode_occupancy(X, mix.centers)
        sw, _ = metrics.switch_rate(X, mix.centers)
        summ = metrics.ess_summary(X)
        eff = metrics.efficiency(summ["ess_pca1"], res["n_grad_evals"], res["seconds"])
        mix_rows.append({
            "W": name, "c": c, "step_size": ss, "accept": float(res["accept"].mean()),
            "switch_rate": sw, "occ0": float(occ[0]), "occ1": float(occ[1]),
            "balance": float(1 - abs(occ[0] - occ[1])),
            "ess_pca1": summ["ess_pca1"], "ess_per_grad": eff["ess_per_grad_eval"],
        })
        print(f"[mix ] {name:16s} acc={res['accept'].mean():.2f} switch={sw:.4f} "
              f"occ=({occ[0]:.2f},{occ[1]:.2f}) ess/grad={eff['ess_per_grad_eval']:.2e}")

    # save tables
    def write_csv(path, rows, cols):
        with open(path, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    write_csv(os.path.join(TAB, "exp1_gate.csv"), gate_rows,
              ["W", "c", "step_size", "accept", "mean0", "var0", "mean1", "var1", "ks_stat", "ks_p"])
    write_csv(os.path.join(TAB, "exp1_mixture.csv"), mix_rows,
              ["W", "c", "step_size", "accept", "switch_rate", "occ0", "occ1", "balance", "ess_pca1", "ess_per_grad"])

    # ----- Figure: gate (left) + what-changes (right) -----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    xs = np.linspace(-4, 4, 200)
    axA.plot(xs, stats.norm.pdf(xs), color=style.TRUTH, lw=2.5, ls="--", label="truth N(0,1)")
    cmap = [style.RT, style.BLUE, "#7BAE7F", style.GREY]
    for (name, _), col in zip(W_CHOICES, cmap):
        axA.hist(gate_samples[name], bins=60, range=(-4, 4), density=True,
                 histtype="step", lw=1.6, color=col, label=name)
    axA.set_title("Honesty gate: same target under every W\n(Gaussian, coordinate 0)")
    axA.set_xlabel("x_0"); axA.set_ylabel("density"); axA.legend(fontsize=8)

    names = [r["W"] for r in mix_rows]
    sw = [r["switch_rate"] for r in mix_rows]
    bal = [r["balance"] for r in mix_rows]
    x = np.arange(len(names))
    axB.bar(x - 0.2, sw, 0.4, color=style.RT, label="mode switch rate")
    axB.bar(x + 0.2, bal, 0.4, color=style.BLUE, label="occupancy balance (1=perfect)")
    axB.set_xticks(x); axB.set_xticklabels(names, rotation=20, ha="right")
    axB.set_title("What W changes: mixing on a 2-mode target")
    axB.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp1_weighting.png"))
    plt.close(fig)

    with open(os.path.join(TAB, "exp1_summary.json"), "w") as f:
        json.dump({"gate": gate_rows, "mixture": mix_rows}, f, indent=2)
    print("\nExp1 done.")


if __name__ == "__main__":
    main()
