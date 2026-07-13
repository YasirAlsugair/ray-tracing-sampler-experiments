"""Step 4: ray tracing MCMC WITH the Metropolis test on the small MNIST posteriors.

Uses the vendor's `sample_raytrace` functional API (full-batch gradients, exact
accept/reject) on the step-2 derived target:

    ln posterior(theta) = -[ sum-CE over all 60k train images + ||theta||^2 / 2 ]

so SCALE is not a dial here, the temperature is the Bayes posterior by construction.
Tuning (per Josh): step size dt and trajectory length, aiming for ~80% acceptance.
refresh_rate stays 0 inside trajectories so the Metropolis test is exact; momenta
are fully redrawn between trajectories by sample_raytrace itself.

Run modes:
  python exp6_sample_metropolis.py pilot                 # dt sweep, short chains
  python exp6_sample_metropolis.py run mlp DT L N [flat] # production chain
  python exp6_sample_metropolis.py auto                  # tune dt, then run
  python exp6_sample_metropolis.py converge mlp          # extend the saved chain
                                                         # in legs until the drift
                                                         # check passes
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import exp6_simple_mnist_train as base

sys.path.insert(0, str(base.ROOT / "vendor" / "ray-tracing-sampler"))
from raytrace_torch import sample_raytrace

TAB = base.TAB
DEV = base.DEV
CHUNK = 10000  # full batch evaluated in chunks that fit comfortably


def load_full_train():
    tr_loader, _, _ = base.loaders()
    xs, ys = [], []
    for x, y in torch.utils.data.DataLoader(tr_loader.dataset, batch_size=CHUNK):
        xs.append(x.to(DEV))
        ys.append(y.to(DEV))
    return xs, ys


def make_log_prob(model, xs, ys, prior="gaussian"):
    """ln posterior as a function of the flat parameter vector (differentiable).

    prior="gaussian" adds the N(0,1) term -||theta||^2/2; prior="flat" drops it
    (improper target, the ablation)."""
    specs = [(n, p.shape, p.numel()) for n, p in model.named_parameters()]

    def unflatten(theta):
        out, i = {}, 0
        for n, sh, sz in specs:
            out[n] = theta[i : i + sz].view(sh)
            i += sz
        return out

    def log_prob(theta):
        pd = unflatten(theta)
        ce = 0.0
        for xb, yb in zip(xs, ys):
            logits = torch.func.functional_call(model, pd, (xb,))
            ce = ce + F.cross_entropy(logits, yb, reduction="sum")
        if prior == "flat":
            return -ce
        return -(ce + 0.5 * (theta**2).sum())

    return log_prob


def flat_point(name, model):
    model.load_state_dict(torch.load(base.CKPT / f"exp6_{name}.pt"))
    return torch.cat([p.detach().flatten() for p in model.parameters()]).to(DEV)


def acceptance(lnp):
    """Fraction of trajectories whose recorded ln-posterior changed (continuous
    target, so a repeat means a rejection)."""
    return float(np.mean(np.diff(lnp) != 0))


def pilot():
    print(f"device={DEV}")
    for name, cls in [("mlp", base.MLP), ("cnn", base.CNN)]:
        model = cls().to(DEV)
        theta0 = flat_point(name, model)
        xs, ys = load_full_train()
        log_prob = make_log_prob(model, xs, ys)

        t0 = time.time()
        lp = log_prob(theta0.requires_grad_(True))
        g = torch.autograd.grad(lp, theta0)[0]
        theta0 = theta0.detach()
        print(f"[{name}] D={theta0.numel()}  ln_post(point)={lp.item():,.0f}  "
              f"|grad|={g.norm().item():.1f}  grad_eval={time.time() - t0:.2f}s",
              flush=True)

        for L in (10, 30):
            for dt in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
                t0 = time.time()
                samples, lnp = sample_raytrace(
                    theta0, log_prob, n_steps=20, n_leapfrog_steps=L,
                    step_size=dt, refresh_rate=0, device=DEV,
                    samples_device="cpu", scale_likelihood=1.0,
                )
                lnp = lnp.numpy()
                print(f"[{name}] L={L:3d} dt={dt:.0e}  acc={acceptance(lnp):.2f}  "
                      f"ln_post: {lnp[0]:,.0f} -> {lnp[-1]:,.0f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)


def run(name, dt, L, n_steps, prior="gaussian", theta0=None, tag=None):
    cls = {"mlp": base.MLP, "cnn": base.CNN}[name]
    model = cls().to(DEV)
    if theta0 is None:
        theta0 = flat_point(name, model)
    xs, ys = load_full_train()
    log_prob = make_log_prob(model, xs, ys, prior=prior)
    t0 = time.time()
    samples, lnp = sample_raytrace(
        theta0, log_prob, n_steps=n_steps, n_leapfrog_steps=L,
        step_size=dt, refresh_rate=0, device=DEV,
        samples_device="cpu", scale_likelihood=1.0,
    )
    wall = time.time() - t0
    lnp = lnp.numpy()
    suffix = "" if prior == "gaussian" else f"_{prior}"
    if tag:
        suffix += f"_{tag}"
    out = TAB / f"exp6_rt_chain_{name}{suffix}.npz"
    np.savez(
        out,
        samples=samples.numpy().astype(np.float32),
        ln_post=lnp,
        dt=dt, L=L, n_steps=n_steps, wall_s=wall, prior=prior,
    )
    print(f"[{name}] {n_steps} trajectories (L={L}, dt={dt:.0e}, prior={prior}) "
          f"in {wall/60:.1f} min, acc={acceptance(lnp):.2f}, saved {out}", flush=True)


def chain_files(name, prior="gaussian"):
    """The base chain file plus any continuation legs, in chain order."""
    suffix = "" if prior == "gaussian" else f"_{prior}"
    files = [TAB / f"exp6_rt_chain_{name}{suffix}.npz"]
    part = 2
    while (TAB / f"exp6_rt_chain_{name}{suffix}_part{part}.npz").exists():
        files.append(TAB / f"exp6_rt_chain_{name}{suffix}_part{part}.npz")
        part += 1
    return files


def drift_check(lnp):
    """Final-quarter drift vs noise, the notebook's stationarity rule."""
    quarter = lnp[3 * len(lnp) // 4:]
    slope = np.polyfit(np.arange(len(quarter)), quarter, 1)[0]
    drift = slope * len(quarter)
    noise = quarter.std()
    return drift, noise, abs(drift) < 2 * noise


def converge(name, leg=10000, max_legs=6, prior="gaussian"):
    """Extend the saved chain from its last stored state in legs of `leg`
    trajectories, stopping when the combined trace passes the drift check.
    Each leg is saved as exp6_rt_chain_<name>[_prior]_part<k>.npz; momenta are
    redrawn every trajectory, so restarting from the last theta continues the
    same chain."""
    files = chain_files(name, prior)
    base_archive = np.load(files[0])
    dt = float(base_archive["dt"])
    L = int(base_archive["L"])
    traces = [np.asarray(np.load(f)["ln_post"]) for f in files]
    last_state = np.load(files[-1])["samples"][-1]

    cls = {"mlp": base.MLP, "cnn": base.CNN}[name]
    model = cls().to(DEV)
    xs, ys = load_full_train()
    log_prob = make_log_prob(model, xs, ys, prior=prior)
    suffix = "" if prior == "gaussian" else f"_{prior}"

    combined = np.concatenate(traces)
    drift, noise, stationary = drift_check(combined)
    print(f"[{name}] resuming at trajectory {len(combined):,} "
          f"(dt={dt:.1e}, L={L}, prior={prior}); current drift "
          f"{drift:+,.0f} vs noise {noise:,.0f}", flush=True)

    for _ in range(max_legs):
        if stationary:
            break
        theta0 = torch.tensor(last_state, dtype=torch.float32, device=DEV)
        t0 = time.time()
        samples, lnp = sample_raytrace(
            theta0, log_prob, n_steps=leg, n_leapfrog_steps=L,
            step_size=dt, refresh_rate=0, device=DEV,
            samples_device="cpu", scale_likelihood=1.0,
        )
        wall = time.time() - t0
        lnp = lnp.numpy()
        part = len(chain_files(name, prior)) + 1
        out = TAB / f"exp6_rt_chain_{name}{suffix}_part{part}.npz"
        np.savez(out, samples=samples.numpy().astype(np.float32), ln_post=lnp,
                 dt=dt, L=L, n_steps=leg, wall_s=wall, prior=prior)
        last_state = samples[-1].numpy()
        traces.append(lnp)
        combined = np.concatenate(traces)
        drift, noise, stationary = drift_check(combined)
        verdict = "STATIONARY" if stationary else "still in transit"
        print(f"[{name}] leg saved to {out.name}: {leg:,} trajectories in "
              f"{wall / 60:.1f} min, acc={acceptance(lnp):.2f}; combined "
              f"{len(combined):,} trajectories -> {verdict} (drift {drift:+,.0f} "
              f"vs noise {noise:,.0f})", flush=True)

    if stationary:
        print(f"[{name}] chain PASSES the drift check at {len(combined):,} "
              f"trajectories", flush=True)
    else:
        print(f"[{name}] max legs reached, still in transit at "
              f"{len(combined):,} trajectories", flush=True)


def axiom(n_steps=20000):
    """The start-point test: does the Adam start actually matter? Two MLP
    chains at the same target and trajectory length as the reference chain,
    from (a) a fresh random initialization (chance-level fit, tiny norm) and
    (b) a draw from the N(0,1) prior (shell-level norm, garbage fit). Each
    start gets its own short step-size pilot first, because acceptance is
    state dependent and a frozen chain would say nothing."""
    model = base.MLP().to(DEV)
    dimension = sum(p.numel() for p in model.parameters())
    generator = torch.Generator().manual_seed(1)

    torch.manual_seed(1)
    random_model = base.MLP()
    random_start = torch.cat(
        [p.detach().flatten() for p in random_model.parameters()]).to(DEV)
    prior_start = torch.randn(dimension, generator=generator).to(DEV)

    xs, ys = load_full_train()
    log_prob = make_log_prob(model, xs, ys)
    L = 30
    for tag, theta0 in [("randominit", random_start), ("priordraw", prior_start)]:
        with torch.no_grad():
            lnp0 = log_prob(theta0).item()
        norm0 = float((theta0 ** 2).sum())
        print(f"[axiom:{tag}] start: ln_post={lnp0:,.0f}, "
              f"||theta||^2={norm0:,.0f}", flush=True)
        results = {}
        for dt in (1e-3, 3.5e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7):
            _, lnp = sample_raytrace(
                theta0, log_prob, n_steps=20, n_leapfrog_steps=L,
                step_size=dt, refresh_rate=0, device=DEV,
                samples_device="cpu", scale_likelihood=1.0,
            )
            results[dt] = acceptance(lnp.numpy())
            print(f"[axiom:{tag}] pilot dt={dt:.1e} acc={results[dt]:.2f}",
                  flush=True)
            if results[dt] >= 0.95:
                break  # the ladder descends; no need to probe smaller steps
        window = {d: a for d, a in results.items() if 0.6 <= a <= 0.95}
        if window:
            best = min(window, key=lambda d: abs(window[d] - 0.80))
        else:
            safe = {d: a for d, a in results.items() if a >= 0.95}
            moving = [d for d, a in results.items() if a > 0]
            best = max(safe) if safe else (max(moving) if moving else None)
        if best is None:
            print(f"[axiom:{tag}] no step size in the ladder moves this "
                  f"start; skipping the production run", flush=True)
            continue
        print(f"[axiom:{tag}] chosen dt={best:.1e} (acc={results[best]:.2f}), "
              f"launching {n_steps} trajectories at L={L}", flush=True)
        run("mlp", best, L, n_steps, theta0=theta0, tag=tag)


def auto():
    """Mini auto-tuner: sweep dt near the acceptance cliff found by the coarse
    pilot, pick the value closest to the 80% target, launch production."""
    TARGET = 0.80
    plans = {
        # name: (dt candidates, L, tune trajectories, production trajectories)
        "mlp": ((3e-4, 3.5e-4, 4e-4), 30, 40, 2000),
        "cnn": ((1.5e-4, 2e-4, 2.5e-4), 30, 20, 600),
    }
    for name, (dts, L, n_tune, n_prod) in plans.items():
        cls = {"mlp": base.MLP, "cnn": base.CNN}[name]
        model = cls().to(DEV)
        theta0 = flat_point(name, model)
        xs, ys = load_full_train()
        log_prob = make_log_prob(model, xs, ys)
        results = {}
        for dt in dts:
            _, lnp = sample_raytrace(
                theta0, log_prob, n_steps=n_tune, n_leapfrog_steps=L,
                step_size=dt, refresh_rate=0, device=DEV,
                samples_device="cpu", scale_likelihood=1.0,
            )
            results[dt] = acceptance(lnp.numpy())
            print(f"[tune:{name}] dt={dt:.1e} acc={results[dt]:.2f}", flush=True)
        # prefer the 0.6-0.95 window closest to the target; if the cliff is so
        # sharp nothing lands there, fall back to the largest dt with acc >= 0.95
        window = {d: a for d, a in results.items() if 0.6 <= a <= 0.95}
        if window:
            best = min(window, key=lambda d: abs(window[d] - TARGET))
        else:
            safe = {d: a for d, a in results.items() if a >= 0.95}
            best = max(safe) if safe else max((d for d, a in results.items() if a > 0))
        print(f"[tune:{name}] chosen dt={best:.1e} (acc={results[best]:.2f}), "
              f"launching {n_prod} trajectories at L={L}", flush=True)
        run(name, best, L, n_prod)


if __name__ == "__main__":
    if sys.argv[1] == "pilot":
        pilot()
    elif sys.argv[1] == "auto":
        auto()
    elif sys.argv[1] == "axiom":
        axiom(n_steps=int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
    elif sys.argv[1] == "converge":
        converge(sys.argv[2],
                 leg=int(sys.argv[3]) if len(sys.argv) > 3 else 10000,
                 max_legs=int(sys.argv[4]) if len(sys.argv) > 4 else 6,
                 prior=sys.argv[5] if len(sys.argv) > 5 else "gaussian")
    else:
        run(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
            prior=sys.argv[6] if len(sys.argv) > 6 else "gaussian")
