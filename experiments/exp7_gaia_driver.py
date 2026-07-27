"""exp7: minibatch ray tracing WITH the Eq. 33 gate on the Gaia [alpha/M]
regression posterior (reference run for the exp7 workbook).

Target (T = 1, per star i in the pristine train split):
    nll_i = (y_i - mu_theta(x_i))^2 / (2 err_i^2) + ln err_i + 0.5 ln 2pi
    ln post = -[ sum_i nll_i + 0.5 sum_j (theta_j / sigma_j)^2 ]
Prior: Glorot per layer (weight sigma_l = sqrt(2/(fan_in+fan_out)), bias 0.1),
so the prior shell sits where sensible nets live (workbook section 3).

Sampler: the vendor's minibatch Raytracer (batch 256) with the paper's
Eq. 33 noise-softened acceptance every 30 steps, exactly the exp6 recipe
(experiments/exp6_minibatch.py); only the likelihood line and the prior
gradient differ. sigma_sto is measured at the chain start and kept fixed for
the tune and base arm; converge() re-measures it once at its (well-fit)
resume state, then holds it across legs so every leg runs one gate.

Modes:
    python exp7_gaia_driver.py smoke                # 600 steps, local sanity
    python exp7_gaia_driver.py tune                 # dt ladder, 3k-step probes
    python exp7_gaia_driver.py run DT [STEPS]       # base arm (default 200k)
    python exp7_gaia_driver.py converge DT [LEG] [MAXLEGS]
    python exp7_gaia_driver.py report DT            # kNN baseline + predictive
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"
sys.path.insert(0, str(ROOT / "vendor" / "ray-tracing-sampler"))
from raytrace_torch import Raytracer

DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH = 1024
TEST_EVERY = 30
SNAPSHOT_EVERY = 250
SEED = 0
HIDDEN = 64
LOG2PI = math.log(2.0 * math.pi)


def load_data():
    d = np.load(TAB / "exp7_gaia_pristine.npz")
    to = lambda k: torch.tensor(d[k], device=DEV)
    return (to("train_Xs"), to("train_y"), to("train_yerr"),
            to("test_Xs"), to("test_y"), to("test_yerr"))


def make_model():
    torch.manual_seed(SEED)
    return nn.Sequential(nn.Linear(110, HIDDEN), nn.Tanh(),
                         nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                         nn.Linear(HIDDEN, 1)).to(DEV)


def glorot_sigmas(model, bias_sigma=0.1):
    """One prior sigma per parameter tensor: Glorot for weights, fixed for
    biases. Returned in model.parameters() order."""
    sigmas = []
    for p in model.parameters():
        if p.ndim == 2:
            fan_out, fan_in = p.shape
            sigmas.append(math.sqrt(2.0 / (fan_in + fan_out)))
        else:
            sigmas.append(bias_sigma)
    return sigmas


def prior_draw(model, sigmas, seed=SEED):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p, s in zip(model.parameters(), sigmas):
            p.normal_(0.0, s)


def batch_nll_mean(model, Xs, y, yerr, batch):
    mu = model(Xs[batch]).squeeze(-1)
    return (((y[batch] - mu) ** 2 / (2 * yerr[batch] ** 2))
            + torch.log(yerr[batch]) + 0.5 * LOG2PI).mean()


@torch.no_grad()
def exact_decomposition(model, sigmas, Xs, y, yerr):
    """Full-train summed NLL (nats) and the prior-weighted squared norm."""
    misfit = 0.0
    for i in range(0, len(Xs), 20000):
        mu = model(Xs[i:i + 20000]).squeeze(-1)
        misfit += (((y[i:i + 20000] - mu) ** 2 / (2 * yerr[i:i + 20000] ** 2))
                   + torch.log(yerr[i:i + 20000]) + 0.5 * LOG2PI).sum().item()
    wnorm = sum(((p / s) ** 2).sum().item()
                for p, s in zip(model.parameters(), sigmas))
    return misfit, wnorm


@torch.no_grad()
def estimate_sigma_sto(model, Xs, y, yerr, n_train, draws=50):
    stream = torch.Generator().manual_seed(SEED + 7)
    vals = []
    for _ in range(draws):
        batch = torch.randint(0, n_train, (BATCH,), generator=stream).to(DEV)
        vals.append(-n_train * batch_nll_mean(model, Xs, y, yerr, batch).item())
    return float(np.std(vals))


@torch.no_grad()
def batch_log_posterior(model, sigmas, Xs, y, yerr, batch_stream, n_train):
    batch = torch.randint(0, n_train, (BATCH,), generator=batch_stream).to(DEV)
    nll = batch_nll_mean(model, Xs, y, yerr, batch).item()
    wnorm = sum(((p / s) ** 2).sum().item()
                for p, s in zip(model.parameters(), sigmas))
    return -(n_train * nll + wnorm / 2)


def run_arm(dt, Xs, y, yerr, n_steps, refresh_rate=5, initial_state=None,
            sigma_fixed=None, out_file=None, seed=SEED, save=True,
            gated=True, prior="glorot"):
    """One arm, the exp6 recipe verbatim with the exp7 target. gated=False
    drops the Eq. 33 test entirely (the exp6 'unadjusted' control arm).
    prior="normal" swaps the Glorot sigmas for N(0,1), the exp6 prior."""
    n_train = len(Xs)
    torch.manual_seed(seed)
    batch_stream = torch.Generator().manual_seed(seed + 99)
    accept_stream = np.random.default_rng(seed + 13)

    model = make_model()
    sigmas = (glorot_sigmas(model) if prior == "glorot"
              else [1.0] * len(list(model.parameters())))
    if initial_state is not None:
        flat = torch.tensor(initial_state, dtype=torch.float32, device=DEV)
        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(flat[offset:offset + p.numel()].view(p.shape))
                offset += p.numel()
    else:
        prior_draw(model, sigmas, seed=seed)

    sampler = Raytracer(model.parameters(), dt=dt,
                        scale_likelihood=float(n_train),
                        refresh_rate=refresh_rate)
    sigma = 0.0
    softening = 1.0
    if gated:
        sigma = (sigma_fixed if sigma_fixed is not None
                 else estimate_sigma_sto(model, Xs, y, yerr, n_train))
        softening = 1.0 / np.sqrt(1.0 + sigma ** 2)
        if save:
            print(f"[dt={dt:g}] sigma_sto = {sigma:,.1f} nats -> Eq.33 "
                  f"exponent {softening:.2e}", flush=True)

    window_raw, window_accepted = [], []
    window_start_state = [p.detach().clone() for p in model.parameters()]
    window_start_estimate = batch_log_posterior(model, sigmas, Xs, y, yerr,
                                                batch_stream, n_train)
    window_start_ledger = 0.0
    snapshots, misfits, wnorms, steps = [], [], [], []
    t0 = time.time()

    for step in range(n_steps):
        batch = torch.randint(0, n_train, (BATCH,), generator=batch_stream).to(DEV)
        sampler.zero_grad()
        batch_nll_mean(model, Xs, y, yerr, batch).backward()
        with torch.no_grad():
            # per-layer Gaussian prior as an explicit gradient term, the
            # exp6 theta/N pattern generalized to theta/(sigma_l^2 N)
            for p, s in zip(model.parameters(), sigmas):
                p.grad.add_(p, alpha=1.0 / (s * s * n_train))
        sampler.step()

        if gated and (step + 1) % TEST_EVERY == 0:
            end_estimate = batch_log_posterior(model, sigmas, Xs, y, yerr,
                                               batch_stream, n_train)
            ledger = float(sampler.param_groups[0]["ln_luminosity"])
            raw = (end_estimate - window_start_estimate) \
                - (ledger - window_start_ledger)
            accept_probability = min(1.0, np.exp(min(0.0, raw * softening)))
            accepted = bool(accept_stream.random() < accept_probability)
            window_raw.append(raw)
            window_accepted.append(accepted)
            with torch.no_grad():
                if accepted:
                    for p, kept in zip(model.parameters(), window_start_state):
                        kept.copy_(p)
                    window_start_estimate = end_estimate
                else:
                    for p, kept in zip(model.parameters(), window_start_state):
                        p.copy_(kept)
                    for p in model.parameters():
                        state = sampler.state[p]
                        if "momenta" in state:
                            state["momenta"].normal_(0.0, 1.0)
            window_start_ledger = ledger

        if save and (step + 1) % SNAPSHOT_EVERY == 0:
            misfit, wnorm = exact_decomposition(model, sigmas, Xs, y, yerr)
            steps.append(step + 1)
            misfits.append(misfit)
            wnorms.append(wnorm)
            snapshots.append(torch.cat([p.detach().flatten()
                                        for p in model.parameters()])
                             .cpu().numpy().astype(np.float32))

    wall = time.time() - t0
    acceptance = float(np.mean(window_accepted)) if window_accepted else None
    if save:
        out = out_file or TAB / f"exp7_rt33_dt{dt:g}.npz"
        np.savez(out, snapshots=np.stack(snapshots), steps=np.array(steps),
                 misfit=np.array(misfits), wnorm=np.array(wnorms),
                 window_raw=np.array(window_raw),
                 window_accepted=np.array(window_accepted),
                 sigma_sto=sigma, dt=dt, refresh_rate=refresh_rate,
                 batch=BATCH, test_every=TEST_EVERY, n_steps=n_steps,
                 wall_s=wall, hidden=HIDDEN)
        D = sum(p.numel() for p in make_model().parameters())
        print(f"[dt={dt:g}] {n_steps:,} steps in {wall / 60:.1f} min -> "
              f"misfit {misfits[-1]:,.0f}, weighted norm {wnorms[-1]:,.0f} "
              f"of shell D={D:,}, acceptance "
              f"{'n/a' if acceptance is None else f'{acceptance:.2f}'}, "
              f"saved {Path(out).name}", flush=True)
    return acceptance


def arm_files(dt):
    files = [TAB / f"exp7_rt33_dt{dt:g}.npz"]
    part = 2
    while (TAB / f"exp7_rt33_dt{dt:g}_part{part}.npz").exists():
        files.append(TAB / f"exp7_rt33_dt{dt:g}_part{part}.npz")
        part += 1
    return files


def drift_check(series):
    quarter = series[3 * len(series) // 4:]
    slope = np.polyfit(np.arange(len(quarter)), quarter, 1)[0]
    drift = slope * len(quarter)
    noise = quarter.std()
    return drift, noise, abs(drift) < 2 * noise


def converge(dt, Xs, y, yerr, leg_steps=200_000, max_legs=8):
    """Extend the saved arm in legs until BOTH the exact misfit and the
    weighted norm pass the drift rule. sigma_sto is re-measured once at the
    (well-fit) resume state, then held fixed across legs."""
    files = arm_files(dt)
    misfits = [np.asarray(np.load(f)["misfit"]) for f in files]
    wnorms = [np.asarray(np.load(f)["wnorm"]) for f in files]
    last_state = np.load(files[-1])["snapshots"][-1]
    total = sum(int(np.load(f)["n_steps"]) for f in files)

    model = make_model()
    sigmas = glorot_sigmas(model)
    flat = torch.tensor(last_state, dtype=torch.float32, device=DEV)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(flat[offset:offset + p.numel()].view(p.shape))
            offset += p.numel()
    sigma = estimate_sigma_sto(model, Xs, y, yerr, len(Xs))
    print(f"[converge dt={dt:g}] resume at step {total:,}, re-measured "
          f"sigma_sto = {sigma:,.1f} nats", flush=True)

    def status():
        m = drift_check(np.concatenate(misfits))
        w = drift_check(np.concatenate(wnorms))
        return (m[2] and w[2],
                f"misfit {np.concatenate(misfits)[-1]:,.0f} "
                f"(drift {m[0]:+,.0f} vs noise {m[1]:,.0f}, "
                f"{'level' if m[2] else 'moving'}), wnorm "
                f"{np.concatenate(wnorms)[-1]:,.0f} "
                f"(drift {w[0]:+,.0f} vs noise {w[1]:,.0f}, "
                f"{'level' if w[2] else 'moving'})")

    stationary, line = status()
    print(f"[converge dt={dt:g}] {line}", flush=True)
    for _ in range(max_legs):
        if stationary:
            break
        part = len(arm_files(dt)) + 1
        out = TAB / f"exp7_rt33_dt{dt:g}_part{part}.npz"
        run_arm(dt, Xs, y, yerr, leg_steps, initial_state=last_state,
                sigma_fixed=sigma, out_file=out, seed=SEED + part)
        archive = np.load(out)
        misfits.append(np.asarray(archive["misfit"]))
        wnorms.append(np.asarray(archive["wnorm"]))
        last_state = archive["snapshots"][-1]
        total += leg_steps
        stationary, line = status()
        print(f"[converge dt={dt:g}] step {total:,} -> "
              f"{'STATIONARY' if stationary else 'in transit'}; {line}",
              flush=True)
    return stationary


@torch.no_grad()
def report(dt, Xs, y, yerr, tXs, ty, tyerr, members=50, files=None):
    """kNN baseline + chain predictive on the held-out split."""
    # kNN k=10 baseline and the global-mean floor
    k = 10
    preds = torch.empty(len(tXs), device=DEV)
    for i in range(0, len(tXs), 1024):
        d = torch.cdist(tXs[i:i + 1024], Xs)
        _, nn_idx = torch.topk(d, k, largest=False)
        preds[i:i + 1024] = y[nn_idx].mean(dim=1)
    rmse_knn = float(((preds - ty) ** 2).mean().sqrt())
    rmse_mean = float(((y.mean() - ty) ** 2).mean().sqrt())

    # chain members: evenly thinned over the final quarter of all snapshots
    files = files or arm_files(dt)
    snaps = np.concatenate([np.load(f)["snapshots"] for f in files])
    quarter = snaps[3 * len(snaps) // 4:]
    take = np.linspace(0, len(quarter) - 1, min(members, len(quarter))).astype(int)
    model = make_model()
    mus = []
    for state in quarter[take]:
        flat = torch.tensor(state, dtype=torch.float32, device=DEV)
        offset = 0
        for p in model.parameters():
            p.copy_(flat[offset:offset + p.numel()].view(p.shape))
            offset += p.numel()
        mus.append(model(tXs).squeeze(-1))
    mus = torch.stack(mus)                      # (M, n_test)
    mu_hat, tau_hat = mus.mean(dim=0), mus.std(dim=0)
    var_tot = tau_hat ** 2 + tyerr ** 2
    rmse_chain = float(((mu_hat - ty) ** 2).mean().sqrt())
    nll_chain = float((0.5 * (ty - mu_hat) ** 2 / var_tot
                       + 0.5 * torch.log(2 * math.pi * var_tot)).mean())
    z = (ty - mu_hat) / var_tot.sqrt()
    print(f"members {len(take)}  (from {len(snaps)} snapshots)")
    print(f"RMSE  global-mean {rmse_mean:.4f}  kNN(k=10) {rmse_knn:.4f}  "
          f"chain {rmse_chain:.4f}")
    print(f"chain test NLL/star {nll_chain:.3f}  z std {float(z.std()):.3f} "
          f"(1.0 = calibrated)  median tau {float(tau_hat.median()):.4f} "
          f"vs median label err {float(tyerr.median()):.4f}")


def tune_ladder(Xs, y, yerr):
    """The paper's recipe: lower dt until Eq. 33 acceptance stops improving."""
    prev_acc, prev_dt, chosen = None, None, None
    for dt in (1e-3, 3.5e-4, 2e-4, 1e-4, 5e-5, 3e-5):
        acc = run_arm(dt, Xs, y, yerr, 3000, save=False)
        print(f"[tune] dt={dt:g}  Eq.33 acceptance={acc:.2f}", flush=True)
        if prev_acc is not None and acc - prev_acc < 0.02:
            chosen = prev_dt
            break
        prev_acc, prev_dt = acc, dt
    chosen = chosen if chosen is not None else prev_dt
    print(f"[tune] chosen dt={chosen:g}", flush=True)
    return chosen


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tune"
    Xs, y, yerr, tXs, ty, tyerr = load_data()
    print(f"device {DEV}, train {len(Xs):,}, test {len(tXs):,}, "
          f"D = {sum(p.numel() for p in make_model().parameters()):,}",
          flush=True)
    if mode == "smoke":
        run_arm(3.5e-4, Xs, y, yerr, 600,
                out_file=TAB / "exp7_rt33_smoke.npz")
    elif mode == "tune":
        tune_ladder(Xs, y, yerr)
    elif mode == "auto":
        chosen = tune_ladder(Xs, y, yerr)
        run_arm(chosen, Xs, y, yerr, 200_000)
        converge(chosen, Xs, y, yerr)
        report(chosen, Xs, y, yerr, tXs, ty, tyerr)
    elif mode == "run":
        dt = float(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
        run_arm(dt, Xs, y, yerr, n)
    elif mode == "converge":
        dt = float(sys.argv[2])
        leg = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
        legs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
        converge(dt, Xs, y, yerr, leg_steps=leg, max_legs=legs)
    elif mode == "report":
        report(float(sys.argv[2]), Xs, y, yerr, tXs, ty, tyerr)
    elif mode == "runnormal":
        # the exp6 prior on the exp7 target: same sampler, same budget,
        # N(0,1) on every weight, started from its own prior draw
        dt = float(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 500_000
        run_arm(dt, Xs, y, yerr, n, prior="normal",
                out_file=TAB / f"exp7n_rt33_dt{dt:g}.npz")
    elif mode == "reportnormal":
        dt = float(sys.argv[2])
        report(dt, Xs, y, yerr, tXs, ty, tyerr,
               files=[TAB / f"exp7n_rt33_dt{dt:g}.npz"])
    elif mode == "tunefit":
        # the ladder done right: probe acceptance FROM THE FITTED STATE,
        # where sigma_sto is small enough for the gate to see anything
        src = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
        state = np.load(arm_files(src)[-1])["snapshots"][-1]
        for dt in (2e-5, 1e-5, 5e-6, 2e-6, 1e-6):
            acc = run_arm(dt, Xs, y, yerr, 3000, initial_state=state,
                          save=False)
            print(f"[tunefit] dt={dt:g}  Eq.33 acceptance={acc:.2f}",
                  flush=True)
    elif mode == "runfrom":
        # gated production arm started from another arm's endpoint
        dt = float(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000
        src = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
        state = np.load(arm_files(src)[-1])["snapshots"][-1]
        run_arm(dt, Xs, y, yerr, n, initial_state=state)
    elif mode == "ungated":
        dt = float(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 500_000
        start = np.load(arm_files(dt)[-1])["snapshots"][-1]
        run_arm(dt, Xs, y, yerr, n, initial_state=start, gated=False,
                out_file=TAB / f"exp7_rtub_dt{dt:g}.npz")
    elif mode == "reportub":
        dt = float(sys.argv[2])
        report(dt, Xs, y, yerr, tXs, ty, tyerr,
               files=[TAB / f"exp7_rtub_dt{dt:g}.npz"])
    print("EXP7-DRIVER-DONE", flush=True)
