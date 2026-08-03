"""exp7 heteroscedastic run: the network predicts its own scatter.

Target per star i:  var_i = err_i^2 + sigma(x_i)^2, with
sigma(x) = exp(LNS0 + r(x)) and r(x) the network's SECOND output channel.
This is the general version of the intrinsic-scatter model: freeze r(x) to
a constant and you recover ln s = LNS0 + u. The z-binning of the scatter
chain motivates it: z std runs 0.39 to 1.43 across label-error bins and
0.64 to 1.22 across [M/H], so one global s fixes the average but not the
structure.

    nll_i = (y_i - mu(x_i))^2 / (2 var_i) + 0.5 ln var_i + 0.5 ln 2pi
    ln post = -[ sum_i nll_i + 0.5 sum_j (theta_j / sigma_j)^2 ]

The 0.5 ln var_i term is load-bearing here: it is what stops the network
from buying easy likelihood by inflating sigma(x) everywhere.

Model: MLP 110-64-64-2 tanh, D = 11,394. Layout note (the ordering-bug
lesson): this model has ONLY the net submodule and no direct parameters,
so the flat vector is simply the net's parameters in registration order.

Warm start (mode "seed"): built from the scatter chain-of-record endpoint
(exp7s_rt33_dt2e-05.npz, layout [u, net]). The trunk and the mu output row
are copied; the sigma output row starts at zero with bias u_final, so at
step 0 sigma(x) is constant and equal to the scatter run's s = 0.0452.
The run therefore starts exactly at the previous solution and only then
lets sigma(x) develop structure.

Everything else is the exp7 recipe: Glorot prior (bias sigma 0.1),
minibatch Raytracer (batch 1024), Eq. 33 gate every 30 steps.

Prior note: the constant offset of ln sigma now lives in the output bias
and so gets the 0.1 bias prior, where the scatter model's u had prior
sigma 1.0. A 10x tighter prior on the global offset; at u_final = -0.094
the penalty change is under one nat, negligible against the misfit, but
it is a model change, not a pure reparameterization.

Modes:
    python exp7_gaia_hetero.py seed            # build exp7_hetero_seed.npz
    python exp7_gaia_hetero.py smoke           # 600 steps, local sanity
    python exp7_gaia_hetero.py auto            # tune, run 2M, report
    python exp7_gaia_hetero.py report DT
    python exp7_gaia_hetero.py converge DT [MAXLEGS]
"""
import math
import os
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
# Batch size, overridable for the bigger-batch escalation: a larger batch
# shrinks sigma_sto, raises the gate's noise ceiling, and moves the tuning
# knee to a larger dt. Recorded in every saved npz.
BATCH = int(os.environ.get("EXP7H_BATCH", "1024"))
TEST_EVERY = 30
SNAPSHOT_EVERY = 250
SEED = 0
LNS0 = -3.0
LOG2PI = math.log(2.0 * math.pi)


def load_data():
    d = np.load(TAB / "exp7_gaia_pristine.npz")
    to = lambda k: torch.tensor(d[k], device=DEV)
    return (to("train_Xs"), to("train_y"), to("train_yerr"),
            to("test_Xs"), to("test_y"), to("test_yerr"))


class HeteroModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.net = nn.Sequential(nn.Linear(110, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 2))

    def mu_r(self, X):
        out = self.net(X)
        return out[..., 0], out[..., 1]


def make_model():
    return HeteroModel().to(DEV)


def sigmas_for(model):
    """Prior sigma per tensor: Glorot for weights, 0.1 for biases."""
    out = []
    for _, p in model.named_parameters():
        if p.ndim == 2:
            out.append(math.sqrt(2.0 / (p.shape[0] + p.shape[1])))
        else:
            out.append(0.1)
    return out


def load_flat(model, flat):
    flat = torch.tensor(np.asarray(flat), dtype=torch.float32, device=DEV)
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(flat[i:i + p.numel()].view(p.shape))
            i += p.numel()
    assert i == len(flat), (i, len(flat))


def build_seed():
    """Warm start from the scatter chain of record ([u, net] layout)."""
    snap = np.load(TAB / "exp7s_rt33_dt2e-05.npz")["snapshots"][-1]
    u_final = float(snap[0])
    old = snap[1:]                              # W1 b1 W2 b2 W3(1x64) b3(1)
    n_trunk = 110 * 64 + 64 + 64 * 64 + 64      # 11,264
    trunk = old[:n_trunk]
    W3 = old[n_trunk:n_trunk + 64]
    b3 = old[n_trunk + 64:n_trunk + 65]
    W3_new = np.concatenate([W3, np.zeros(64, dtype=old.dtype)])  # (2,64) rows
    b3_new = np.array([b3[0], u_final], dtype=old.dtype)
    state = np.concatenate([trunk, W3_new, b3_new]).astype(np.float32)
    np.savez(TAB / "exp7_hetero_seed.npz", state=state)
    model = make_model()
    load_flat(model, state)
    with torch.no_grad():
        probe = torch.zeros(4, 110, device=DEV)
        _, r = model.mu_r(probe)
    print(f"seed built: D = {len(state):,}, u_final = {u_final:+.4f}, "
          f"sigma(x) at start = {math.exp(LNS0 + float(r[0])):.4f} "
          f"(the scatter endpoint's s; its posterior median was 0.0452)",
          flush=True)


def seed_state():
    return np.load(TAB / "exp7_hetero_seed.npz")["state"]


def batch_nll_mean(model, Xs, y, yerr, batch):
    mu, r = model.mu_r(Xs[batch])
    var = yerr[batch] ** 2 + torch.exp(2.0 * (LNS0 + r))
    return (((y[batch] - mu) ** 2 / (2 * var))
            + 0.5 * torch.log(var) + 0.5 * LOG2PI).mean()


@torch.no_grad()
def exact_decomposition(model, sigmas, Xs, y, yerr):
    misfit = 0.0
    for i in range(0, len(Xs), 20000):
        mu, r = model.mu_r(Xs[i:i + 20000])
        var = yerr[i:i + 20000] ** 2 + torch.exp(2.0 * (LNS0 + r))
        misfit += (((y[i:i + 20000] - mu) ** 2 / (2 * var))
                   + 0.5 * torch.log(var) + 0.5 * LOG2PI).sum().item()
    wnorm = sum(((p / sg) ** 2).sum().item()
                for p, sg in zip(model.parameters(), sigmas))
    return misfit, wnorm


@torch.no_grad()
def sigma_quantiles(model, Xs):
    """Median and 16/84 percentiles of sigma(x) over a fixed probe set."""
    _, r = model.mu_r(Xs[:4096])
    s = torch.exp(LNS0 + r).cpu().numpy()
    return (float(np.median(s)), float(np.percentile(s, 16)),
            float(np.percentile(s, 84)))


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
    wnorm = sum(((p / sg) ** 2).sum().item()
                for p, sg in zip(model.parameters(), sigmas))
    return -(n_train * nll + wnorm / 2)


def run(dt, Xs, y, yerr, n_steps, initial_state=None, sigma_fixed=None,
        out_file=None, seed=SEED, save=True):
    n_train = len(Xs)
    torch.manual_seed(seed)
    batch_stream = torch.Generator().manual_seed(seed + 99)
    accept_stream = np.random.default_rng(seed + 13)

    model = make_model()
    sigmas = sigmas_for(model)
    load_flat(model, initial_state if initial_state is not None else seed_state())

    sampler = Raytracer(model.parameters(), dt=dt,
                        scale_likelihood=float(n_train), refresh_rate=5)
    sigma = (sigma_fixed if sigma_fixed is not None
             else estimate_sigma_sto(model, Xs, y, yerr, n_train))
    softening = 1.0 / np.sqrt(1.0 + sigma ** 2)
    if save:
        print(f"[dt={dt:g}] sigma_sto = {sigma:,.1f} nats -> Eq.33 exponent "
              f"{softening:.2e}", flush=True)

    window_raw, window_accepted = [], []
    window_start_state = [p.detach().clone() for p in model.parameters()]
    window_start_estimate = batch_log_posterior(model, sigmas, Xs, y, yerr,
                                                batch_stream, n_train)
    window_start_ledger = 0.0
    snapshots, misfits, wnorms, steps = [], [], [], []
    sig_med, sig_lo, sig_hi = [], [], []
    t0 = time.time()

    for step in range(n_steps):
        batch = torch.randint(0, n_train, (BATCH,), generator=batch_stream).to(DEV)
        sampler.zero_grad()
        batch_nll_mean(model, Xs, y, yerr, batch).backward()
        with torch.no_grad():
            for p, sg in zip(model.parameters(), sigmas):
                p.grad.add_(p, alpha=1.0 / (sg * sg * n_train))
        sampler.step()

        if (step + 1) % TEST_EVERY == 0:
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
            med, lo, hi = sigma_quantiles(model, Xs)
            steps.append(step + 1)
            misfits.append(misfit)
            wnorms.append(wnorm)
            sig_med.append(med)
            sig_lo.append(lo)
            sig_hi.append(hi)
            snapshots.append(torch.cat([p.detach().flatten()
                                        for p in model.parameters()])
                             .cpu().numpy().astype(np.float32))

    wall = time.time() - t0
    acceptance = float(np.mean(window_accepted)) if window_accepted else None
    if save:
        out = out_file or TAB / f"exp7h_rt33_dt{dt:g}.npz"
        np.savez(out, snapshots=np.stack(snapshots), steps=np.array(steps),
                 misfit=np.array(misfits), wnorm=np.array(wnorms),
                 sig_med=np.array(sig_med), sig_lo=np.array(sig_lo),
                 sig_hi=np.array(sig_hi),
                 window_raw=np.array(window_raw),
                 window_accepted=np.array(window_accepted),
                 sigma_sto=sigma, dt=dt, batch=BATCH,
                 test_every=TEST_EVERY, n_steps=n_steps, wall_s=wall)
        print(f"[dt={dt:g}] {n_steps:,} steps in {wall / 60:.1f} min -> "
              f"misfit {misfits[-1]:,.0f}, wnorm {wnorms[-1]:,.0f} of shell "
              f"D=11,394, sigma(x) median {sig_med[-1]:.4f} "
              f"[{sig_lo[-1]:.4f}, {sig_hi[-1]:.4f}], acceptance "
              f"{'n/a' if acceptance is None else f'{acceptance:.2f}'}, "
              f"saved {Path(out).name}", flush=True)
    return acceptance


def drift_check(series):
    quarter = series[3 * len(series) // 4:]
    slope = np.polyfit(np.arange(len(quarter)), quarter, 1)[0]
    return slope * len(quarter), quarter.std()


def arm_files_h(dt):
    files = [TAB / f"exp7h_rt33_dt{dt:g}.npz"]
    part = 2
    while (TAB / f"exp7h_rt33_dt{dt:g}_part{part}.npz").exists():
        files.append(TAB / f"exp7h_rt33_dt{dt:g}_part{part}.npz")
        part += 1
    return files


def last_state(dt):
    files = arm_files_h(dt)
    if files[0].exists():
        return np.load(files[-1])["snapshots"][-1]
    return seed_state()


def converge(dt, Xs, y, yerr, leg_steps=500_000, max_legs=4):
    """Extend in continuations until misfit, weighted norm, and the median
    sigma(x) all pass the drift rule, read with acceptance."""
    model = make_model()
    load_flat(model, last_state(dt))
    sigma = estimate_sigma_sto(model, Xs, y, yerr, len(Xs))
    print(f"[converge dt={dt:g}] re-measured sigma_sto = {sigma:,.1f}",
          flush=True)

    def status():
        files = arm_files_h(dt)
        misfit = np.concatenate([np.load(f)["misfit"] for f in files])
        wnorm = np.concatenate([np.load(f)["wnorm"] for f in files])
        smed = np.concatenate([np.load(f)["sig_med"] for f in files])
        oks, notes = [], []
        for name, series in (("misfit", misfit), ("wnorm", wnorm),
                             ("sig_med", smed)):
            dr, no = drift_check(series)
            ok = abs(dr) < 2 * no
            oks.append(ok)
            notes.append(f"{name} {dr:+.4g}/{no:.4g} "
                         f"{'level' if ok else 'moving'}")
        return all(oks), "; ".join(notes)

    stationary, line = status()
    print(f"[converge dt={dt:g}] {line}", flush=True)
    for _ in range(max_legs):
        if stationary:
            break
        part = len(arm_files_h(dt)) + 1
        out = TAB / f"exp7h_rt33_dt{dt:g}_part{part}.npz"
        run(dt, Xs, y, yerr, leg_steps, initial_state=last_state(dt),
            sigma_fixed=sigma, out_file=out, seed=SEED + part)
        stationary, line = status()
        print(f"[converge dt={dt:g}] -> "
              f"{'STATIONARY' if stationary else 'in transit'}; {line}",
              flush=True)
    return stationary


def convergepack(dt, Xs, y, yerr, leg_steps=500_000, max_legs=6):
    """Resume convergence from exp7h_pack.npz when the raw chain files are
    gone (pod terminated): the pack's final_state seeds the continuation and
    its concatenated traces are the drift-check history. New legs land in
    _cont{k} files."""
    pack = np.load(TAB / "exp7h_pack.npz")

    def cont_files():
        files, k = [], 1
        while (TAB / f"exp7h_rt33_dt{dt:g}_cont{k}.npz").exists():
            files.append(TAB / f"exp7h_rt33_dt{dt:g}_cont{k}.npz")
            k += 1
        return files

    def cur_state():
        f = cont_files()
        return np.load(f[-1])["snapshots"][-1] if f else pack["final_state"]

    model = make_model()
    load_flat(model, cur_state())
    sigma = estimate_sigma_sto(model, Xs, y, yerr, len(Xs))
    print(f"[convergepack dt={dt:g}] resumed from pack final state, "
          f"sigma_sto = {sigma:,.1f}", flush=True)

    def status():
        f = cont_files()
        oks, notes = [], []
        for name in ("misfit", "wnorm", "sig_med"):
            series = np.concatenate([pack[name]] +
                                    [np.load(x)[name] for x in f])
            dr, no = drift_check(series)
            ok = abs(dr) < 2 * no
            oks.append(ok)
            notes.append(f"{name} {dr:+.4g}/{no:.4g} "
                         f"{'level' if ok else 'moving'}")
        return all(oks), "; ".join(notes)

    stationary, line = status()
    print(f"[convergepack dt={dt:g}] {line}", flush=True)
    for _ in range(max_legs):
        if stationary:
            break
        k = len(cont_files()) + 1
        out = TAB / f"exp7h_rt33_dt{dt:g}_cont{k}.npz"
        run(dt, Xs, y, yerr, leg_steps, initial_state=cur_state(),
            sigma_fixed=sigma, out_file=out, seed=SEED + 10 + k)
        stationary, line = status()
        print(f"[convergepack dt={dt:g}] -> "
              f"{'STATIONARY' if stationary else 'in transit'}; {line}",
              flush=True)
    return stationary


def bigstep(Xs, y, yerr, max_legs=6):
    """The bigger-batch escalation: re-tune the dt ladder from the pack's
    final state at the current BATCH (set EXP7H_BATCH), pick the knee, then
    run convergence legs at that dt via convergepack."""
    state = np.load(TAB / "exp7h_pack.npz")["final_state"]
    prev_acc, prev_dt, chosen = None, None, None
    for dt in (1e-4, 5e-5, 2e-5, 1e-5):
        acc = run(dt, Xs, y, yerr, 3000, initial_state=state, save=False)
        print(f"[bigstep tune] batch={BATCH}  dt={dt:g}  "
              f"Eq.33 acceptance={acc:.2f}", flush=True)
        if prev_acc is not None and acc - prev_acc < 0.02:
            chosen = prev_dt
            break
        prev_acc, prev_dt = acc, dt
    chosen = chosen if chosen is not None else prev_dt
    print(f"[bigstep] batch={BATCH}, chosen dt={chosen:g}", flush=True)
    convergepack(chosen, Xs, y, yerr, max_legs=max_legs)


@torch.no_grad()
def report(dt, Xs, y, yerr, tXs, ty, tyerr, members=50):
    files = arm_files_h(dt)
    snaps = np.concatenate([np.load(f)["snapshots"] for f in files])
    quarter = snaps[3 * len(snaps) // 4:]
    take = np.linspace(0, len(quarter) - 1, min(members, len(quarter))).astype(int)
    model = make_model()
    mus, sigs = [], []
    for state in quarter[take]:
        load_flat(model, state)
        mu, r = model.mu_r(tXs)
        mus.append(mu)
        sigs.append(torch.exp(LNS0 + r))
    mus = torch.stack(mus)
    sigs = torch.stack(sigs)
    mu_hat, tau_hat = mus.mean(dim=0), mus.std(dim=0)
    sig_hat = sigs.median(dim=0).values          # per-star sigma(x)
    var_tot = tau_hat ** 2 + tyerr ** 2 + sig_hat ** 2
    rmse = float(((mu_hat - ty) ** 2).mean().sqrt())
    z = (ty - mu_hat) / var_tot.sqrt()
    misfit = np.concatenate([np.load(f)["misfit"] for f in files])
    wnorm = np.concatenate([np.load(f)["wnorm"] for f in files])
    smed = np.concatenate([np.load(f)["sig_med"] for f in files])
    for name, series in (("misfit", misfit), ("wnorm", wnorm),
                         ("sig_med", smed)):
        dr, no = drift_check(series)
        print(f"drift check {name}: drift {dr:+.4g} vs noise {no:.4g} "
              f"({'level' if abs(dr) < 2 * no else 'moving'})")
    sh = sig_hat.cpu().numpy()
    print(f"sigma(x) over test stars: median {np.median(sh):.4f}, "
          f"16/84 pct [{np.percentile(sh, 16):.4f}, "
          f"{np.percentile(sh, 84):.4f}], "
          f"min/max [{sh.min():.4f}, {sh.max():.4f}]")
    print(f"RMSE chain {rmse:.4f}  (scatter chain was 0.0466)")
    print(f"z std {float(z.std()):.3f} (1.0 = calibrated)  "
          f"median tau {float(tau_hat.median()):.4f}")
    zc = z.cpu().numpy()
    te = tyerr.cpu().numpy()
    print("z std by label error (scatter chain ran 0.39 to 1.43):")
    for lo, hi in ((0.0, 0.004), (0.004, 0.006), (0.006, 0.01),
                   (0.01, 0.02), (0.02, 0.2)):
        m = (te >= lo) & (te < hi)
        if m.sum() < 200:
            continue
        print(f"  [{lo:g}, {hi:g})  n={m.sum():>6}  z std {zc[m].std():.2f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    Xs, y, yerr, tXs, ty, tyerr = load_data()
    model0 = make_model()
    D = sum(p.numel() for p in model0.parameters())
    print(f"device {DEV}, train {len(Xs):,}, test {len(tXs):,}, D = {D:,}",
          flush=True)
    if mode == "seed":
        build_seed()
    elif mode == "smoke":
        run(3.5e-4, Xs, y, yerr, 600, out_file=TAB / "exp7h_smoke.npz")
    elif mode == "auto":
        prev_acc, prev_dt, chosen = None, None, None
        for dt in (1e-4, 5e-5, 2e-5, 1e-5, 5e-6):
            acc = run(dt, Xs, y, yerr, 3000, save=False)
            print(f"[tune] dt={dt:g}  Eq.33 acceptance={acc:.2f}", flush=True)
            if prev_acc is not None and acc - prev_acc < 0.02:
                chosen = prev_dt
                break
            prev_acc, prev_dt = acc, dt
        chosen = chosen if chosen is not None else prev_dt
        print(f"[tune] chosen dt={chosen:g}", flush=True)
        run(chosen, Xs, y, yerr, 2_000_000)
        report(chosen, Xs, y, yerr, tXs, ty, tyerr)
    elif mode == "report":
        report(float(sys.argv[2]), Xs, y, yerr, tXs, ty, tyerr)
    elif mode == "converge":
        dt = float(sys.argv[2])
        legs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        converge(dt, Xs, y, yerr, max_legs=legs)
    elif mode == "convergepack":
        dt = float(sys.argv[2])
        legs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        convergepack(dt, Xs, y, yerr, max_legs=legs)
    elif mode == "bigstep":
        legs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        bigstep(Xs, y, yerr, max_legs=legs)
    print("EXP7H-DONE", flush=True)
