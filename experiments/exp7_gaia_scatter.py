"""exp7 intrinsic-scatter run: the corrected noise model, sampled.

Target per star i:  var_i = err_i^2 + s^2,  with ln s = -3 + u and u the
11,330th sampled parameter (prior u ~ N(0,1), so s has a lognormal prior
centered on 0.0498, roughly the measured residual RMSE).

    nll_i = (y_i - mu(x_i))^2 / (2 var_i) + 0.5 ln var_i + 0.5 ln 2pi
    ln post = -[ sum_i nll_i + 0.5 sum_j (theta_j / sigma_j)^2 ]

Everything else is the exp7 recipe: Glorot prior on the network, minibatch
Raytracer (batch 1024), Eq. 33 gate every 30 steps, started from the
gated chain-of-record endpoint (exp7_scatter_seed.npz) with u = 0.

Modes:
    python exp7_gaia_scatter.py smoke          # 600 steps, local sanity
    python exp7_gaia_scatter.py auto           # tune from seed state, run 2M, report
    python exp7_gaia_scatter.py report DT
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
from raytrace_torch import Raytracer, sample_raytrace

DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH = 1024
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


class ScatterModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.net = nn.Sequential(nn.Linear(110, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 1))
        self.u = nn.Parameter(torch.zeros(1))   # ln s = LNS0 + u

    def s(self):
        return torch.exp(LNS0 + self.u)


def make_model():
    return ScatterModel().to(DEV)


def sigmas_for(model):
    """Prior sigma per tensor: Glorot for weights, 0.1 biases, 1.0 for u."""
    out = []
    for name, p in model.named_parameters():
        if name == "u":
            out.append(1.0)
        elif p.ndim == 2:
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


def batch_nll_mean(model, Xs, y, yerr, batch):
    mu = model.net(Xs[batch]).squeeze(-1)
    var = yerr[batch] ** 2 + model.s() ** 2
    return (((y[batch] - mu) ** 2 / (2 * var))
            + 0.5 * torch.log(var) + 0.5 * LOG2PI).mean()


@torch.no_grad()
def exact_decomposition(model, sigmas, Xs, y, yerr):
    misfit = 0.0
    s2 = float(model.s() ** 2)
    for i in range(0, len(Xs), 20000):
        mu = model.net(Xs[i:i + 20000]).squeeze(-1)
        var = yerr[i:i + 20000] ** 2 + s2
        misfit += (((y[i:i + 20000] - mu) ** 2 / (2 * var))
                   + 0.5 * torch.log(var) + 0.5 * LOG2PI).sum().item()
    wnorm = sum(((p / sg) ** 2).sum().item()
                for p, sg in zip(model.parameters(), sigmas))
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
    wnorm = sum(((p / sg) ** 2).sum().item()
                for p, sg in zip(model.parameters(), sigmas))
    return -(n_train * nll + wnorm / 2)


def seed_state():
    return np.load(TAB / "exp7_scatter_seed.npz")["state"]


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
    snapshots, misfits, wnorms, s_trace, steps = [], [], [], [], []
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
            steps.append(step + 1)
            misfits.append(misfit)
            wnorms.append(wnorm)
            s_trace.append(float(model.s()))
            snapshots.append(torch.cat([p.detach().flatten()
                                        for p in model.parameters()])
                             .cpu().numpy().astype(np.float32))

    wall = time.time() - t0
    acceptance = float(np.mean(window_accepted)) if window_accepted else None
    if save:
        out = out_file or TAB / f"exp7s_rt33_dt{dt:g}.npz"
        np.savez(out, snapshots=np.stack(snapshots), steps=np.array(steps),
                 misfit=np.array(misfits), wnorm=np.array(wnorms),
                 s_trace=np.array(s_trace),
                 window_raw=np.array(window_raw),
                 window_accepted=np.array(window_accepted),
                 sigma_sto=sigma, dt=dt, batch=BATCH,
                 test_every=TEST_EVERY, n_steps=n_steps, wall_s=wall)
        print(f"[dt={dt:g}] {n_steps:,} steps in {wall / 60:.1f} min -> "
              f"misfit {misfits[-1]:,.0f}, wnorm {wnorms[-1]:,.0f} of shell "
              f"D=11,330, s = {s_trace[-1]:.4f}, acceptance "
              f"{'n/a' if acceptance is None else f'{acceptance:.2f}'}, "
              f"saved {Path(out).name}", flush=True)
    return acceptance


def drift_check(series):
    quarter = series[3 * len(series) // 4:]
    slope = np.polyfit(np.arange(len(quarter)), quarter, 1)[0]
    return slope * len(quarter), quarter.std()


def arm_files_s(dt):
    files = [TAB / f"exp7s_rt33_dt{dt:g}.npz"]
    part = 2
    while (TAB / f"exp7s_rt33_dt{dt:g}_part{part}.npz").exists():
        files.append(TAB / f"exp7s_rt33_dt{dt:g}_part{part}.npz")
        part += 1
    return files


def last_state(dt):
    files = arm_files_s(dt)
    if files[0].exists():
        return np.load(files[-1])["snapshots"][-1]
    return seed_state()


def converge(dt, Xs, y, yerr, leg_steps=500_000, max_legs=4):
    """Extend the run in continuations until misfit, weighted norm, and s
    all pass the drift rule. sigma_sto re-measured once at the resume
    state, then held fixed."""
    model = make_model()
    load_flat(model, last_state(dt))
    sigma = estimate_sigma_sto(model, Xs, y, yerr, len(Xs))
    print(f"[converge dt={dt:g}] re-measured sigma_sto = {sigma:,.1f}",
          flush=True)

    def status():
        files = arm_files_s(dt)
        misfit = np.concatenate([np.load(f)["misfit"] for f in files])
        wnorm = np.concatenate([np.load(f)["wnorm"] for f in files])
        s_tr = np.concatenate([np.load(f)["s_trace"] for f in files])
        oks, notes = [], []
        for name, series in (("misfit", misfit), ("wnorm", wnorm), ("s", s_tr)):
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
        part = len(arm_files_s(dt)) + 1
        out = TAB / f"exp7s_rt33_dt{dt:g}_part{part}.npz"
        run(dt, Xs, y, yerr, leg_steps, initial_state=last_state(dt),
            sigma_fixed=sigma, out_file=out, seed=SEED + part)
        stationary, line = status()
        print(f"[converge dt={dt:g}] -> "
              f"{'STATIONARY' if stationary else 'in transit'}; {line}",
              flush=True)
    return stationary


def make_log_prob_full(Xs, y, yerr):
    """Exact full-batch differentiable ln posterior of the flat vector."""
    from torch.func import functional_call
    model = make_model()
    specs = [(n, p.shape, p.numel()) for n, p in model.net.named_parameters()]
    sig_flat = torch.cat([torch.full((n,), s, device=DEV) for s, n in
                          zip(sigmas_for(model),
                              [p.numel() for p in model.parameters()])])

    def log_prob(theta):
        out, i = {}, 0
        for n, sh, sz in specs:
            out[n] = theta[i:i + sz].view(sh)
            i += sz
        u = theta[-1]
        mu = functional_call(model.net, out, (Xs,)).squeeze(-1)
        var = yerr ** 2 + torch.exp(2 * (LNS0 + u))
        nll = (((y - mu) ** 2 / (2 * var))
               + 0.5 * torch.log(var) + 0.5 * LOG2PI).sum()
        prior = 0.5 * ((theta / sig_flat) ** 2).sum()
        return -(nll + prior)

    return log_prob


def exact(dt_chain, Xs, y, yerr, n_traj=5000, L=30):
    """The exact full-batch finisher: noiseless Metropolis test, started
    from the minibatch run's endpoint. Pilots the step size first."""
    log_prob = make_log_prob_full(Xs, y, yerr)
    theta0 = torch.tensor(last_state(dt_chain), dtype=torch.float32,
                          device=DEV)

    def acceptance(lnp):
        return float(np.mean(np.diff(lnp) != 0))

    chosen, best_gap = None, 1e9
    for dt in (2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6):
        samples, lnp = sample_raytrace(
            theta0, log_prob, n_steps=20, n_leapfrog_steps=L,
            step_size=dt, refresh_rate=0, device=DEV,
            samples_device="cpu", scale_likelihood=1.0)
        acc = acceptance(lnp.numpy())
        print(f"[exact pilot] dt={dt:g}  acc={acc:.2f}", flush=True)
        if acc >= 0.65:
            chosen = dt          # largest step clearing the bar
            break
        if abs(acc - 0.8) < best_gap:
            best_gap, chosen = abs(acc - 0.8), dt
    print(f"[exact] chosen dt={chosen:g}, running {n_traj:,} trajectories "
          f"(L={L})", flush=True)
    t0 = time.time()
    samples, lnp = sample_raytrace(
        theta0, log_prob, n_steps=n_traj, n_leapfrog_steps=L,
        step_size=chosen, refresh_rate=0, device=DEV,
        samples_device="cpu", scale_likelihood=1.0)
    lnp = lnp.numpy()
    out = TAB / f"exp7s_exact_dt{chosen:g}.npz"
    np.savez(out, samples=samples.numpy().astype(np.float32), ln_post=lnp,
             dt=chosen, L=L, n_traj=n_traj, wall_s=time.time() - t0)
    dr, no = drift_check(lnp)
    print(f"[exact] {n_traj:,} trajectories in {(time.time()-t0)/60:.1f} min, "
          f"acceptance {acceptance(lnp):.2f}, ln_post {lnp[0]:,.0f} -> "
          f"{lnp[-1]:,.0f}, drift {dr:+,.0f} vs noise {no:,.0f} "
          f"({'level' if abs(dr) < 2*no else 'moving'}), saved {out.name}",
          flush=True)


@torch.no_grad()
def report_exact(dt_exact, tXs, ty, tyerr, members=50):
    d_ex = np.load(TAB / f"exp7s_exact_dt{dt_exact:g}.npz")
    snaps = d_ex["samples"]
    quarter = snaps[3 * len(snaps) // 4:]
    take = np.linspace(0, len(quarter) - 1, members).astype(int)
    model = make_model()
    mus, ss = [], []
    for st in quarter[take]:
        load_flat(model, st)
        mus.append(model.net(tXs).squeeze(-1))
        ss.append(float(model.s()))
    mus = torch.stack(mus)
    ss = np.array(ss)
    mu_hat, tau_hat = mus.mean(dim=0), mus.std(dim=0)
    s_med = float(np.median(ss))
    var_tot = tau_hat ** 2 + tyerr ** 2 + s_med ** 2
    z = (ty - mu_hat) / var_tot.sqrt()
    print(f"[exact report] s median {s_med:.4f} spread {ss.std():.4f}  "
          f"RMSE {float(((mu_hat - ty) ** 2).mean().sqrt()):.4f}  "
          f"z std {float(z.std()):.3f}  median tau "
          f"{float(tau_hat.median()):.4f}", flush=True)


@torch.no_grad()
def report(dt, Xs, y, yerr, tXs, ty, tyerr, members=50):
    archive = np.load(TAB / f"exp7s_rt33_dt{dt:g}.npz")
    snaps = archive["snapshots"]
    quarter = snaps[3 * len(snaps) // 4:]
    take = np.linspace(0, len(quarter) - 1, min(members, len(quarter))).astype(int)
    model = make_model()
    mus, ss = [], []
    for state in quarter[take]:
        load_flat(model, state)
        mus.append(model.net(tXs).squeeze(-1))
        ss.append(float(model.s()))
    mus = torch.stack(mus)
    ss = np.array(ss)
    mu_hat, tau_hat = mus.mean(dim=0), mus.std(dim=0)
    s_med = float(np.median(ss))
    var_tot = tau_hat ** 2 + tyerr ** 2 + s_med ** 2
    rmse = float(((mu_hat - ty) ** 2).mean().sqrt())
    nll = float((0.5 * (ty - mu_hat) ** 2 / var_tot
                 + 0.5 * torch.log(2 * math.pi * var_tot)).mean())
    z = (ty - mu_hat) / var_tot.sqrt()
    for name, series in (("misfit", archive["misfit"]),
                         ("wnorm", archive["wnorm"]),
                         ("s", archive["s_trace"])):
        dr, no = drift_check(series)
        print(f"drift check {name}: drift {dr:+.4g} vs noise {no:.4g} "
              f"({'level' if abs(dr) < 2 * no else 'moving'})")
    print(f"s: median {s_med:.4f}, member spread {ss.std():.4f}, "
          f"trace last {archive['s_trace'][-1]:.4f}")
    print(f"RMSE chain {rmse:.4f}  (k-NN reference 0.0742)")
    print(f"test NLL/star {nll:.3f}  z std {float(z.std()):.3f} "
          f"(1.0 = calibrated)  median tau {float(tau_hat.median()):.4f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    Xs, y, yerr, tXs, ty, tyerr = load_data()
    model0 = make_model()
    D = sum(p.numel() for p in model0.parameters())
    print(f"device {DEV}, train {len(Xs):,}, test {len(tXs):,}, D = {D:,}, "
          f"s0 = {math.exp(LNS0):.4f}", flush=True)
    if mode == "smoke":
        run(3.5e-4, Xs, y, yerr, 600, out_file=TAB / "exp7s_smoke.npz")
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
    elif mode == "exact":
        dt_chain = float(sys.argv[2])
        n_traj = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
        L = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        exact(dt_chain, Xs, y, yerr, n_traj=n_traj, L=L)
    elif mode == "reportexact":
        report_exact(float(sys.argv[2]), tXs, ty, tyerr)
    print("EXP7S-DONE", flush=True)
