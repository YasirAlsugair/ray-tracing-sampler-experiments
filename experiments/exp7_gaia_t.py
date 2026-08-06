"""exp7 Student-t run: heavy tails with the per-star scale.

The converged per-star chain fixed the calibration WIDTHS (z std 0.99 to
1.09 across label-error bins) but not the SHAPE: z kurtosis 8.5 against
the Gaussian 3, with 118 stars past |z| = 4. A Gaussian with any width
cannot produce that. The motivated rung 3 replaces the Gaussian with a
Student-t of the same per-star scale:

    c_i^2 = err_i^2 + sigma(x_i)^2,   sigma(x) = exp(LNS0 + r(x))
    (y_i - mu(x_i)) / c_i  ~  t_nu

    nll_i = -lgamma((nu+1)/2) + lgamma(nu/2) + 0.5 ln(nu pi)
            + 0.5 ln c_i^2 + ((nu+1)/2) ln(1 + ((y_i-mu_i)/c_i)^2 / nu)

The 0.5 ln c_i^2 fee survives, so per-star claims stay earned. The df is
sampled like s was: ln(nu - 2) = NU0 + w with w ~ N(0,1) and NU0 = 1.6,
so the prior centers near nu = 7 and covers the 5..10 window within one
sigma; nu > 2 keeps the variance finite by construction. The hetero
chain's kurtosis 8.5 implies nu ~ 5.1 (kurt = 3 + 6/(nu-4)), so w starts
at the nu = 5 point, not at the prior center.

LAYOUT (the ordering-bug lesson, this time applied on purpose): torch
registers a module's direct parameters BEFORE its submodules, so with
self.w and self.net the flat vector is [w, net], with w at index 0.
build_seed constructs exactly that and round-trip checks it.

Warm start (mode "seed"): [w0] + the converged hetero chain-of-record
final state (exp7h_pack3.npz), with the sigma output bias (the LAST
element) shifted by 0.5 ln((nu0-2)/nu0). The shift variance-matches the
sigma(x)-dominated (majority) stars; the err^2 component of c^2 is fixed
data and still starts nu/(nu-2) high, up to ~20 percent total for the
largest-label-error stars. Burn-in plus the drift rule absorb that; the
shift is a head start, not an exactness statement. D = 11,395.

Everything else is the exp7 recipe verbatim: Glorot prior (bias sigma
0.1, w sigma 1.0), minibatch Raytracer, Eq. 33 gate every 30 steps with
honest softening. Batch defaults to 4096: the hetero campaign showed the
bigger batch halves sigma_sto and raises the gate ceiling without moving
the dt knee.

Modes:
    python exp7_gaia_t.py seed             # build exp7_t_seed.npz
    python exp7_gaia_t.py smoke            # 600 steps, local sanity
    python exp7_gaia_t.py auto             # tune, run 2M, report
    python exp7_gaia_t.py report DT
    python exp7_gaia_t.py converge DT [MAXLEGS]
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
BATCH = int(os.environ.get("EXP7T_BATCH", "4096"))
TEST_EVERY = 30
SNAPSHOT_EVERY = 250
SEED = 0
LNS0 = -3.0
NU0 = 1.6              # prior center of ln(nu - 2): nu ~ 7
W_START_NU = 5.0       # kurtosis-implied start (8.5 -> nu ~ 5.1)


def load_data():
    d = np.load(TAB / "exp7_gaia_pristine.npz")
    to = lambda k: torch.tensor(d[k], device=DEV)
    return (to("train_Xs"), to("train_y"), to("train_yerr"),
            to("test_Xs"), to("test_y"), to("test_yerr"))


class TModel(nn.Module):
    """Direct param w (the df) + the 110-64-64-2 net. Flat = [w, net]."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.w = nn.Parameter(torch.zeros(1))
        self.net = nn.Sequential(nn.Linear(110, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 2))

    def mu_r(self, X):
        out = self.net(X)
        return out[..., 0], out[..., 1]

    def nu(self):
        return 2.0 + torch.exp(NU0 + self.w[0])


def make_model():
    return TModel().to(DEV)


def sigmas_for(model):
    """Prior sigma per tensor: 1.0 for w, Glorot for weights, 0.1 biases."""
    out = []
    for name, p in model.named_parameters():
        if name == "w":
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


def build_seed():
    """[w0] + hetero chain-of-record final state, sigma bias shifted."""
    hstate = np.load(TAB / "exp7h_pack3.npz")["final_state"]
    assert len(hstate) == 11_394, len(hstate)
    w0 = math.log(W_START_NU - 2.0) - NU0
    shift = 0.5 * math.log((W_START_NU - 2.0) / W_START_NU)
    hstate = hstate.copy()
    hstate[-1] += shift                          # sigma output bias is last
    state = np.concatenate([[w0], hstate]).astype(np.float32)
    np.savez(TAB / "exp7_t_seed.npz", state=state)
    model = make_model()
    load_flat(model, state)
    back = torch.cat([p.detach().flatten()
                      for p in model.parameters()]).cpu().numpy()
    assert np.allclose(back, state, atol=1e-7), "flat round-trip failed"
    with torch.no_grad():
        nu = float(model.nu())
    print(f"seed built: D = {len(state):,}, w0 = {w0:+.4f} (nu = {nu:.2f}), "
          f"sigma bias shifted {shift:+.4f} so the t predictive variance "
          f"starts near the hetero fit", flush=True)


def seed_state():
    return np.load(TAB / "exp7_t_seed.npz")["state"]


def batch_nll_mean(model, Xs, y, yerr, batch):
    mu, r = model.mu_r(Xs[batch])
    c2 = yerr[batch] ** 2 + torch.exp(2.0 * (LNS0 + r))
    nu = model.nu()
    z2 = (y[batch] - mu) ** 2 / c2
    return (-torch.lgamma((nu + 1.0) / 2.0) + torch.lgamma(nu / 2.0)
            + 0.5 * torch.log(nu * math.pi)
            + 0.5 * torch.log(c2)
            + ((nu + 1.0) / 2.0) * torch.log1p(z2 / nu)).mean()


@torch.no_grad()
def exact_decomposition(model, sigmas, Xs, y, yerr):
    misfit, nu = 0.0, model.nu()
    for i in range(0, len(Xs), 20000):
        mu, r = model.mu_r(Xs[i:i + 20000])
        c2 = yerr[i:i + 20000] ** 2 + torch.exp(2.0 * (LNS0 + r))
        z2 = (y[i:i + 20000] - mu) ** 2 / c2
        misfit += (-torch.lgamma((nu + 1.0) / 2.0) + torch.lgamma(nu / 2.0)
                   + 0.5 * torch.log(nu * math.pi)
                   + 0.5 * torch.log(c2)
                   + ((nu + 1.0) / 2.0) * torch.log1p(z2 / nu)).sum().item()
    wnorm = sum(((p / sg) ** 2).sum().item()
                for p, sg in zip(model.parameters(), sigmas))
    return misfit, wnorm


@torch.no_grad()
def sigma_quantiles(model, Xs):
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
    sig_med, sig_lo, sig_hi, nus = [], [], [], []
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
            with torch.no_grad():
                nus.append(float(model.nu()))
            snapshots.append(torch.cat([p.detach().flatten()
                                        for p in model.parameters()])
                             .cpu().numpy().astype(np.float32))

    wall = time.time() - t0
    acceptance = float(np.mean(window_accepted)) if window_accepted else None
    if save:
        out = out_file or TAB / f"exp7t_rt33_dt{dt:g}.npz"
        np.savez(out, snapshots=np.stack(snapshots), steps=np.array(steps),
                 misfit=np.array(misfits), wnorm=np.array(wnorms),
                 sig_med=np.array(sig_med), sig_lo=np.array(sig_lo),
                 sig_hi=np.array(sig_hi), nu=np.array(nus),
                 window_raw=np.array(window_raw),
                 window_accepted=np.array(window_accepted),
                 sigma_sto=sigma, dt=dt, batch=BATCH,
                 test_every=TEST_EVERY, n_steps=n_steps, wall_s=wall)
        print(f"[dt={dt:g}] {n_steps:,} steps in {wall / 60:.1f} min -> "
              f"misfit {misfits[-1]:,.0f}, wnorm {wnorms[-1]:,.0f} of shell "
              f"D=11,395, nu {nus[-1]:.2f}, sigma(x) median {sig_med[-1]:.4f} "
              f"[{sig_lo[-1]:.4f}, {sig_hi[-1]:.4f}], acceptance "
              f"{'n/a' if acceptance is None else f'{acceptance:.2f}'}, "
              f"saved {Path(out).name}", flush=True)
    return acceptance


def drift_check(series):
    quarter = series[3 * len(series) // 4:]
    if len(quarter) < 8:            # too short to judge: never fake "level"
        return float("inf"), 0.0
    slope = np.polyfit(np.arange(len(quarter)), quarter, 1)[0]
    return slope * len(quarter), quarter.std()


def arm_files_t(dt):
    files = [TAB / f"exp7t_rt33_dt{dt:g}.npz"]
    part = 2
    while (TAB / f"exp7t_rt33_dt{dt:g}_part{part}.npz").exists():
        files.append(TAB / f"exp7t_rt33_dt{dt:g}_part{part}.npz")
        part += 1
    return files


def last_state(dt):
    files = arm_files_t(dt)
    if files[0].exists():
        return np.load(files[-1])["snapshots"][-1]
    return seed_state()


def converge(dt, Xs, y, yerr, leg_steps=500_000, max_legs=4):
    """Extend until misfit, weighted norm, median sigma(x), AND nu all
    pass the drift rule (nu is a fourth check here: the df must settle)."""
    model = make_model()
    load_flat(model, last_state(dt))
    sigma = estimate_sigma_sto(model, Xs, y, yerr, len(Xs))
    print(f"[converge dt={dt:g}] re-measured sigma_sto = {sigma:,.1f}",
          flush=True)

    def status():
        files = arm_files_t(dt)
        oks, notes = [], []
        for name in ("misfit", "wnorm", "sig_med", "nu"):
            series = np.concatenate([np.load(f)[name] for f in files])
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
        part = len(arm_files_t(dt)) + 1
        out = TAB / f"exp7t_rt33_dt{dt:g}_part{part}.npz"
        run(dt, Xs, y, yerr, leg_steps, initial_state=last_state(dt),
            sigma_fixed=sigma, out_file=out, seed=SEED + part)
        stationary, line = status()
        print(f"[converge dt={dt:g}] -> "
              f"{'STATIONARY' if stationary else 'in transit'}; {line}",
              flush=True)
    return stationary


def convergepack(dt, Xs, y, yerr, leg_steps=500_000, max_legs=6):
    """Resume convergence from exp7t_pack.npz when the raw chain files are
    gone (pod terminated): the pack's final_state seeds the continuation and
    its concatenated traces are the drift-check history. New legs land in
    _cont{k} files. Ported from the hetero driver after the 2026-08-06
    termination proved the review's point."""
    pack = np.load(TAB / "exp7t_pack.npz")

    def cont_files():
        files, k = [], 1
        while (TAB / f"exp7t_rt33_dt{dt:g}_cont{k}.npz").exists():
            files.append(TAB / f"exp7t_rt33_dt{dt:g}_cont{k}.npz")
            k += 1
        return files

    def cur_state():
        f = cont_files()
        return np.load(f[-1])["snapshots"][-1] if f else pack["final_state"]

    model = make_model()
    load_flat(model, cur_state())
    sigma = estimate_sigma_sto(model, Xs, y, yerr, len(Xs))
    print(f"[convergepack dt={dt:g}] resumed from pack final state "
          f"({int(pack['n_steps_total']):,} steps of history), "
          f"sigma_sto = {sigma:,.1f}", flush=True)

    def status():
        f = cont_files()
        oks, notes = [], []
        for name in ("misfit", "wnorm", "sig_med", "nu"):
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
        out = TAB / f"exp7t_rt33_dt{dt:g}_cont{k}.npz"
        run(dt, Xs, y, yerr, leg_steps, initial_state=cur_state(),
            sigma_fixed=sigma, out_file=out, seed=SEED + 20 + k)
        stationary, line = status()
        print(f"[convergepack dt={dt:g}] -> "
              f"{'STATIONARY' if stationary else 'in transit'}; {line}",
              flush=True)
    return stationary


@torch.no_grad()
def report(dt, Xs, y, yerr, tXs, ty, tyerr, members=50):
    files = arm_files_t(dt)
    snaps = np.concatenate([np.load(f)["snapshots"] for f in files])
    quarter = snaps[3 * len(snaps) // 4:]
    take = np.linspace(0, len(quarter) - 1, min(members, len(quarter))).astype(int)
    model = make_model()
    mus, sigs, nus = [], [], []
    for state in quarter[take]:
        load_flat(model, state)
        mu, r = model.mu_r(tXs)
        mus.append(mu)
        sigs.append(torch.exp(LNS0 + r))
        nus.append(float(model.nu()))
    mus = torch.stack(mus)
    sigs = torch.stack(sigs)
    nus = np.array(nus)
    mu_hat, tau_hat = mus.mean(dim=0), mus.std(dim=0)
    sig_hat = sigs.median(dim=0).values
    nu_hat = float(np.median(nus))
    rmse = float(((mu_hat - ty) ** 2).mean().sqrt())

    for name in ("misfit", "wnorm", "sig_med", "nu"):
        series = np.concatenate([np.load(f)[name] for f in files])
        dr, no = drift_check(series)
        print(f"drift check {name}: drift {dr:+.4g} vs noise {no:.4g} "
              f"({'level' if abs(dr) < 2 * no else 'moving'})")

    print(f"nu over members: median {nu_hat:.2f}, "
          f"16/84 pct [{np.percentile(nus, 16):.2f}, "
          f"{np.percentile(nus, 84):.2f}]  "
          f"(kurtosis-implied guess was ~5.1; prior center 7)")
    sh = sig_hat.cpu().numpy()
    print(f"sigma(x) over test stars: median {np.median(sh):.4f}, "
          f"16/84 pct [{np.percentile(sh, 16):.4f}, "
          f"{np.percentile(sh, 84):.4f}]  (hetero chain was 0.0322)")
    print(f"RMSE chain {rmse:.4f}  (hetero 0.0494, scatter 0.0466); "
          f"median tau {float(tau_hat.median()):.4f}")

    # variance-standardized z: std ~ 1 if variance-calibrated (variances of
    # the member mixture add exactly; the SHAPE test is the PIT below)
    c2 = tyerr ** 2 + sig_hat ** 2
    var_tot = tau_hat ** 2 + c2 * nu_hat / (nu_hat - 2.0)
    zv = ((ty - mu_hat) / var_tot.sqrt()).cpu().numpy()
    print(f"variance-z std {zv.std():.3f} (1.0 = variance-calibrated)")

    # the shape test: member-mixture PIT. Exact under the sampled
    # posterior: each star's percentile is averaged over every member's
    # own t (its mu, its sigma(x), its nu); no Gaussian-only
    # variance-folding of tau (that shortcut is wrong for a t).
    try:
        from scipy import stats
        y_np = ty.cpu().numpy()
        yerr2 = (tyerr ** 2).cpu().numpy()
        mus_np = mus.cpu().numpy()
        sigs_np = sigs.cpu().numpy()
        pit = np.zeros(len(y_np))
        for m in range(len(mus_np)):
            c_m = np.sqrt(yerr2 + sigs_np[m] ** 2)
            pit += stats.t.cdf((y_np - mus_np[m]) / c_m, df=nus[m])
        pit /= len(mus_np)
        print("PIT coverage (fraction inside the central interval; "
              "calibrated shape matches the nominal):")
        for nominal in (0.6827, 0.9545, 0.9973):
            inside = int((np.abs(pit - 0.5) < nominal / 2.0).sum())
            print(f"  nominal {nominal:.4f}  ->  {inside / len(pit):.4f}")
        p4 = 2.0 * stats.norm.sf(4.0)          # Gaussian |z|>4 tail prob
        p_two = 2.0 * np.minimum(pit, 1.0 - pit)
        n_tail = int((p_two < p4).sum())
        print(f"stars beyond the |z|=4-equivalent tail probability: "
              f"{n_tail}  (chance expectation ~{p4 * len(pit):.1f}; the "
              f"Gaussian hetero chain had 118)")
    except ImportError:
        print("scipy not present: skipping the PIT table")

    zc = zv
    te = tyerr.cpu().numpy()
    print("variance-z std by label error (hetero ran 0.99 to 1.09):")
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
    print(f"device {DEV}, train {len(Xs):,}, test {len(tXs):,}, D = {D:,}, "
          f"batch {BATCH}", flush=True)
    if mode == "seed":
        build_seed()
    elif mode == "smoke":
        run(3.5e-4, Xs, y, yerr, 600, out_file=TAB / "exp7t_smoke.npz")
    elif mode == "auto":
        # knee rule: stop when the gain flattens, but ONLY once acceptance
        # is alive; a 0.00 -> 0.00 plateau at the top of the ladder is
        # wholesale rejection, not a knee (bug caught 2026-08-05: it chose
        # the largest fully-rejected dt)
        accs, chosen = [], None
        for dt in (1e-4, 5e-5, 2e-5, 1e-5, 5e-6):
            acc = run(dt, Xs, y, yerr, 3000, save=False)
            print(f"[tune] dt={dt:g}  Eq.33 acceptance={acc:.2f}", flush=True)
            accs.append((dt, acc))
            if (len(accs) >= 2 and accs[-2][1] > 0.05
                    and acc - accs[-2][1] < 0.02):
                chosen = accs[-2][0]
                break
        if chosen is None:
            chosen = max(accs, key=lambda t: t[1])[0]
        print(f"[tune] chosen dt={chosen:g}", flush=True)
        # production runs as 500k legs (the first + converge parts) so a
        # dead pod costs one leg, not the campaign: the pod-loss lesson
        run(chosen, Xs, y, yerr, 500_000)
        converge(chosen, Xs, y, yerr, max_legs=7)
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
    elif mode == "pack":
        # small pullable pack: final state + traces + 50 members, so flaky
        # links and terminated pods cannot orphan the campaign
        dt = float(sys.argv[2])
        files = arm_files_t(dt)
        snaps = np.concatenate([np.load(f)["snapshots"] for f in files])
        quarter = snaps[3 * len(snaps) // 4:]
        take = np.linspace(0, len(quarter) - 1,
                           min(50, len(quarter))).astype(int)
        packed = {k: np.concatenate([np.load(f)[k] for f in files])
                  for k in ("misfit", "wnorm", "sig_med", "nu")}
        n_total = int(sum(np.load(f)["n_steps"] for f in files))
        np.savez(TAB / "exp7t_pack.npz", members=quarter[take],
                 final_state=snaps[-1], n_steps_total=n_total,
                 dt=dt, batch=BATCH, **packed)
        print(f"packed {len(take)} members, {n_total:,} total steps -> "
              f"exp7t_pack.npz", flush=True)
    print("EXP7T-DONE", flush=True)
