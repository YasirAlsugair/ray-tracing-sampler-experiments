"""Unadjusted minibatch ray tracing on the MLP posterior: the tuning sweep.

Keeps the minibatch dynamics and drops the Metropolis test (a batch-256 estimate
of the accept ratio carries ~270 nats of noise against a ~10 nat signal, so the
test would decide at random). Error is controlled with the knobs instead: step
size down, refresh rate up. In this continuous mode there are no discrete
trajectories; the effective trajectory length is about 1/(refresh_rate * dt)
steps between momentum decorrelations. The exact full-batch chain is the ground
truth the arms are compared against.

Arms: dt in {3.5e-4, 1e-4, 3e-5} x refresh_rate in {5, 50}, batch 256, 100k
steps each, all from the step 1 checkpoint at the derived temperature
(scale_likelihood = N = 60,000 on the mean loss; the N(0,1) prior enters as an
explicit theta/N addition to the gradient).

Output per arm: results/tables/exp6_mb_dt{...}_rr{...}.npz with parameter
snapshots every 250 steps and the exact full-data decomposition (sum CE,
||theta||^2) at every snapshot.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import exp6_simple_mnist_train as base

sys.path.insert(0, str(base.ROOT / "vendor" / "ray-tracing-sampler"))
from raytrace_torch import Raytracer

DEV = base.DEV
TAB = base.TAB
BATCH = 256
N_STEPS = 100_000
SNAPSHOT_EVERY = 250
SEED = 0


def full_train_tensors():
    loader, _, n = base.loaders()
    chunks = torch.utils.data.DataLoader(loader.dataset, batch_size=10000)
    images = torch.cat([x for x, _ in chunks]).to(DEV)
    chunks = torch.utils.data.DataLoader(loader.dataset, batch_size=10000)
    labels = torch.cat([y for _, y in chunks]).to(DEV)
    return images, labels, n


@torch.no_grad()
def exact_decomposition(model, images, labels):
    """Full-data sum CE and squared parameter norm, both in nats."""
    misfit = 0.0
    for i in range(0, len(images), 10000):
        misfit += F.cross_entropy(model(images[i:i + 10000]),
                                  labels[i:i + 10000], reduction="sum").item()
    squared_norm = sum((p ** 2).sum().item() for p in model.parameters())
    return misfit, squared_norm


def estimate_sigma_sto(model, images, labels, n_train, draws=50):
    """Std of the batch estimate of ln L at the current state (paper Eq. 33)."""
    stream = torch.Generator().manual_seed(SEED + 7)
    values = []
    with torch.no_grad():
        for _ in range(draws):
            batch = torch.randint(0, n_train, (BATCH,), generator=stream).to(DEV)
            values.append(-n_train * F.cross_entropy(model(images[batch]),
                                                     labels[batch]).item())
    return float(np.std(values))


def batch_log_posterior(model, images, labels, batch_stream, n_train):
    """One-minibatch (noisy) estimate of the log posterior at the current state."""
    batch = torch.randint(0, n_train, (BATCH,), generator=batch_stream).to(DEV)
    with torch.no_grad():
        ce_mean = F.cross_entropy(model(images[batch]), labels[batch]).item()
        squared_norm = sum((p ** 2).sum().item() for p in model.parameters())
    return -(n_train * ce_mean + squared_norm / 2)


ARCH_CLASSES = {"mlp": None, "cnn": None}  # filled after base import below


def model_for(arch):
    cls = {"mlp": base.MLP, "cnn": base.CNN}[arch]
    model = cls().to(DEV)
    model.load_state_dict(torch.load(base.CKPT / f"exp6_{arch}.pt"))
    return model


def run_arm(dt, refresh_rate, images, labels, n_train, gated=False,
            test_every=30, n_steps=None, save=True, criterion="eq33",
            initial_state=None, sigma_fixed=None, out_file=None, seed=SEED,
            arch="mlp"):
    """criterion="eq33": the paper's noise-softened acceptance (exponent
    1/sqrt(1+sigma^2)); criterion="eq19": the plain Metropolis ratio applied to
    the same noisy estimates (exponent 1). Returns the gate's acceptance
    fraction when gated, else None.

    initial_state: flat parameter vector to start from instead of the step 1
    checkpoint (used to continue a saved arm). sigma_fixed: reuse a stored
    sigma_sto so a continued arm keeps the exact gate it started with.
    out_file: override the output path (continuation legs)."""
    if n_steps is None:
        n_steps = N_STEPS
    torch.manual_seed(seed)
    batch_stream = torch.Generator().manual_seed(seed + 99)  # same stream per arm
    accept_stream = np.random.default_rng(seed + 13)
    model = model_for(arch)
    if initial_state is not None:
        flat = torch.tensor(initial_state, dtype=torch.float32, device=DEV)
        offset = 0
        with torch.no_grad():
            for parameter in model.parameters():
                n = parameter.numel()
                parameter.copy_(flat[offset:offset + n].view(parameter.shape))
                offset += n
    # The N(0,1) prior enters as an explicit theta/N term added to the mean-loss
    # gradient, matching the exact chains' target definition. (The optimizer's
    # own weight_decay path is a volume-tracked contraction, a different object,
    # and it also trips a torch 2.12 incompatibility.)
    sampler = Raytracer(model.parameters(), dt=dt,
                        scale_likelihood=float(n_train),
                        refresh_rate=refresh_rate)
    if gated:
        sigma = (sigma_fixed if sigma_fixed is not None
                 else estimate_sigma_sto(model, images, labels, n_train))
        if criterion == "eq33":
            softening = 1.0 / np.sqrt(1.0 + sigma ** 2)  # the Eq. 33 exponent
        else:
            softening = 1.0                              # plain Eq. 19 on noise
        if save:
            print(f"[dt={dt:g} rr={refresh_rate:g}] sigma_sto = {sigma:,.0f} "
                  f"nats -> {criterion} exponent {softening:.2e}", flush=True)
        window_raw, window_accepted = [], []
        n_rejects = 0
        window_start_state = [p.detach().clone() for p in model.parameters()]
        window_start_estimate = batch_log_posterior(model, images, labels,
                                                    batch_stream, n_train)
        window_start_ledger = 0.0

    snapshots, misfits, norms, steps, luminosities = [], [], [], [], []
    t0 = time.time()
    for step in range(n_steps):
        batch = torch.randint(0, n_train, (BATCH,), generator=batch_stream).to(DEV)
        sampler.zero_grad()
        F.cross_entropy(model(images[batch]), labels[batch]).backward()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.grad.add_(parameter, alpha=1.0 / n_train)
        sampler.step()

        if gated and (step + 1) % test_every == 0:
            end_estimate = batch_log_posterior(model, images, labels,
                                               batch_stream, n_train)
            ledger = float(sampler.param_groups[0]["ln_luminosity"])
            raw = (end_estimate - window_start_estimate) \
                - (ledger - window_start_ledger)
            accept_probability = min(1.0, np.exp(min(0.0, raw * softening)))
            accepted = bool(accept_stream.random() < accept_probability)
            window_raw.append(raw)
            window_accepted.append(accepted)
            if accepted:
                with torch.no_grad():
                    for parameter, kept in zip(model.parameters(),
                                               window_start_state):
                        kept.copy_(parameter)
                window_start_estimate = end_estimate
            else:
                n_rejects += 1
                with torch.no_grad():
                    for parameter, kept in zip(model.parameters(),
                                               window_start_state):
                        parameter.copy_(kept)
                    for parameter in model.parameters():
                        state = sampler.state[parameter]
                        if "momenta" in state:
                            state["momenta"].normal_(0.0, 1.0)
                # the restored state keeps its original estimate; refreshing it
                # here would slowly bias the gate toward accepting
            window_start_ledger = ledger

        if save and (step + 1) % SNAPSHOT_EVERY == 0:
            misfit, squared_norm = exact_decomposition(model, images, labels)
            steps.append(step + 1)
            misfits.append(misfit)
            norms.append(squared_norm)
            # the sampler's running luminosity ledger: with the exact
            # log-posterior at the same snapshots this gives the shadow
            # Metropolis verdict per window (see the notebook)
            luminosities.append(float(sampler.param_groups[0]["ln_luminosity"]))
            snapshots.append(torch.cat([p.detach().flatten()
                                        for p in model.parameters()])
                             .cpu().numpy().astype(np.float32))
    wall = time.time() - t0
    acceptance_fraction = (float(np.mean(window_accepted))
                           if gated and window_accepted else None)
    if save:
        prefix = {False: "exp6_mb"}.get(gated) or \
            ("exp6_mb33" if criterion == "eq33" else "exp6_mb19")
        if arch != "mlp":
            prefix += f"_{arch}"
        out = out_file or TAB / f"{prefix}_dt{dt:g}_rr{refresh_rate:g}.npz"
        extras = {}
        if gated:
            extras = dict(window_raw=np.array(window_raw),
                          window_accepted=np.array(window_accepted),
                          sigma_sto=sigma, test_every=test_every,
                          criterion=criterion)
        np.savez(out, snapshots=np.stack(snapshots), steps=np.array(steps),
                 misfit=np.array(misfits), norm=np.array(norms),
                 ln_luminosity=np.array(luminosities),
                 dt=dt, refresh_rate=refresh_rate, batch=BATCH,
                 n_steps=n_steps, wall_s=wall, **extras)
        gate_note = ""
        if gated:
            gate_note = (f", {criterion} rejected "
                         f"{n_rejects}/{len(window_accepted)} windows")
        print(f"[dt={dt:g} rr={refresh_rate:g}] {n_steps:,} steps in "
              f"{wall / 60:.1f} min -> misfit {misfits[-1]:,.0f}, "
              f"||theta||^2 {norms[-1]:,.0f}{gate_note}, saved {out.name}",
              flush=True)
    return acceptance_fraction


def arm_files(dt, refresh_rate, criterion="eq33", arch="mlp"):
    """The base arm file plus any continuation legs, in chain order."""
    prefix = "exp6_mb33" if criterion == "eq33" else "exp6_mb19"
    if arch != "mlp":
        prefix += f"_{arch}"
    files = [TAB / f"{prefix}_dt{dt:g}_rr{refresh_rate:g}.npz"]
    part = 2
    while (TAB / f"{prefix}_dt{dt:g}_rr{refresh_rate:g}_part{part}.npz").exists():
        files.append(TAB / f"{prefix}_dt{dt:g}_rr{refresh_rate:g}_part{part}.npz")
        part += 1
    return files


def drift_check(series):
    """Final-quarter drift vs noise, the notebook's stationarity rule."""
    quarter = series[3 * len(series) // 4:]
    slope = np.polyfit(np.arange(len(quarter)), quarter, 1)[0]
    drift = slope * len(quarter)
    noise = quarter.std()
    return drift, noise, abs(drift) < 2 * noise


def converge_gated(dt, refresh_rate, images, labels, n_train,
                   leg_steps=250_000, max_legs=10, criterion="eq33",
                   arch="mlp"):
    """Extend a saved gated arm from its last snapshot in legs, stopping when
    BOTH the exact misfit and the weight norm pass the drift check on the
    combined snapshot series (a single leveled curve is not enough: the
    unadjusted sweep showed the misfit levels long before the norm stops
    marching). The stored sigma_sto is reused so every leg runs the same gate
    the arm started with. Where the norm levels relative to the shell at
    D = 50,890 is the measurement: the gate's stationary point need not be
    the posterior's."""
    files = arm_files(dt, refresh_rate, criterion, arch)
    shell = sum(p.numel() for p in model_for(arch).parameters())
    base_archive = np.load(files[0])
    sigma = float(base_archive["sigma_sto"])
    misfits = [np.asarray(np.load(f)["misfit"]) for f in files]
    norms = [np.asarray(np.load(f)["norm"]) for f in files]
    last_state = np.load(files[-1])["snapshots"][-1]
    total_steps = sum(int(np.load(f)["n_steps"]) for f in files)

    def status():
        m_drift, m_noise, m_ok = drift_check(np.concatenate(misfits))
        n_drift, n_noise, n_ok = drift_check(np.concatenate(norms))
        return (m_ok and n_ok,
                f"misfit {np.concatenate(misfits)[-1]:,.0f} "
                f"(drift {m_drift:+,.0f} vs noise {m_noise:,.0f}, "
                f"{'level' if m_ok else 'moving'}), "
                f"norm {np.concatenate(norms)[-1]:,.0f} of shell {shell:,} "
                f"(drift {n_drift:+,.0f} vs noise {n_noise:,.0f}, "
                f"{'level' if n_ok else 'moving'})")

    stationary, line = status()
    print(f"[{criterion} dt={dt:g} rr={refresh_rate:g}] resuming at step "
          f"{total_steps:,} with stored sigma_sto={sigma:,.0f}; {line}",
          flush=True)

    for _ in range(max_legs):
        if stationary:
            break
        part = len(arm_files(dt, refresh_rate, criterion, arch)) + 1
        prefix = "exp6_mb33" if criterion == "eq33" else "exp6_mb19"
        if arch != "mlp":
            prefix += f"_{arch}"
        out = TAB / f"{prefix}_dt{dt:g}_rr{refresh_rate:g}_part{part}.npz"
        run_arm(dt, refresh_rate, images, labels, n_train, gated=True,
                n_steps=leg_steps, criterion=criterion,
                initial_state=last_state, sigma_fixed=sigma, out_file=out,
                seed=SEED + part, arch=arch)  # a fresh stream per leg
        archive = np.load(out)
        misfits.append(np.asarray(archive["misfit"]))
        norms.append(np.asarray(archive["norm"]))
        last_state = archive["snapshots"][-1]
        total_steps += leg_steps
        stationary, line = status()
        verdict = "STATIONARY on both curves" if stationary else "still in transit"
        print(f"[{criterion} dt={dt:g} rr={refresh_rate:g}] step "
              f"{total_steps:,} -> {verdict}; {line}", flush=True)

    if not stationary:
        print(f"[{criterion} dt={dt:g} rr={refresh_rate:g}] max legs reached, "
              f"still in transit at step {total_steps:,}", flush=True)


def tune_step_size_by_eq33(images, labels, n_train, probe_steps=3000):
    """The paper's recipe: lower the step size until the Eq. 33 acceptance
    rate stops improving significantly."""
    IMPROVEMENT_FLOOR = 0.02
    candidates = (3.5e-4, 2e-4, 1e-4, 5e-5, 3e-5, 1.5e-5)
    for refresh_rate in (5, 50):
        previous_acceptance, previous_dt, chosen = None, None, None
        for dt in candidates:
            acceptance = run_arm(dt, refresh_rate, images, labels, n_train,
                                 gated=True, n_steps=probe_steps, save=False)
            print(f"[tune33 rr={refresh_rate:g}] dt={dt:g}  "
                  f"Eq.33 acceptance={acceptance:.2f}", flush=True)
            if (previous_acceptance is not None
                    and acceptance - previous_acceptance < IMPROVEMENT_FLOOR):
                chosen = previous_dt   # lowering stopped helping: keep the
                break                  # largest step size on the plateau
            previous_acceptance, previous_dt = acceptance, dt
        if chosen is None:
            chosen = previous_dt
        print(f"[tune33 rr={refresh_rate:g}] chosen dt={chosen:g} "
              f"(acceptance stopped improving below +{IMPROVEMENT_FLOOR})",
              flush=True)


def cnn_pipeline(leg_steps=250_000, max_legs=20):
    """The full CNN minibatch program in one command: recipe-tune the step
    size at refresh 5, lay down the 100k-step Eq. 33 base arm, then extend
    in legs until the drift rule passes on both curves."""
    images, labels, n_train = full_train_tensors()
    IMPROVEMENT_FLOOR = 0.02
    refresh_rate = 5
    previous_acceptance, previous_dt, chosen = None, None, None
    for dt in (3.5e-4, 2e-4, 1e-4, 5e-5, 3e-5, 1.5e-5):
        acceptance = run_arm(dt, refresh_rate, images, labels, n_train,
                             gated=True, n_steps=3000, save=False, arch="cnn")
        print(f"[cnn tune33 rr={refresh_rate:g}] dt={dt:g}  "
              f"Eq.33 acceptance={acceptance:.2f}", flush=True)
        if (previous_acceptance is not None
                and acceptance - previous_acceptance < IMPROVEMENT_FLOOR):
            chosen = previous_dt
            break
        previous_acceptance, previous_dt = acceptance, dt
    if chosen is None:
        chosen = previous_dt
    print(f"[cnn tune33] chosen dt={chosen:g}", flush=True)
    run_arm(chosen, refresh_rate, images, labels, n_train, gated=True,
            arch="cnn")
    converge_gated(chosen, refresh_rate, images, labels, n_train,
                   leg_steps=leg_steps, max_legs=max_legs, arch="cnn")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "unadjusted"
    if mode != "cnn33":
        images, labels, n_train = full_train_tensors()
    else:
        images = labels = n_train = None
    if mode == "cnn33":
        cnn_pipeline(
            leg_steps=int(sys.argv[2]) if len(sys.argv) > 2 else 250_000,
            max_legs=int(sys.argv[3]) if len(sys.argv) > 3 else 20)
        raise SystemExit
    if mode == "tune33":
        tune_step_size_by_eq33(images, labels, n_train)
    elif mode == "converge33":
        converge_gated(float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4,
                       float(sys.argv[3]) if len(sys.argv) > 3 else 5,
                       images, labels, n_train,
                       leg_steps=int(sys.argv[4]) if len(sys.argv) > 4 else 250_000,
                       max_legs=int(sys.argv[5]) if len(sys.argv) > 5 else 10)
    else:
        for dt in (3.5e-4, 1e-4, 3e-5):
            for refresh_rate in (5, 50):
                run_arm(dt, refresh_rate, images, labels, n_train,
                        gated=(mode in ("eq33", "eq19")), criterion=mode)
    print("sweep complete", flush=True)
