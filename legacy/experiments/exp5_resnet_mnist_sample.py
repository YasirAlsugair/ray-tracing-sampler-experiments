"""Exp 5b: sample the ResNet-50 MNIST weight posterior with ray tracing, HMC baseline.

Question (Josh's request): train a ResNet-50 on MNIST, then sample from it. This script
does the sampling stage, mirroring the paper's neural-network recipe (Behroozi 2025,
App. E / Sec. 3.4): the vendor Raytracer optimizer is the sampler, one fresh minibatch
per step supplies the stochastic gradient, there is NO Metropolis test (the paper drops
it once gradient noise dominates, Sec. 2.11.3), and bad samples are masked post hoc by
a validation-loss threshold (the GPT-2 masking rule). This is therefore approximate
MCMC with fixed hyperparameters, exactly as the paper frames it, not an exact sampler.

Likelihood: ln L = -SCALE * (mean CE per image), flat prior, no augmentation. SCALE
follows the ResNet-34 regime of the paper (D_eff ~ N_train, since N_train << D):
SCALE = N_train / (2 * TARGET_DLOSS), so the posterior should sit ~TARGET_DLOSS nats
of CE above the trained point if the effective dimension really is ~N_train. Watching
where the loss actually plateaus is part of the result (it measures D_eff).

The net runs in eval() mode throughout, so BatchNorm buffers are frozen at their
end-of-training values and the posterior is well-defined. HMC mode of the vendor code
silently redraws momenta when kinetic energy exceeds 100x equilibrium; we log kinetic
energy every step so those resets (and any stochastic-gradient heating) are visible
rather than hidden.

The dt pilot sweep below is the paper's hand-tuning step ("test a range of timesteps,
see where trajectories fail"). It is the concrete surface the auto-tuning project aims
to replace.

Outputs:
  results/tables/exp5_trace_rt.npz, exp5_trace_hmc.npz   (full traces)
  results/tables/exp5_summary.json, exp5_summary.csv
  results/checkpoints/exp5_sample_{rt,hmc}_final.pt
  results/figures/exp5_pilot.png, exp5_loss_traces.png, exp5_kinetic.png,
                  exp5_function_tau.png
"""
import sys, os, json, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "ray-tracing-sampler"))
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.models import resnet50
from raytrace_torch import Raytracer
from rts import metrics
from plots import style
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
CKPT = os.path.join(HERE, "..", "results", "checkpoints")
FIG = os.path.join(HERE, "..", "results", "figures")
TAB = os.path.join(HERE, "..", "results", "tables")

SEED = 20260706
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

BATCH = 256              # minibatch per sampler step (the stochastic-gradient source)
N_TRAIN = 60000
TARGET_DLOSS = 0.1       # desired posterior rise in mean CE (nats/image)
SCALE = N_TRAIN / (2 * TARGET_DLOSS)   # = 3e5, paper's D_eff ~ N_train regime
REFRESH = 5              # partial momentum refresh rate, paper App. E.1 uses f = 5 dt
HMC_PRE_STEPS = 300      # HMC pre-burn-in with strong refresh (paper: f = 50 dt)
HMC_PRE_REFRESH = 50

DT_GRID = [5e-7, 1e-6, 2e-6, 5e-6, 1e-5]   # paper ResNet-34 (D=22M) used 2e-6
PILOT_STEPS = 150
PILOT_FAIL_RISE = 1.0    # nats of CE above seed on the train monitor batch = failed

N_STEPS = 3000
EVAL_EVERY = 10          # monitor/probe cadence (steps)
BURN_STEPS = 1000        # analysis burn-in; also where ensemble collection starts
ENSEMBLE_EVERY = 250
N_MONITOR = 2048         # fixed train and val monitor batches
N_PROBE = 256            # fixed probe images, full class probs stored (function space)


def load_split(train):
    ds = datasets.MNIST(DATA, train=train, download=True)
    x = ds.data.float().div_(255.0).sub_(MNIST_MEAN).div_(MNIST_STD).unsqueeze(1)
    return x.to(DEVICE), ds.targets.to(DEVICE)


def forward(model, x):
    return model(x.expand(x.shape[0], 3, 28, 28))


def build_model(state_cpu):
    model = resnet50(weights=None, num_classes=10).to(DEVICE)
    model.load_state_dict(state_cpu)
    model.eval()                      # freeze BN buffers: posterior is well-defined
    for p in model.parameters():
        p.requires_grad_(True)
    return model


@torch.no_grad()
def ce_acc(model, x, y, batch=1024):
    ce_sum, correct = 0.0, 0
    for i in range(0, len(x), batch):
        logits = forward(model, x[i:i + batch])
        ce_sum += F.cross_entropy(logits, y[i:i + batch], reduction="sum").item()
        correct += (logits.argmax(1) == y[i:i + batch]).sum().item()
    return ce_sum / len(x), correct / len(x)


@torch.no_grad()
def test_probs(model, x, batch=1024):
    out = []
    for i in range(0, len(x), batch):
        out.append(F.softmax(forward(model, x[i:i + batch]), dim=1))
    return torch.cat(out)


def nll_ece(probs, y, bins=15):
    p_true = probs[torch.arange(len(y)), y].clamp_min(1e-12)
    nll = -p_true.log().mean().item()
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1, device=probs.device)
    ece = 0.0
    for b in range(bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.any():
            ece += (m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()).item()
    return nll, ece


def make_sampler(model, kind, dt, refresh):
    return Raytracer(model.parameters(), dt=dt, scale_likelihood=SCALE,
                     refresh_rate=refresh, stochastic_hmc=(kind == "hmc"))


def run_chain(kind, dt, n_steps, state_cpu, xtr, ytr, mon, collect=True):
    """One minibatch-gradient chain from the trained point. Returns trace dict."""
    torch.manual_seed(SEED + (1 if kind == "hmc" else 0))   # momenta init
    g = torch.Generator().manual_seed(SEED + 99)            # same batch stream per kind
    model = build_model(state_cpu)
    refresh0 = HMC_PRE_REFRESH if (kind == "hmc" and collect) else REFRESH
    opt = make_sampler(model, kind, dt, refresh0)

    tr = {"step_loss": np.zeros(n_steps), "ke": np.zeros(n_steps),
          "lnlum": np.zeros(n_steps), "eval_steps": [], "train_mon_ce": [],
          "val_ce": [], "val_acc": [], "probe_probs": [],
          "ens_steps": [], "ens_val_ce": [], "ens_test_acc": [], "diverged": False}
    ens_prob_sum, ens_n = None, 0
    t0 = time.time()

    for step in range(n_steps):
        if kind == "hmc" and collect and step == HMC_PRE_STEPS:
            opt.set_refresh_rate(REFRESH)
        idx = torch.randint(0, N_TRAIN, (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(model, xtr[idx.to(DEVICE)]), ytr[idx.to(DEVICE)])
        loss.backward()
        opt.step()
        lval = loss.item()
        tr["step_loss"][step] = lval
        tr["ke"][step] = float(opt.param_groups[0]["mom_loss"])
        tr["lnlum"][step] = float(opt.param_groups[0]["ln_luminosity"])
        if not np.isfinite(lval):
            print(f"  [{kind}] diverged (non-finite loss) at step {step}")
            tr["diverged"] = True
            break

        if collect and (step + 1) % EVAL_EVERY == 0:
            with torch.no_grad():
                mon_logits = forward(model, mon["xtr"])
                val_logits = forward(model, mon["xva"])
            tr["eval_steps"].append(step + 1)
            tr["train_mon_ce"].append(F.cross_entropy(mon_logits, mon["ytr"]).item())
            tr["val_ce"].append(F.cross_entropy(val_logits, mon["yva"]).item())
            tr["val_acc"].append((val_logits.argmax(1) == mon["yva"]).float().mean().item())
            probe = F.softmax(val_logits[:N_PROBE], dim=1)
            tr["probe_probs"].append(probe.cpu().numpy().astype(np.float16))

        if collect and step + 1 >= BURN_STEPS and (step + 1) % ENSEMBLE_EVERY == 0:
            probs = test_probs(model, mon["xte"])
            acc = (probs.argmax(1) == mon["yte"]).float().mean().item()
            with torch.no_grad():
                vce = F.cross_entropy(forward(model, mon["xva"]), mon["yva"]).item()
            tr["ens_steps"].append(step + 1)
            tr["ens_val_ce"].append(vce)
            tr["ens_test_acc"].append(acc)
            if vce <= mon["mask_threshold"]:
                ens_prob_sum = probs if ens_prob_sum is None else ens_prob_sum + probs
                ens_n += 1
        if collect and (step + 1) % 100 == 0:
            print(f"  [{kind}] step {step+1}/{n_steps}  batch_ce {lval:.4f}  "
                  f"KE/D {tr['ke'][step]/mon['D']:.3f}  "
                  f"({(step+1)/(time.time()-t0):.2f} it/s)", flush=True)

    tr["seconds"] = time.time() - t0
    tr["ens_prob_mean"] = None if ens_n == 0 else (ens_prob_sum / ens_n)
    tr["ens_n_used"] = ens_n
    tr["model"] = model
    return tr


def pilot(kind, state_cpu, xtr, ytr, mon, seed_mon_ce):
    """Paper's hand-tuning: largest dt whose short trajectory does not fail."""
    rows = []
    for dt in DT_GRID:
        t = run_chain(kind, dt, PILOT_STEPS, state_cpu, xtr, ytr, mon, collect=False)
        model = t["model"]
        with torch.no_grad():
            ce = F.cross_entropy(forward(model, mon["xtr"]), mon["ytr"]).item()
        ok = np.isfinite(ce) and not t["diverged"] and ce < seed_mon_ce + PILOT_FAIL_RISE
        rows.append({"dt": dt, "final_mon_ce": ce if np.isfinite(ce) else float("inf"),
                     "ok": bool(ok)})
        print(f"  pilot [{kind}] dt {dt:.1e}  mon_ce {ce:.4f}  {'ok' if ok else 'FAIL'}")
        del model, t
    ok_dts = [r["dt"] for r in rows if r["ok"]]
    chosen = max(ok_dts) if ok_dts else min(DT_GRID)
    return chosen, rows


def main():
    xtr, ytr = load_split(train=True)
    xte, yte = load_split(train=False)
    state_cpu = torch.load(os.path.join(CKPT, "exp5_resnet50_mnist.pt"),
                           map_location="cpu", weights_only=True)

    model0 = build_model(state_cpu)
    D = sum(p.numel() for p in model0.parameters())
    torch.manual_seed(SEED + 5)
    mon_tr_idx = torch.randperm(N_TRAIN)[:N_MONITOR].to(DEVICE)
    mon = {"xtr": xtr[mon_tr_idx], "ytr": ytr[mon_tr_idx],
           "xva": xte[:N_MONITOR], "yva": yte[:N_MONITOR],
           "xte": xte, "yte": yte, "D": D}

    with torch.no_grad():
        seed_mon_ce = F.cross_entropy(forward(model0, mon["xtr"]), mon["ytr"]).item()
        val_logits = forward(model0, mon["xva"])
        per_img = F.cross_entropy(val_logits, mon["yva"], reduction="none")
    seed_val_ce = per_img.mean().item()
    val_unc = per_img.std().item() / np.sqrt(N_MONITOR)
    mon["mask_threshold"] = seed_val_ce + TARGET_DLOSS + 2 * val_unc
    seed_test_ce, seed_test_acc = ce_acc(model0, xte, yte)
    probs0 = test_probs(model0, xte)
    seed_nll, seed_ece = nll_ece(probs0, yte)
    print(f"D = {D:,}  SCALE = {SCALE:.3g}  target dloss = {TARGET_DLOSS}")
    print(f"seed: mon_train_ce {seed_mon_ce:.4f}  val_ce {seed_val_ce:.4f}  "
          f"test_acc {seed_test_acc:.4f}  mask_threshold {mon['mask_threshold']:.4f}")
    del model0, probs0

    summary = {"config": {
        "seed": SEED, "device": DEVICE, "D": D, "batch": BATCH, "scale": SCALE,
        "target_dloss": TARGET_DLOSS, "refresh": REFRESH, "n_steps": N_STEPS,
        "burn_steps": BURN_STEPS, "dt_grid": DT_GRID, "pilot_steps": PILOT_STEPS,
        "eval_every": EVAL_EVERY, "ensemble_every": ENSEMBLE_EVERY,
        "hmc_pre": [HMC_PRE_STEPS, HMC_PRE_REFRESH],
        "mask_threshold": mon["mask_threshold"]},
        "seed_model": {"mon_train_ce": seed_mon_ce, "val_ce": seed_val_ce,
                       "test_ce": seed_test_ce, "test_acc": seed_test_acc,
                       "test_nll": seed_nll, "test_ece": seed_ece},
        "pilot": {}, "runs": {}}

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    results = {}
    for kind in ("rt", "hmc"):
        print(f"== pilot sweep [{kind}] ==")
        dt, rows = pilot(kind, state_cpu, xtr, ytr, mon, seed_mon_ce)
        summary["pilot"][kind] = {"chosen_dt": dt, "grid": rows}
        print(f"== main chain [{kind}]  dt = {dt:.1e} ==")
        tr = run_chain(kind, dt, N_STEPS, state_cpu, xtr, ytr, mon, collect=True)
        results[kind] = (dt, tr)
        torch.save(tr["model"].state_dict(),
                   os.path.join(CKPT, f"exp5_sample_{kind}_final.pt"))

        ev = np.array(tr["eval_steps"])
        post = ev >= BURN_STEPS
        probe = np.array(tr["probe_probs"], dtype=np.float32)  # (n_ev, P, 10)
        run = {"dt": dt, "seconds": tr["seconds"], "diverged": tr["diverged"],
               "it_per_s": len(tr["step_loss"]) / tr["seconds"]}
        if post.sum() > 20:
            X = probe[post].reshape(post.sum(), -1)[:, None, :]     # (n, 1, P*10)
            taus = metrics.tau_per_coordinate(X) * EVAL_EVERY        # in sampler steps
            run.update(
                post_train_mon_ce_med=float(np.median(np.array(tr["train_mon_ce"])[post])),
                post_val_ce_med=float(np.median(np.array(tr["val_ce"])[post])),
                post_val_acc_med=float(np.median(np.array(tr["val_acc"])[post])),
                tau_probe_med_steps=float(np.median(taus)),
                tau_probe_p90_steps=float(np.percentile(taus, 90)))
        ke = tr["ke"]
        run["ke_over_D_final"] = float(ke[ke > 0][-1] / D) if (ke > 0).any() else None
        run["ke_over_D_max"] = float(ke.max() / D)
        run["hmc_momentum_resets"] = int(np.sum((ke[:-1] > 10 * D) & (ke[1:] < ke[:-1] / 10)))
        run["ensemble"] = {"n_collected": len(tr["ens_steps"]), "n_used": tr["ens_n_used"],
                           "member_test_acc": tr["ens_test_acc"],
                           "member_val_ce": tr["ens_val_ce"]}
        if tr["ens_prob_mean"] is not None:
            enll, eece = nll_ece(tr["ens_prob_mean"], yte)
            run["ensemble"].update(
                test_acc=float((tr["ens_prob_mean"].argmax(1) == yte).float().mean()),
                test_nll=enll, test_ece=eece)
        summary["runs"][kind] = run

        np.savez_compressed(
            os.path.join(TAB, f"exp5_trace_{kind}.npz"),
            step_loss=tr["step_loss"], ke=ke, lnlum=tr["lnlum"],
            eval_steps=ev, train_mon_ce=np.array(tr["train_mon_ce"]),
            val_ce=np.array(tr["val_ce"]), val_acc=np.array(tr["val_acc"]),
            probe_probs=np.array(tr["probe_probs"]),
            ens_steps=np.array(tr["ens_steps"]), ens_val_ce=np.array(tr["ens_val_ce"]),
            ens_test_acc=np.array(tr["ens_test_acc"]))
        del tr["model"]

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for kind in ("rt", "hmc"):
        rows = summary["pilot"][kind]["grid"]
        ax.plot([r["dt"] for r in rows],
                [min(r["final_mon_ce"], 10) for r in rows], "o-",
                color=style.COLOR[kind], ls="--" if kind == "hmc" else "-",
                label=style.LABEL[kind])
    ax.axhline(seed_mon_ce + PILOT_FAIL_RISE, color=style.GREY, lw=1, label="fail threshold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dt"); ax.set_ylabel(f"train CE after {PILOT_STEPS} steps")
    ax.set_title("Pilot sweep: the hand-tuning that auto-tuning should replace")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp5_pilot.png")); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for kind in ("rt", "hmc"):
        _, tr = results[kind]
        ev = np.array(tr["eval_steps"])
        ls = "--" if kind == "hmc" else "-"
        axes[0].plot(ev, tr["train_mon_ce"], ls, color=style.COLOR[kind],
                     label=style.LABEL[kind])
        axes[1].plot(ev, tr["val_ce"], ls, color=style.COLOR[kind])
    for ax, name, seed_v in ((axes[0], "train monitor CE", seed_mon_ce),
                             (axes[1], "validation CE", seed_val_ce)):
        ax.axhline(seed_v, color=style.TRUTH, lw=1, label="trained point")
        ax.axhline(seed_v + TARGET_DLOSS, color=style.BLUE, lw=1, ls=":",
                   label="target (seed + 0.1)")
        ax.axvline(BURN_STEPS, color=style.GREY, lw=1, alpha=0.6)
        ax.set_xlabel("sampler step"); ax.set_ylabel("CE (nats/image)")
        ax.set_title(name)
    axes[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp5_loss_traces.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for kind in ("rt", "hmc"):
        _, tr = results[kind]
        ke = tr["ke"]; steps = np.arange(1, len(ke) + 1)
        ax.plot(steps[ke > 0], ke[ke > 0] / D, "--" if kind == "hmc" else "-",
                color=style.COLOR[kind], label=style.LABEL[kind], lw=1)
    ax.axhline(0.5, color=style.TRUTH, lw=1, label="equilibrium (KE/D = 1/2)")
    ax.axhline(100, color=style.GREY, lw=1, ls=":", label="vendor HMC reset level")
    ax.set_yscale("log"); ax.set_xlabel("sampler step"); ax.set_ylabel("kinetic energy / D")
    ax.set_title("Constant speed vs stochastic-gradient heating")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp5_kinetic.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for kind in ("rt", "hmc"):
        _, tr = results[kind]
        ev = np.array(tr["eval_steps"]); post = ev >= BURN_STEPS
        if post.sum() > 20:
            probe = np.array(tr["probe_probs"], dtype=np.float32)
            X = probe[post].reshape(post.sum(), -1)[:, None, :]
            taus = np.sort(metrics.tau_per_coordinate(X) * EVAL_EVERY)
            ax.plot(taus, np.linspace(0, 1, len(taus)),
                    "--" if kind == "hmc" else "-", color=style.COLOR[kind],
                    label=style.LABEL[kind])
    ax.set_xscale("log")
    ax.set_xlabel("autocorrelation time (sampler steps, lower bound)")
    ax.set_ylabel("CDF over probe class probabilities")
    ax.set_title(f"Function-space mixing ({N_PROBE} probe images x 10 classes)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp5_function_tau.png")); plt.close(fig)

    # ---- tables ----
    with open(os.path.join(TAB, "exp5_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    cols = ["kind", "dt", "it_per_s", "post_val_ce_med", "post_val_acc_med",
            "tau_probe_med_steps", "ke_over_D_final", "hmc_momentum_resets",
            "ens_n_used", "ens_test_acc", "ens_test_nll", "ens_test_ece"]
    with open(os.path.join(TAB, "exp5_summary.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for kind in ("rt", "hmc"):
            r = summary["runs"][kind]
            e = r["ensemble"]
            vals = [kind, r["dt"], r["it_per_s"], r.get("post_val_ce_med"),
                    r.get("post_val_acc_med"), r.get("tau_probe_med_steps"),
                    r["ke_over_D_final"], r["hmc_momentum_resets"], e["n_used"],
                    e.get("test_acc"), e.get("test_nll"), e.get("test_ece")]
            f.write(",".join("" if v is None else (v if isinstance(v, str) else f"{v:.6g}")
                             for v in vals) + "\n")
    print("...done. Traces, figures + tables written.")


if __name__ == "__main__":
    main()
