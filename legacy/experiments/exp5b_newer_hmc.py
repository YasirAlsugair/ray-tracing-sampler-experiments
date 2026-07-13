"""Exp 5b: newer stochastic-gradient MCMC baselines on the ResNet-50 MNIST posterior.

Extends exp5_resnet_mnist_sample.py (which runs ray tracing + the vendor's stochastic
HMC) with the modern samplers that are valid under minibatch gradients, so Josh can see
RTS against the current field on the identical problem:

  sgld    Welling & Teh 2011. Langevin, no momentum: the classic SG-MCMC reference.
  sghmc   Chen, Fox & Guestrin 2014. HMC with friction: the standard fix for
          stochastic-gradient heating. Friction ALPHA = 0.1, temperature 1,
          stationary per-coordinate momentum variance = eta.
  mclmc   Microcanonical Langevin (Ver Steeg & Galstyan 2021 ESH dynamics; Robnik &
          Seljak 2022+). Unit-speed velocity with the exact hyperbolic bend toward
          the gradient: RTS's closest modern cousin (the paper's W framework contains
          microcanonical HMC as the W = L^{1/D} special case). Partial refresh matched
          to the RT run's per-step momentum decay.

NUTS is deliberately absent: it needs exact full-data gradients plus a Metropolis
test, which is invalid under minibatching and costs hours per iteration at D = 23.5M.
The comparison of interest here is the stochastic-gradient regime.

Everything else mirrors exp5b's parent run: same checkpoint, same likelihood
(ln L = -SCALE * mean CE, flat prior, eval-mode BatchNorm), same minibatch stream
(same generator seed), same pilot rule (largest step that does not fail in 150 steps),
same monitors, GPT-2-style validation masking, and posterior ensembles. All chains are
unadjusted (no Metropolis), like the parent run and the paper's NN recipe.

Outputs:
  results/tables/exp5_trace_{sgld,sghmc,mclmc}.npz
  results/tables/exp5b_summary.json
  results/tables/exp5_all_summary.csv            (merged with exp5 runs if present)
  results/figures/exp5_all_loss.png, exp5_all_tau.png, exp5_all_kinetic.png
  results/checkpoints/exp5_sample_{sgld,sghmc,mclmc}_final.pt
"""
import sys, os, json, time, math

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn.functional as F

import exp5_resnet_mnist_sample as base
from rts import metrics
from plots import style
import matplotlib.pyplot as plt

SEED = base.SEED
DEVICE = base.DEVICE
BATCH = base.BATCH
N_TRAIN = base.N_TRAIN
SCALE = base.SCALE
TARGET_DLOSS = base.TARGET_DLOSS
N_STEPS = base.N_STEPS
EVAL_EVERY = base.EVAL_EVERY
BURN_STEPS = base.BURN_STEPS
ENSEMBLE_EVERY = base.ENSEMBLE_EVERY
N_MONITOR = base.N_MONITOR
N_PROBE = base.N_PROBE
PILOT_STEPS = base.PILOT_STEPS
PILOT_FAIL_RISE = base.PILOT_FAIL_RISE
CKPT, FIG, TAB = base.CKPT, base.FIG, base.TAB

ALPHA = 0.1                      # SGHMC friction per step
MCLMC_DECAY_PER_STEP = 5e-5      # matched to the RT run: refresh_rate 5 x dt 1e-5

STEP_GRIDS = {
    "sgld": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    "sghmc": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    "mclmc": [0.01, 0.03, 0.1, 0.3, 1.0],   # arc length per step; RT moves ~0.05
}
COLOR = {"rt": style.RT, "hmc": style.HMC, "sgld": style.GREY,
         "sghmc": "#34C6A8", "mclmc": style.BLUE}
LABEL = {"rt": "Ray tracing", "hmc": "HMC (vendor)", "sgld": "SGLD",
         "sghmc": "SGHMC (friction)", "mclmc": "MCLMC (microcanonical)"}
LSTYLE = {"rt": "-", "hmc": "--", "sgld": ":", "sghmc": "-.", "mclmc": "-"}


def global_norm(tensors):
    return math.sqrt(sum(float(t.pow(2).sum()) for t in tensors))


def make_state(kind, params, eta):
    if kind == "sghmc":
        return [torch.randn_like(p) * math.sqrt(eta) for p in params]
    if kind == "mclmc":
        u = [torch.randn_like(p) for p in params]
        un = global_norm(u)
        for t in u:
            t.div_(un)
        return u
    return None


@torch.no_grad()
def sampler_step(kind, params, state, eta, D):
    """One update from current p.grad (grad of mean CE). Returns kinetic energy."""
    if kind == "sgld":
        for p in params:
            p.add_(p.grad, alpha=-0.5 * eta * SCALE)
            p.add_(torch.randn_like(p), alpha=math.sqrt(eta))
        return 0.0
    if kind == "sghmc":
        noise = math.sqrt(2 * ALPHA * eta)
        for p, v in zip(params, state):
            v.mul_(1 - ALPHA).add_(p.grad, alpha=-eta * SCALE)
            v.add_(torch.randn_like(p), alpha=noise)
            p.add_(v)
        return sum(float(v.pow(2).sum()) for v in state) / (2 * eta)
    if kind == "mclmc":
        u = state
        gnorm = global_norm([p.grad for p in params])
        if gnorm > 0 and np.isfinite(gnorm):
            delta = eta * SCALE * gnorm / (D - 1)
            eu = -sum(float((p.grad * up).sum()) for p, up in zip(params, u)) / gnorm
            sh, ch = math.sinh(delta), math.cosh(delta)
            coef, denom = sh + eu * (ch - 1), ch + eu * sh
            for p, up in zip(params, u):
                up.add_(p.grad, alpha=-coef / gnorm).div_(denom)
        c = math.exp(-MCLMC_DECAY_PER_STEP)
        z = [torch.randn_like(up) for up in u]
        zn = global_norm(z)
        for up, zp in zip(u, z):
            up.mul_(c).add_(zp, alpha=math.sqrt(1 - c * c) / zn)
        un = global_norm(u)
        for up, zp in zip(u, z):
            up.div_(un)
        for p, up in zip(params, u):
            p.add_(up, alpha=eta)
        return 0.5 * D * un * un
    raise ValueError(kind)


def run_chain(kind, eta, n_steps, state_cpu, xtr, ytr, mon, collect=True):
    torch.manual_seed(SEED + {"sgld": 11, "sghmc": 12, "mclmc": 13}[kind])
    g = torch.Generator().manual_seed(SEED + 99)      # same batch stream as exp5
    model = base.build_model(state_cpu)
    params = [p for p in model.parameters()]
    D = mon["D"]
    st = make_state(kind, params, eta)

    tr = {"step_loss": np.zeros(n_steps), "ke": np.zeros(n_steps),
          "lnlum": np.zeros(n_steps), "eval_steps": [], "train_mon_ce": [],
          "val_ce": [], "val_acc": [], "probe_probs": [],
          "ens_steps": [], "ens_val_ce": [], "ens_test_acc": [], "diverged": False}
    ens_prob_sum, ens_n = None, 0
    t0 = time.time()

    for step in range(n_steps):
        idx = torch.randint(0, N_TRAIN, (BATCH,), generator=g)
        for p in params:
            p.grad = None
        loss = F.cross_entropy(base.forward(model, xtr[idx.to(DEVICE)]),
                               ytr[idx.to(DEVICE)])
        loss.backward()
        ke = sampler_step(kind, params, st, eta, D)
        lval = loss.item()
        tr["step_loss"][step] = lval
        tr["ke"][step] = ke
        if not np.isfinite(lval):
            print(f"  [{kind}] diverged (non-finite loss) at step {step}")
            tr["diverged"] = True
            break

        if collect and (step + 1) % EVAL_EVERY == 0:
            with torch.no_grad():
                mon_logits = base.forward(model, mon["xtr"])
                val_logits = base.forward(model, mon["xva"])
            tr["eval_steps"].append(step + 1)
            tr["train_mon_ce"].append(F.cross_entropy(mon_logits, mon["ytr"]).item())
            tr["val_ce"].append(F.cross_entropy(val_logits, mon["yva"]).item())
            tr["val_acc"].append((val_logits.argmax(1) == mon["yva"]).float().mean().item())
            probe = F.softmax(val_logits[:N_PROBE], dim=1)
            tr["probe_probs"].append(probe.cpu().numpy().astype(np.float16))

        if collect and step + 1 >= BURN_STEPS and (step + 1) % ENSEMBLE_EVERY == 0:
            probs = base.test_probs(model, mon["xte"])
            acc = (probs.argmax(1) == mon["yte"]).float().mean().item()
            with torch.no_grad():
                vce = F.cross_entropy(base.forward(model, mon["xva"]), mon["yva"]).item()
            tr["ens_steps"].append(step + 1)
            tr["ens_val_ce"].append(vce)
            tr["ens_test_acc"].append(acc)
            if vce <= mon["mask_threshold"]:
                ens_prob_sum = probs if ens_prob_sum is None else ens_prob_sum + probs
                ens_n += 1
        if collect and (step + 1) % 100 == 0:
            print(f"  [{kind}] step {step+1}/{n_steps}  batch_ce {lval:.4f}  "
                  f"KE/D {ke/D:.3f}  ({(step+1)/(time.time()-t0):.2f} it/s)", flush=True)

    tr["seconds"] = time.time() - t0
    tr["ens_prob_mean"] = None if ens_n == 0 else (ens_prob_sum / ens_n)
    tr["ens_n_used"] = ens_n
    tr["model"] = model
    return tr


def pilot(kind, state_cpu, xtr, ytr, mon, seed_mon_ce):
    rows = []
    for eta in STEP_GRIDS[kind]:
        t = run_chain(kind, eta, PILOT_STEPS, state_cpu, xtr, ytr, mon, collect=False)
        with torch.no_grad():
            ce = F.cross_entropy(base.forward(t["model"], mon["xtr"]), mon["ytr"]).item()
        ok = np.isfinite(ce) and not t["diverged"] and ce < seed_mon_ce + PILOT_FAIL_RISE
        rows.append({"eta": eta, "final_mon_ce": ce if np.isfinite(ce) else float("inf"),
                     "ok": bool(ok)})
        print(f"  pilot [{kind}] eta {eta:.1e}  mon_ce {ce:.4f}  {'ok' if ok else 'FAIL'}")
        del t
    ok_etas = [r["eta"] for r in rows if r["ok"]]
    chosen = max(ok_etas) if ok_etas else min(STEP_GRIDS[kind])
    return chosen, rows


def main():
    xtr, ytr = base.load_split(train=True)
    xte, yte = base.load_split(train=False)
    state_cpu = torch.load(os.path.join(CKPT, "exp5_resnet50_mnist.pt"),
                           map_location="cpu", weights_only=True)

    model0 = base.build_model(state_cpu)
    D = sum(p.numel() for p in model0.parameters())
    torch.manual_seed(SEED + 5)
    mon_tr_idx = torch.randperm(N_TRAIN)[:N_MONITOR].to(DEVICE)
    mon = {"xtr": xtr[mon_tr_idx], "ytr": ytr[mon_tr_idx],
           "xva": xte[:N_MONITOR], "yva": yte[:N_MONITOR],
           "xte": xte, "yte": yte, "D": D}
    with torch.no_grad():
        seed_mon_ce = F.cross_entropy(base.forward(model0, mon["xtr"]), mon["ytr"]).item()
        per_img = F.cross_entropy(base.forward(model0, mon["xva"]), mon["yva"],
                                  reduction="none")
    seed_val_ce = per_img.mean().item()
    mon["mask_threshold"] = (seed_val_ce + TARGET_DLOSS
                             + 2 * per_img.std().item() / np.sqrt(N_MONITOR))
    print(f"D = {D:,}  SCALE = {SCALE:.3g}  mask_threshold {mon['mask_threshold']:.4f}")
    del model0

    summary = {"config": {"alpha_sghmc": ALPHA, "mclmc_decay_per_step": MCLMC_DECAY_PER_STEP,
                          "step_grids": STEP_GRIDS, "inherits": "exp5_summary.json"},
               "pilot": {}, "runs": {}}
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)

    for kind in ("sgld", "sghmc", "mclmc"):
        print(f"== pilot sweep [{kind}] ==")
        eta, rows = pilot(kind, state_cpu, xtr, ytr, mon, seed_mon_ce)
        summary["pilot"][kind] = {"chosen_eta": eta, "grid": rows}
        print(f"== main chain [{kind}]  eta = {eta:.1e} ==")
        tr = run_chain(kind, eta, N_STEPS, state_cpu, xtr, ytr, mon, collect=True)
        torch.save(tr["model"].state_dict(),
                   os.path.join(CKPT, f"exp5_sample_{kind}_final.pt"))
        del tr["model"]

        ev = np.array(tr["eval_steps"])
        post = ev >= BURN_STEPS
        run = {"eta": eta, "seconds": tr["seconds"], "diverged": tr["diverged"],
               "it_per_s": len(tr["step_loss"]) / tr["seconds"]}
        if post.sum() > 20:
            probe = np.array(tr["probe_probs"], dtype=np.float32)
            X = probe[post].reshape(post.sum(), -1)[:, None, :]
            taus = metrics.tau_per_coordinate(X) * EVAL_EVERY
            run.update(
                post_train_mon_ce_med=float(np.median(np.array(tr["train_mon_ce"])[post])),
                post_val_ce_med=float(np.median(np.array(tr["val_ce"])[post])),
                post_val_acc_med=float(np.median(np.array(tr["val_acc"])[post])),
                tau_probe_med_steps=float(np.median(taus)),
                tau_probe_p90_steps=float(np.percentile(taus, 90)))
        ke = tr["ke"]
        run["ke_over_D_final"] = float(ke[ke > 0][-1] / D) if (ke > 0).any() else None
        run["ke_over_D_max"] = float(ke.max() / D) if (ke > 0).any() else None
        run["ensemble"] = {"n_collected": len(tr["ens_steps"]), "n_used": tr["ens_n_used"],
                           "member_test_acc": tr["ens_test_acc"],
                           "member_val_ce": tr["ens_val_ce"]}
        if tr["ens_prob_mean"] is not None:
            enll, eece = base.nll_ece(tr["ens_prob_mean"], yte)
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

    with open(os.path.join(TAB, "exp5b_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- combined figures over every sampler with a stored trace ----
    kinds = [k for k in ("rt", "hmc", "sgld", "sghmc", "mclmc")
             if os.path.exists(os.path.join(TAB, f"exp5_trace_{k}.npz"))]
    traces = {k: np.load(os.path.join(TAB, f"exp5_trace_{k}.npz")) for k in kinds}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for k in kinds:
        t = traces[k]
        axes[0].plot(t["eval_steps"], t["train_mon_ce"], LSTYLE[k], color=COLOR[k],
                     label=LABEL[k], lw=1.2)
        axes[1].plot(t["eval_steps"], t["val_ce"], LSTYLE[k], color=COLOR[k], lw=1.2)
    for ax, name in ((axes[0], "train monitor CE"), (axes[1], "validation CE")):
        ax.axhline(seed_val_ce + TARGET_DLOSS if name.startswith("val") else
                   seed_mon_ce + TARGET_DLOSS, color=style.TRUTH, lw=1, ls=":",
                   label="target (seed + 0.1)")
        ax.axvline(BURN_STEPS, color=style.GREY, lw=1, alpha=0.6)
        ax.set_xlabel("sampler step"); ax.set_ylabel("CE (nats/image)")
        ax.set_title(name)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp5_all_loss.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k in kinds:
        t = traces[k]
        ev = t["eval_steps"]; post = ev >= BURN_STEPS
        if post.sum() > 20:
            X = t["probe_probs"][post].astype(np.float32).reshape(post.sum(), -1)[:, None, :]
            taus = np.sort(metrics.tau_per_coordinate(X) * EVAL_EVERY)
            ax.plot(taus, np.linspace(0, 1, len(taus)), LSTYLE[k], color=COLOR[k],
                    label=LABEL[k], lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("autocorrelation time (sampler steps, lower bound)")
    ax.set_ylabel("CDF over probe class probabilities")
    ax.set_title("Function-space mixing, all samplers")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp5_all_tau.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k in kinds:
        t = traces[k]
        ke = t["ke"]; steps = np.arange(1, len(ke) + 1)
        if (ke > 0).any():
            ax.plot(steps[ke > 0], ke[ke > 0] / D, LSTYLE[k], color=COLOR[k],
                    label=LABEL[k], lw=1)
    ax.axhline(0.5, color=style.TRUTH, lw=1, label="equilibrium (KE/D = 1/2)")
    ax.set_yscale("log"); ax.set_xlabel("sampler step")
    ax.set_ylabel("kinetic energy / D (native units)")
    ax.set_title("Constant speed vs stochastic-gradient heating, all samplers")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp5_all_kinetic.png")); plt.close(fig)

    # ---- merged summary CSV ----
    merged = {}
    exp5_json = os.path.join(TAB, "exp5_summary.json")
    if os.path.exists(exp5_json):
        with open(exp5_json) as f:
            merged.update(json.load(f)["runs"])
    merged.update(summary["runs"])
    cols = ["kind", "step_param", "it_per_s", "post_val_ce_med", "post_val_acc_med",
            "tau_probe_med_steps", "ke_over_D_final", "ens_n_used", "ens_test_acc",
            "ens_test_nll", "ens_test_ece"]
    with open(os.path.join(TAB, "exp5_all_summary.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for k in ("rt", "hmc", "sgld", "sghmc", "mclmc"):
            if k not in merged:
                continue
            r = merged[k]
            e = r.get("ensemble", {})
            vals = [k, r.get("dt", r.get("eta")), r.get("it_per_s"),
                    r.get("post_val_ce_med"), r.get("post_val_acc_med"),
                    r.get("tau_probe_med_steps"), r.get("ke_over_D_final"),
                    e.get("n_used"), e.get("test_acc"), e.get("test_nll"),
                    e.get("test_ece")]
            f.write(",".join("" if v is None else (v if isinstance(v, str) else f"{v:.6g}")
                             for v in vals) + "\n")
    print("...done. Baseline traces, combined figures + merged table written.")


if __name__ == "__main__":
    main()
