"""exp7 MAP baseline: does a single Adam-optimized network with the same
sigma(x) head reach the chain's calibration?

The question this answers: is the per-star calibration result a property
of SAMPLING or of the LIKELIHOOD? If one optimized network calibrates the
same, the figure is a noise-model result, not a sampler result.

The target is imported from exp7_gaia_hetero (batch_nll_mean,
exact_decomposition, sigmas_for), never re-derived: the loss is the same
negative log posterior the chain samples,

    N * mean_batch_nll + 0.5 * sum_j (theta_j / sigma_prior_j)^2

with the Glorot per-layer priors as an explicit sum (weight_decay cannot
express per-layer sigmas).

Two-stage schedule mirroring the chain's warm start (joint mu/sigma
training from scratch collapses sigma on well-fit points):
  stage 1: sigma head frozen at the constant sigma = 0.045
           (row zeroed, bias = 3 + ln 0.045), train mu only;
  stage 2: unfreeze, train both.
Adam, batch 1024, ~200 epochs per stage, 3 learning rates, keep the best
final exact training loss.

Report on the 25,232 test stars with tau = 0 (a point estimate has no
member spread): z = (y - mu) / sqrt(err^2 + sigma(x)^2).

    python exp7_gaia_map.py            # sweep, report, save exp7_map.npz
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp7_gaia_hetero as H

TAB = H.TAB
DEV = H.DEV
BATCH = 1024
EPOCH_STEPS = 100_924 // BATCH + 1          # 99 steps ~ one epoch
STAGE_EPOCHS = 200
SIGMA_START = 0.045
BINS = ((0.0, 0.004), (0.004, 0.006), (0.006, 0.01), (0.01, 0.02))


def init_model():
    model = H.make_model()
    with torch.no_grad():
        W3 = model.net[4].weight            # (2, 64): row 0 mu, row 1 sigma
        b3 = model.net[4].bias
        W3[1].zero_()
        b3[1] = 3.0 + math.log(SIGMA_START) # exp(-3 + b) = SIGMA_START
    return model


def train_one(lr, Xs, y, yerr):
    n_train = len(Xs)
    model = init_model()
    sigmas = H.sigmas_for(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stream = torch.Generator().manual_seed(2003)

    def step(freeze_sigma):
        batch = torch.randint(0, n_train, (BATCH,), generator=stream).to(DEV)
        opt.zero_grad()
        loss = n_train * H.batch_nll_mean(model, Xs, y, yerr, batch)
        prior = sum(((p / sg) ** 2).sum()
                    for p, sg in zip(model.parameters(), sigmas))
        (loss + 0.5 * prior).backward()
        if freeze_sigma:
            model.net[4].weight.grad[1].zero_()
            model.net[4].bias.grad[1].zero_()
        opt.step()

    for _ in range(STAGE_EPOCHS * EPOCH_STEPS):
        step(freeze_sigma=True)
    for _ in range(STAGE_EPOCHS * EPOCH_STEPS):
        step(freeze_sigma=False)

    misfit, wnorm = H.exact_decomposition(model, sigmas, Xs, y, yerr)
    return model, misfit + 0.5 * wnorm


@torch.no_grad()
def metrics(model, tXs, ty, tyerr):
    mu, r = model.mu_r(tXs)
    sig = torch.exp(2.0 * (H.LNS0 + r)).sqrt()
    z = ((ty - mu) / (tyerr ** 2 + sig ** 2).sqrt()).cpu().numpy()
    mu_np, sig_np = mu.cpu().numpy(), sig.cpu().numpy()
    te = tyerr.cpu().numpy()
    kurt = float(((z - z.mean()) ** 4).mean() / z.std() ** 4)
    return {
        "mu": mu_np, "sig": sig_np, "z": z,
        "rmse": float(np.sqrt(((mu_np - ty.cpu().numpy()) ** 2).mean())),
        "z_std": float(z.std()),
        "bins": [float(z[(te >= lo) & (te < hi)].std()) for lo, hi in BINS],
        "kurt": kurt,
        "n_tail": int((np.abs(z) > 4).sum()),
        "sig_med": float(np.median(sig_np)),
    }


if __name__ == "__main__":
    Xs, y, yerr, tXs, ty, tyerr = H.load_data()
    print(f"device {DEV}, MAP with the hetero target, batch {BATCH}, "
          f"{STAGE_EPOCHS} epochs x 2 stages", flush=True)
    best = None
    for lr in (3e-3, 1e-3, 3e-4):
        model, loss = train_one(lr, Xs, y, yerr)
        m = metrics(model, tXs, ty, tyerr)
        print(f"[lr={lr:g}] exact train loss {loss:,.0f}  "
              f"RMSE {m['rmse']:.4f}  SD(z) {m['z_std']:.3f}  "
              f"sigma med {m['sig_med']:.4f}", flush=True)
        if best is None or loss < best[1]:
            best = (lr, loss, m)

    lr, loss, m = best
    print(f"\nbest lr {lr:g} (train loss {loss:,.0f}); test metrics, tau=0:")
    print(f"  RMSE          {m['rmse']:.4f}   (chain 0.0494)")
    print(f"  SD(z)         {m['z_std']:.3f}    (chain 1.08, mixture-variance "
          f"convention)")
    print(f"  SD(z) by bin  {' / '.join(f'{b:.2f}' for b in m['bins'])}"
          f"   (chain 0.97 / 1.07 / 1.07 / 1.07)")
    print(f"  kurtosis      {m['kurt']:.1f}      (chain 8.1)")
    print(f"  |z| > 4       {m['n_tail']}      (chain 108)")
    print(f"  sigma med     {m['sig_med']:.4f}   (chain 0.033)")
    np.savez(TAB / "exp7_map.npz", mu=m["mu"], sig=m["sig"], z=m["z"],
             lr=lr, train_loss=loss, rmse=m["rmse"], z_std=m["z_std"],
             bins=np.array(m["bins"]), kurt=m["kurt"], n_tail=m["n_tail"],
             sig_med=m["sig_med"], batch=BATCH, stage_epochs=STAGE_EPOCHS,
             sigma_start=SIGMA_START)
    print("saved exp7_map.npz", flush=True)
