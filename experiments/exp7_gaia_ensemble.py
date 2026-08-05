"""exp7 deep ensemble baseline: 10 MAP fits, predictive by total variance.

Task 2 of the sampler-vs-likelihood question. Same architecture, same
objective (imported hetero target + explicit Glorot prior), same
two-stage schedule and learning rate as the Task 1 winner; only the init
and the batch stream vary by seed. Predictive per star by the law of
total variance:

    var_i = err_i^2 + mean_m[sigma_m(x_i)^2] + var_m[mu_m(x_i)]

exp6 note: the ensemble beat the chain on NLL there, so no assumed loser.

    python exp7_gaia_ensemble.py       # ~10 fits, report, exp7_ensemble.npz
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp7_gaia_hetero as H
from exp7_gaia_map import BATCH, BINS, EPOCH_STEPS, SIGMA_START, STAGE_EPOCHS

TAB = H.TAB
DEV = H.DEV
LR = 1e-3                 # Task 1's best-training-loss learning rate
N_SEEDS = 10


def init_model(seed):
    model = H.make_model()
    torch.manual_seed(seed)
    with torch.no_grad():
        for layer in model.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        W3, b3 = model.net[4].weight, model.net[4].bias
        W3[1].zero_()
        b3[1] = 3.0 + math.log(SIGMA_START)
    return model.to(DEV)


def train_one(seed, Xs, y, yerr):
    n_train = len(Xs)
    model = init_model(seed)
    sigmas = H.sigmas_for(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    stream = torch.Generator().manual_seed(2003 + seed)

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


if __name__ == "__main__":
    Xs, y, yerr, tXs, ty, tyerr = H.load_data()
    print(f"device {DEV}, ensemble of {N_SEEDS}, lr {LR:g}, batch {BATCH}, "
          f"{STAGE_EPOCHS} epochs x 2 stages", flush=True)
    mus, sig2s = [], []
    for seed in range(N_SEEDS):
        model, loss = train_one(seed, Xs, y, yerr)
        with torch.no_grad():
            mu, r = model.mu_r(tXs)
            mus.append(mu.cpu().numpy())
            sig2s.append(np.exp(2.0 * (H.LNS0 + r.cpu().numpy())))
        print(f"[seed {seed}] exact train loss {loss:,.0f}", flush=True)

    mus, sig2s = np.array(mus), np.array(sig2s)
    y_np, te = ty.cpu().numpy(), tyerr.cpu().numpy()
    mu_hat = mus.mean(0)
    var_tot = te ** 2 + sig2s.mean(0) + mus.var(0)
    z = (y_np - mu_hat) / np.sqrt(var_tot)
    kurt = float(((z - z.mean()) ** 4).mean() / z.std() ** 4)
    bins = [float(z[(te >= lo) & (te < hi)].std()) for lo, hi in BINS]
    rmse = float(np.sqrt(((mu_hat - y_np) ** 2).mean()))
    sig_med = float(np.median(np.sqrt(sig2s.mean(0))))
    tau_med = float(np.median(mus.std(0)))
    n_tail = int((np.abs(z) > 4).sum())

    print(f"\nensemble of {N_SEEDS}; test metrics, total-variance predictive:")
    print(f"  RMSE          {rmse:.4f}   (MAP 0.0489, chain 0.0494)")
    print(f"  SD(z)         {z.std():.3f}    (MAP 1.180, chain 1.08)")
    print(f"  SD(z) by bin  {' / '.join(f'{b:.2f}' for b in bins)}"
          f"   (MAP 1.26/1.24/1.13/1.04, chain 0.97/1.07/1.07/1.07)")
    print(f"  kurtosis      {kurt:.1f}      (MAP 15.9, chain 8.1)")
    print(f"  |z| > 4       {n_tail}      (MAP 209, chain 108)")
    print(f"  sigma med     {sig_med:.4f}   (MAP 0.0354, chain 0.033)")
    print(f"  member-spread tau median {tau_med:.4f}  (chain 0.0049)")
    np.savez(TAB / "exp7_ensemble.npz", mus=mus, sig2s=sig2s,
             rmse=rmse, z_std=float(z.std()), bins=np.array(bins),
             kurt=kurt, n_tail=n_tail, sig_med=sig_med, tau_med=tau_med,
             lr=LR, n_seeds=N_SEEDS, batch=BATCH,
             stage_epochs=STAGE_EPOCHS, sigma_start=SIGMA_START)
    print("saved exp7_ensemble.npz", flush=True)
