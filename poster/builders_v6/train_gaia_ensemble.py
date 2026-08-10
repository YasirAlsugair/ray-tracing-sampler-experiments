#!/usr/bin/env python3
"""Extend the exact Gaia deep-ensemble baseline from 10 to 50 models.

Seeds 0--9 are the preserved upstream fits.  This script trains seeds 10--49
with the upstream recipe verbatim: 110-64-64-2 tanh network, explicit Glorot
prior, Adam at 1e-3, batch 1024, 200 epochs with the scatter head frozen at
0.045 followed by 200 joint epochs.  Only initialization and batch stream vary.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
POSTER = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
DATA = TABLES / "exp7_gaia_pristine.npz"
BASE_PREDICTIONS = TABLES / "exp7_ensemble.npz"
BASE_WEIGHTS = TABLES / "exp7_ensemble_weights.npz"
OUT = POSTER / "artifacts" / "data_v6" / "exp7_ensemble_50.npz"

LNS0 = -3.0
SIGMA_START = 0.045
LR = 1e-3
LOG2PI = math.log(2.0 * math.pi)


class BatchedHeteroEnsemble(nn.Module):
    """Independent MLPs stored along a leading member dimension."""

    def __init__(self, seeds: list[int]):
        super().__init__()
        members = len(seeds)
        self.members = members
        self.w1 = nn.Parameter(torch.empty(members, 64, 110))
        self.b1 = nn.Parameter(torch.empty(members, 64))
        self.w2 = nn.Parameter(torch.empty(members, 64, 64))
        self.b2 = nn.Parameter(torch.empty(members, 64))
        self.w3 = nn.Parameter(torch.empty(members, 2, 64))
        self.b3 = nn.Parameter(torch.empty(members, 2))

        # Match torch.nn.Linear.reset_parameters, with one independent global
        # stream per upstream seed and the original layer/reset ordering.
        with torch.no_grad():
            for member, seed in enumerate(seeds):
                generator = torch.Generator().manual_seed(seed)
                for weight, bias, fan_in in (
                    (self.w1[member], self.b1[member], 110),
                    (self.w2[member], self.b2[member], 64),
                    (self.w3[member], self.b3[member], 64),
                ):
                    nn.init.kaiming_uniform_(weight, a=math.sqrt(5),
                                             generator=generator)
                    bound = 1.0 / math.sqrt(fan_in)
                    nn.init.uniform_(bias, -bound, bound, generator=generator)
                self.w3[member, 1].zero_()
                self.b3[member, 1] = 3.0 + math.log(SIGMA_START)

    def member_batches(self, x: torch.Tensor):
        h = torch.tanh(torch.bmm(x, self.w1.transpose(1, 2)) + self.b1[:, None])
        h = torch.tanh(torch.bmm(h, self.w2.transpose(1, 2)) + self.b2[:, None])
        out = torch.bmm(h, self.w3.transpose(1, 2)) + self.b3[:, None]
        return out[..., 0], out[..., 1]

    def shared_batch(self, x: torch.Tensor):
        h = torch.tanh(torch.einsum("bi,moi->mbo", x, self.w1) + self.b1[:, None])
        h = torch.tanh(torch.einsum("mbi,moi->mbo", h, self.w2) + self.b2[:, None])
        out = torch.einsum("mbi,moi->mbo", h, self.w3) + self.b3[:, None]
        return out[..., 0], out[..., 1]

    def prior_by_member(self):
        sigmas = (
            math.sqrt(2.0 / (110 + 64)), 0.1,
            math.sqrt(2.0 / (64 + 64)), 0.1,
            math.sqrt(2.0 / (64 + 2)), 0.1,
        )
        return sum((parameter / sigma).reshape(self.members, -1).square().sum(1)
                   for parameter, sigma in zip(self.parameters(), sigmas))

    def flat_states(self):
        chunks = [p.detach().reshape(self.members, -1) for p in self.parameters()]
        return torch.cat(chunks, dim=1).cpu().numpy().astype(np.float32)


def nll(mu, r, y, yerr):
    var = yerr.square() + torch.exp(2.0 * (LNS0 + r))
    return ((y - mu).square() / (2.0 * var)
            + 0.5 * torch.log(var) + 0.5 * LOG2PI)


@torch.no_grad()
def predict(model, x, chunk=4096):
    mus, sig2s = [], []
    for start in range(0, len(x), chunk):
        mu, r = model.shared_batch(x[start:start + chunk])
        mus.append(mu.cpu())
        sig2s.append(torch.exp(2.0 * (LNS0 + r)).cpu())
    return (torch.cat(mus, 1).numpy().astype(np.float32),
            torch.cat(sig2s, 1).numpy().astype(np.float32))


@torch.no_grad()
def exact_train_loss(model, x, y, yerr, chunk=4096):
    values = torch.zeros(model.members)
    for start in range(0, len(x), chunk):
        mu, r = model.shared_batch(x[start:start + chunk])
        values += nll(mu, r, y[start:start + chunk][None, :],
                      yerr[start:start + chunk][None, :]).sum(1)
    return (values + 0.5 * model.prior_by_member()).cpu().numpy()


def ensemble_metrics(mus, sig2s, y, yerr):
    mu = mus.mean(0)
    var = yerr**2 + sig2s.mean(0) + mus.var(0)
    z = (y - mu) / np.sqrt(var)
    rmse = float(np.sqrt(np.mean((mu - y)**2)))
    member_var = yerr[None, :]**2 + sig2s
    lp = (-0.5 * (y[None, :] - mus)**2 / member_var
          - 0.5 * np.log(member_var) - 0.5 * LOG2PI)
    mixture_nll = float(-(np.logaddexp.reduce(lp, axis=0)
                          - np.log(len(lp))).mean())
    return rmse, float(z.std()), mixture_nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-seed", type=int, default=10)
    parser.add_argument("--new-members", type=int, default=40)
    parser.add_argument("--stage-epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    if not BASE_PREDICTIONS.exists() or not BASE_WEIGHTS.exists():
        raise FileNotFoundError("preserved seed-0--9 baseline artifacts are missing")
    torch.set_num_threads(min(12, os.cpu_count() or 1))
    data = np.load(DATA)
    x = torch.from_numpy(data["train_Xs"].astype(np.float32, copy=False))
    y = torch.from_numpy(data["train_y"].astype(np.float32, copy=False))
    yerr = torch.from_numpy(data["train_yerr"].astype(np.float32, copy=False))
    test_x = torch.from_numpy(data["test_Xs"].astype(np.float32, copy=False))
    test_y = data["test_y"].astype(np.float32, copy=False)
    test_yerr = data["test_yerr"].astype(np.float32, copy=False)

    seeds = list(range(args.first_seed, args.first_seed + args.new_members))
    model = BatchedHeteroEnsemble(seeds)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    streams = [torch.Generator().manual_seed(2003 + seed) for seed in seeds]
    steps_per_epoch = len(x) // args.batch + 1
    t0 = time.time()
    print(f"device cpu, extending 10 preserved fits with seeds {seeds[0]}--{seeds[-1]}; "
          f"batch {args.batch}, {args.stage_epochs} epochs x 2 stages", flush=True)

    for stage, freeze_sigma in ((1, True), (2, False)):
        for epoch in range(1, args.stage_epochs + 1):
            running = 0.0
            for _ in range(steps_per_epoch):
                index = torch.stack([
                    torch.randint(len(x), (args.batch,), generator=stream)
                    for stream in streams
                ])
                xb, yb, eb = x[index], y[index], yerr[index]
                mu, r = model.member_batches(xb)
                member_loss = len(x) * nll(mu, r, yb, eb).mean(1)
                member_loss = member_loss + 0.5 * model.prior_by_member()
                optimizer.zero_grad(set_to_none=True)
                member_loss.sum().backward()
                if freeze_sigma:
                    model.w3.grad[:, 1].zero_()
                    model.b3.grad[:, 1].zero_()
                optimizer.step()
                running += float(member_loss.mean().detach())
            if epoch == 1 or epoch % 10 == 0 or epoch == args.stage_epochs:
                elapsed = (time.time() - t0) / 60.0
                print(f"stage {stage}, epoch {epoch:03d}/{args.stage_epochs}: "
                      f"batch objective {running / steps_per_epoch:,.0f}, "
                      f"{elapsed:.1f} min", flush=True)

    new_mus, new_sig2s = predict(model, test_x)
    new_states = model.flat_states()
    train_losses = exact_train_loss(model, x, y, yerr)
    base = np.load(BASE_PREDICTIONS)
    base_weights = np.load(BASE_WEIGHTS)["states"]
    mus = np.concatenate([base["mus"], new_mus])
    sig2s = np.concatenate([base["sig2s"], new_sig2s])
    states = np.concatenate([base_weights, new_states])
    if len(mus) != 50 or len(states) != 50:
        raise RuntimeError(f"expected 50 final members, got {len(mus)} predictions "
                           f"and {len(states)} states")

    rmse, zstd, mixture_nll = ensemble_metrics(mus, sig2s, test_y, test_yerr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, mus=mus, sig2s=sig2s, states=states,
        test_y=test_y, test_yerr=test_yerr, train_loss_new=train_losses,
        members=50, base_members=10, added_members=args.new_members,
        seeds=np.arange(50), batch=args.batch, stage_epochs=args.stage_epochs,
        learning_rate=LR, sigma_start=SIGMA_START,
        split_seed=int(data["split_seed"]), test_rmse=rmse,
        test_zstd=zstd, test_mixture_nll=mixture_nll,
        architecture="110-64-64-2 tanh; sigma(x)=exp(-3+r(x))",
        training="exact upstream two-stage MAP recipe; seeds 0--9 preserved, 10--49 newly trained",
    )
    print(f"new-seed exact train loss range {train_losses.min():,.0f}--"
          f"{train_losses.max():,.0f}", flush=True)
    print(f"saved {args.output}", flush=True)
    print(f"50-member test RMSE {rmse:.5f}, mixture NLL {mixture_nll:+.5f}, "
          f"z std {zstd:.3f}", flush=True)


if __name__ == "__main__":
    main()
