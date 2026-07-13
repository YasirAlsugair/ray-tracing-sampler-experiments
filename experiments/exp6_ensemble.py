"""Step 3: deep-ensemble baseline on MNIST (random-seed variant).

10 members per architecture, same recipe as the step-1 point estimates.
Saves per-member test probabilities for the step-5 comparison.

Outputs:
  results/tables/exp6_ensemble_{mlp,cnn}.npz  (probs [M,10000,10], labels)
  results/tables/exp6_ensemble_summary.csv
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import exp6_simple_mnist_train as base

TAB = base.TAB
M = 10  # ensemble members


def train_member(cls, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = cls().to(base.DEV)
    tr, te, _ = base.loaders()
    opt = torch.optim.Adam(model.parameters(), lr=base.LR)
    for _ in range(base.EPOCHS):
        model.train()
        for x, y in tr:
            x, y = x.to(base.DEV), y.to(base.DEV)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    acc, nll, probs, labels = base.evaluate(model, te)
    return acc, nll, probs, labels


def ece(probs, labels, bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return out


if __name__ == "__main__":
    rows = []
    for name, cls in [("mlp", base.MLP), ("cnn", base.CNN)]:
        t0 = time.time()
        member_probs, accs = [], []
        labels = None
        for seed in range(M):
            acc, nll, probs, labels = train_member(cls, seed)
            member_probs.append(probs)
            accs.append(acc)
            print(f"[{name}] member {seed}: acc={acc:.4f} nll={nll:.4f}", flush=True)
        P = np.stack(member_probs)  # [M, 10000, 10]
        ens = P.mean(0)
        ens_acc = (ens.argmax(1) == labels).mean()
        ens_nll = -np.log(ens[np.arange(len(labels)), labels] + 1e-12).mean()
        spread = P.std(0).max(1)  # per-image max-class std across members
        np.savez(TAB / f"exp6_ensemble_{name}.npz", probs=P, labels=labels)
        row = dict(
            name=name,
            mean_member_acc=float(np.mean(accs)),
            ens_acc=float(ens_acc),
            ens_nll=float(ens_nll),
            ens_ece=float(ece(ens, labels)),
            median_spread=float(np.median(spread)),
            train_s=time.time() - t0,
        )
        rows.append(row)
        print(f"[{name}] ensemble of {M}: acc={ens_acc:.4f} nll={ens_nll:.4f} "
              f"ece={row['ens_ece']:.4f} ({row['train_s']:.0f}s)", flush=True)
    with open(TAB / "exp6_ensemble_summary.csv", "w") as f:
        cols = list(rows[0].keys())
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print("saved", TAB / "exp6_ensemble_summary.csv")
