"""Step 1 of the 2026-07-07 plan: simple MLP vs CNN point estimates on MNIST.

Models are deliberately tiny (D in the tens of thousands) so that step 4 can run
the ray tracing sampler with full-batch gradients and the exact Metropolis test.

Outputs:
  results/checkpoints/exp6_mlp.pt, exp6_cnn.pt
  results/tables/exp6_point_{mlp,cnn}.npz   (test probs + labels, for steps 3/5)
  results/tables/exp6_point_summary.csv
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results" / "checkpoints"
TAB = ROOT / "results" / "tables"
CKPT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

SEED = 0
EPOCHS = 5
BATCH = 128
LR = 1e-3
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.linear = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.linear(x.flatten(1))


def loaders():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    tr = datasets.MNIST(ROOT / "data", train=True, download=True, transform=tfm)
    te = datasets.MNIST(ROOT / "data", train=False, download=True, transform=tfm)
    return (
        torch.utils.data.DataLoader(tr, batch_size=BATCH, shuffle=True),
        torch.utils.data.DataLoader(te, batch_size=1000),
        len(tr),
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs, labels, nll_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        logits = model(x)
        nll_sum += F.cross_entropy(logits, y, reduction="sum").item()
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
        n += len(y)
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    acc = (probs.argmax(1) == labels).mean()
    return acc, nll_sum / n, probs, labels


def train_one(name, model):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = model.to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    tr, te, n_train = loaders()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        run = 0.0
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item()
        acc, nll, _, _ = evaluate(model, te)
        print(
            f"[{name}] epoch {ep + 1}/{EPOCHS} "
            f"train_loss={run / len(tr):.4f} test_acc={acc:.4f} test_nll={nll:.4f}"
        )
    dt = time.time() - t0
    acc, nll, probs, labels = evaluate(model, te)
    torch.save(model.state_dict(), CKPT / f"exp6_{name}.pt")
    np.savez(TAB / f"exp6_point_{name}.npz", probs=probs, labels=labels)
    print(f"[{name}] D={n_params}  final test_acc={acc:.4f}  test_nll={nll:.4f}  ({dt:.0f}s)")
    return dict(name=name, D=n_params, acc=acc, nll=nll, train_s=dt, n_train=n_train)


if __name__ == "__main__":
    rows = [train_one("mlp", MLP()), train_one("cnn", CNN())]
    with open(TAB / "exp6_point_summary.csv", "w") as f:
        f.write("model,D,test_acc,test_nll,train_seconds,n_train\n")
        for r in rows:
            f.write(
                f"{r['name']},{r['D']},{r['acc']:.4f},{r['nll']:.4f},"
                f"{r['train_s']:.0f},{r['n_train']}\n"
            )
    print("saved", TAB / "exp6_point_summary.csv")
