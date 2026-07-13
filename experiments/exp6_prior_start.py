"""Josh's suggestion (2026-07-21): run an optimizer WITH the prior enabled to
get chain starting positions, hoping to pre-pay the burn-in walk to the shell.

Three MAP optimizations of the same objective, sum CE + 0.5||theta||^2
(implemented per batch as mean CE + ||theta||^2 / (2N)), differing only in the
starting point:

  adam_ckpt   from the step 1 Adam checkpoint (norm^2 ~ 174)
  random      from a fresh default initialization
  prior_draw  from theta ~ N(0, I)  (norm^2 ~ D = 50,890, saturated start)

Logged per epoch: training misfit (sum CE, nats) and ||theta||^2, against the
typical-set targets (misfit ~ 350, norm^2 ~ 49,600). Final states saved to
results/tables/exp6_map_starts.npz for use as chain starting points.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"
N = 60_000
EPOCHS = 5
LR = 1e-3
BATCH = 128


class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 10))

    def forward(self, x):
        return self.net(x)


def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def norm2(model):
    return float(sum((p.detach() ** 2).sum() for p in model.parameters()))


@torch.no_grad()
def misfit(model, images, labels):
    total = 0.0
    for i in range(0, N, 4096):
        total += F.cross_entropy(model(images[i:i + 4096]),
                                 labels[i:i + 4096], reduction="sum").item()
    return total


def run(tag, model, images, labels):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(0)
    print(f"[{tag}] epoch 0: misfit {misfit(model, images, labels):>12,.0f}  "
          f"norm^2 {norm2(model):>10,.0f}", flush=True)
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(N, generator=generator)
        for i in range(0, N, BATCH):
            idx = order[i:i + BATCH].to(DEV)
            optimizer.zero_grad()
            prior = sum((p ** 2).sum() for p in model.parameters()) / (2 * N)
            loss = F.cross_entropy(model(images[idx]), labels[idx]) + prior
            loss.backward()
            optimizer.step()
        print(f"[{tag}] epoch {epoch}: misfit {misfit(model, images, labels):>12,.0f}  "
              f"norm^2 {norm2(model):>10,.0f}", flush=True)
    return flat(model).cpu().numpy()


def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(ROOT / "data", train=True, download=True,
                           transform=transform)
    images = torch.stack([x for x, _ in train]).to(DEV)
    labels = torch.tensor(train.targets).to(DEV)

    states = {}

    model = MLP().to(DEV)
    model.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / "exp6_mlp.pt",
                                     map_location=DEV))
    states["adam_ckpt"] = run("adam_ckpt", model, images, labels)

    torch.manual_seed(1)
    states["random"] = run("random", MLP().to(DEV), images, labels)

    torch.manual_seed(2)
    model = MLP().to(DEV)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p))
    states["prior_draw"] = run("prior_draw", model, images, labels)

    np.savez(TAB / "exp6_map_starts.npz", **states)
    print(f"\ntargets: typical set misfit ~350, norm^2 ~49,600 (shell D = 50,890)")
    print(f"saved {TAB / 'exp6_map_starts.npz'}")


if __name__ == "__main__":
    main()
