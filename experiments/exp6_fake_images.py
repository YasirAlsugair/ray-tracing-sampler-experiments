"""Fake-image (out-of-distribution) test: show each method images that are
not digits and measure who stays confident.

Three fake sets, 10,000 images each, in the same normalized space as the
training data: gaussian noise, Fashion-MNIST (real structure, wrong domain),
and inverted MNIST (digits with polarity flipped). For each method and set:
the median top-class confidence (an honest method should drop), the fraction
flagged as uncertain by the same threshold that flags 11% of the real test
set, and the AUROC separating real from fake images by the uncertainty
signal alone (0.5 = blind, 1.0 = perfect).

Ensemble members are rebuilt with their original seeds (they were saved only
as test-set probabilities). Saves results/tables/exp6_fake_images.npz.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import exp6_simple_mnist_train as base
import exp6_ensemble as ensemble_script

DEV = base.DEV
TAB = base.TAB
CHAIN_MEMBERS = 50
DROPOUT_PASSES = 50
TEST_FLAG_RATE = 0.11  # the review budget used throughout the notebook


class MLPWithDropout(nn.Module):
    def __init__(self, hidden=64, dropout_rate=0.25):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden),
                                 nn.ReLU(), nn.Dropout(dropout_rate),
                                 nn.Linear(hidden, 10))

    def forward(self, images):
        return self.net(images)


def image_sets():
    normalize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    mnist = datasets.MNIST(base.ROOT / "data", train=False, download=True,
                           transform=normalize)
    real = torch.stack([x for x, _ in mnist])

    fashion = datasets.FashionMNIST(base.ROOT / "data", train=False,
                                    download=True, transform=normalize)
    fashion_images = torch.stack([x for x, _ in fashion])

    raw = datasets.MNIST(base.ROOT / "data", train=False, download=True,
                         transform=transforms.ToTensor())
    inverted = torch.stack([
        transforms.functional.normalize(1.0 - x, (0.1307,), (0.3081,))
        for x, _ in raw])

    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(10000, 1, 28, 28, generator=generator)

    return real, {"gaussian noise": noise,
                  "Fashion-MNIST": fashion_images,
                  "inverted MNIST": inverted}


@torch.no_grad()
def probabilities(model, images, stochastic=False):
    model.train() if stochastic else model.eval()
    out = []
    for i in range(0, len(images), 2000):
        out.append(torch.softmax(model(images[i:i + 2000].to(DEV)), dim=1)
                   .cpu().numpy())
    return np.concatenate(out)


def chain_member_vectors():
    files = [TAB / "exp6_mb33_dt0.0001_rr5.npz"] + sorted(
        TAB.glob("exp6_mb33_dt0.0001_rr5_part*.npz"),
        key=lambda p: int(p.stem.rsplit("part", 1)[1]))
    counts = [len(np.load(f)["steps"]) for f in files]
    offsets = np.cumsum([0] + counts)
    wanted = np.linspace(3 * offsets[-1] // 4, offsets[-1] - 1,
                         CHAIN_MEMBERS).astype(int)
    rows_by_file = {}
    for index in wanted:
        k = int(np.searchsorted(offsets, index, side="right") - 1)
        rows_by_file.setdefault(k, []).append(int(index - offsets[k]))
    vectors = []
    for k in sorted(rows_by_file):
        snapshots = np.load(files[k])["snapshots"]
        vectors.extend(snapshots[row] for row in rows_by_file[k])
    return vectors


def vector_into(model, vector):
    theta, i = torch.tensor(vector, device=DEV), 0
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(theta[i:i + parameter.numel()]
                            .view(parameter.shape))
            i += parameter.numel()


def auroc(real_signal, fake_signal):
    """P(fake more uncertain than real), rank-based."""
    combined = np.concatenate([real_signal, fake_signal])
    ranks = combined.argsort().argsort().astype(float) + 1
    fake_ranks = ranks[len(real_signal):]
    n_real, n_fake = len(real_signal), len(fake_signal)
    return (fake_ranks.sum() - n_fake * (n_fake + 1) / 2) / (n_real * n_fake)


def main():
    real, fakes = image_sets()
    all_sets = {"real test": real, **fakes}

    print("building the four methods...", flush=True)
    point = base.MLP().to(DEV)
    point.load_state_dict(torch.load(base.CKPT / "exp6_mlp.pt"))

    members = []
    for seed in range(ensemble_script.M):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = base.MLP().to(DEV)
        train_loader, _, _ = base.loaders()
        optimizer = torch.optim.Adam(model.parameters(), lr=base.LR)
        for _ in range(base.EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEV), y.to(DEV)
                optimizer.zero_grad()
                F.cross_entropy(model(x), y).backward()
                optimizer.step()
        members.append(model)
        print(f"  ensemble member {seed} retrained", flush=True)

    dropout_model = MLPWithDropout().to(DEV)
    dropout_model.load_state_dict(torch.load(base.CKPT / "exp6_mlp_dropout.pt"))
    chain_vectors = chain_member_vectors()
    chain_model = base.MLP().to(DEV)

    def method_probabilities(images):
        """Per method: stacked member probabilities on one image set."""
        out = {"point": probabilities(point, images)[None]}
        out["ensemble"] = np.stack([probabilities(m, images) for m in members])
        torch.manual_seed(0)
        out["dropout"] = np.stack([probabilities(dropout_model, images,
                                                 stochastic=True)
                                   for _ in range(DROPOUT_PASSES)])
        chain = []
        for vector in chain_vectors:
            vector_into(chain_model, vector)
            chain.append(probabilities(chain_model, images))
        out["chain"] = np.stack(chain)
        return out

    def uncertainty(name, member_probabilities):
        if name == "point":
            return 1.0 - member_probabilities[0].max(axis=1)
        return member_probabilities.std(axis=0).max(axis=1)

    signals, medians = {}, {}
    for set_name, images in all_sets.items():
        print(f"scoring {set_name}...", flush=True)
        per_method = method_probabilities(images)
        for method, member_probs in per_method.items():
            signals[(method, set_name)] = uncertainty(method, member_probs)
            medians[(method, set_name)] = float(
                np.median(member_probs.mean(axis=0).max(axis=1)))

    methods = ["point", "ensemble", "dropout", "chain"]
    thresholds = {m: np.quantile(signals[(m, "real test")],
                                 1 - TEST_FLAG_RATE) for m in methods}

    results = {}
    print(f"\n{'method':>10} {'fake set':>16} {'median conf':>12} "
          f"{'flagged':>8} {'AUROC':>6}")
    for method in methods:
        for set_name in fakes:
            flag_rate = float(np.mean(signals[(method, set_name)]
                                      > thresholds[method]))
            score = auroc(signals[(method, "real test")],
                          signals[(method, set_name)])
            results[f"{method}|{set_name}"] = (medians[(method, set_name)],
                                               flag_rate, score)
            print(f"{method:>10} {set_name:>16} "
                  f"{medians[(method, set_name)]:>12.3f} {flag_rate:>7.0%} "
                  f"{score:>6.2f}")
        print(f"{method:>10} {'(real test)':>16} "
              f"{medians[(method, 'real test')]:>12.3f} "
              f"{TEST_FLAG_RATE:>7.0%}   -")

    np.savez(TAB / "exp6_fake_images.npz",
             methods=methods, sets=list(fakes),
             medians=np.array([[medians[(m, s)] for s in fakes]
                               for m in methods]),
             real_medians=np.array([medians[(m, "real test")]
                                    for m in methods]),
             flag_rates=np.array([[results[f"{m}|{s}"][1] for s in fakes]
                                  for m in methods]),
             aurocs=np.array([[results[f"{m}|{s}"][2] for s in fakes]
                              for m in methods]),
             test_flag_rate=TEST_FLAG_RATE)
    print(f"\nsaved {TAB / 'exp6_fake_images.npz'}")


if __name__ == "__main__":
    main()
