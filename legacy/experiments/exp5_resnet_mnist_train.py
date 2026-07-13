"""Exp 5a: train a ResNet-50 on MNIST (the point estimate that seeds the samplers).

Question: get a stock torchvision ResNet-50 (num_classes=10, 1-channel input handled
by repeating the channel) to a converged MNIST fit, so exp5_resnet_mnist_sample.py can
sample the weight posterior around it with the ray tracer and an HMC baseline.

Choices that define the posterior later (kept deliberately clean):
  - No data augmentation: the likelihood is a fixed function of the 60k training images.
  - No weight decay: flat prior, matching the paper's NN runs (App. E).
  - The sampler will run the net in eval() mode, so the BatchNorm buffers frozen at the
    end of this training run become part of the model definition.

Runs on Apple MPS in float32. Does NOT import rts.config (that pins jax/CPU/float64,
the wrong stack for this experiment).

Outputs:
  results/checkpoints/exp5_resnet50_mnist.pt   (state_dict)
  results/tables/exp5_train_log.json           (per-epoch stats + final eval-mode losses)
"""
import sys, os, json, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.models import resnet50

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
CKPT = os.path.join(HERE, "..", "results", "checkpoints")
TAB = os.path.join(HERE, "..", "results", "tables")

SEED = 20260706
EPOCHS = 6
BATCH = 256
LR = 1e-3
LR_DROP_EPOCH = 5   # from this epoch on, LR/10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_split(train):
    ds = datasets.MNIST(DATA, train=train, download=True)
    x = ds.data.float().div_(255.0).sub_(MNIST_MEAN).div_(MNIST_STD).unsqueeze(1)
    return x.to(DEVICE), ds.targets.to(DEVICE)


def forward(model, x):
    return model(x.expand(x.shape[0], 3, 28, 28))


@torch.no_grad()
def evaluate(model, x, y, batch=1024):
    model.eval()
    ce_sum, correct = 0.0, 0
    for i in range(0, len(x), batch):
        logits = forward(model, x[i:i + batch])
        ce_sum += F.cross_entropy(logits, y[i:i + batch], reduction="sum").item()
        correct += (logits.argmax(1) == y[i:i + batch]).sum().item()
    return ce_sum / len(x), correct / len(x)


def main():
    torch.manual_seed(SEED)
    xtr, ytr = load_split(train=True)
    xte, yte = load_split(train=False)
    n = len(xtr)
    print(f"device={DEVICE}  train={n}  test={len(xte)}")

    model = resnet50(weights=None, num_classes=10).to(DEVICE)
    D = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet-50 (10 classes): D = {D:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    log = {"config": {"seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                      "lr_drop_epoch": LR_DROP_EPOCH, "device": DEVICE, "D": D},
           "epochs": []}

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        if epoch == LR_DROP_EPOCH:
            for g in opt.param_groups:
                g["lr"] = LR / 10
        model.train()
        perm = torch.randperm(n)
        ce_run, seen, te = 0.0, 0, time.time()
        for k, i in enumerate(range(0, n, BATCH)):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(forward(model, xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            ce_run += loss.item() * len(idx)
            seen += len(idx)
            if epoch == 1 and k == 19:
                print(f"  step rate ~ {20 / (time.time() - te):.2f} it/s at batch {BATCH}")
        test_ce, test_acc = evaluate(model, xte, yte)
        row = {"epoch": epoch, "train_ce": ce_run / seen, "test_ce": test_ce,
               "test_acc": test_acc, "seconds": time.time() - te}
        log["epochs"].append(row)
        print(f"epoch {epoch}/{EPOCHS}  train_ce {row['train_ce']:.4f}  "
              f"test_ce {test_ce:.4f}  test_acc {test_acc:.4f}  ({row['seconds']:.0f}s)")

    train_ce_eval, train_acc_eval = evaluate(model, xtr, ytr)
    test_ce, test_acc = evaluate(model, xte, yte)
    log["final"] = {"train_ce_eval_mode": train_ce_eval, "train_acc_eval_mode": train_acc_eval,
                    "test_ce": test_ce, "test_acc": test_acc,
                    "seconds_total": time.time() - t0}
    print(f"final (eval mode): train_ce {train_ce_eval:.5f}  train_acc {train_acc_eval:.4f}  "
          f"test_ce {test_ce:.4f}  test_acc {test_acc:.4f}")

    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CKPT, "exp5_resnet50_mnist.pt"))
    with open(os.path.join(TAB, "exp5_train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("...done. Checkpoint + training log written.")


if __name__ == "__main__":
    main()
