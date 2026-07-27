"""Build results/tables/exp7_analysis_pack.npz, the compact handoff for
downstream analysis (calibration anatomy, star cards, saliency vs Teff).

Collapses the ~0.5 GB of chain snapshot files into what the analysis
actually consumes: 50-member test-split predictions per chain, the
member-mean input gradient per test star, and the star metadata. Everything
here regenerates from the chain npz files (not in git) plus the data
payload from exp7_gaia_prep.py.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")

d = np.load(TAB / "exp7_gaia_pristine.npz")
sp = np.load(TAB / "exp7_gaia_spectra_payload.npz")
X = torch.tensor(d["test_Xs"], device=DEV)

CHAINS = {
    "gated5e6": [TAB / "exp7_rt33_dt5e-06.npz"],   # the chain of record
    "normal": [TAB / "exp7n_rt33_dt0.0001.npz"],   # N(0,1) prior ablation
}


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(110, 64), nn.Tanh(),
                         nn.Linear(64, 64), nn.Tanh(),
                         nn.Linear(64, 1)).to(DEV)


def members(files, n=50):
    snaps = np.concatenate([np.load(f)["snapshots"] for f in files])
    quarter = snaps[3 * len(snaps) // 4:]
    return quarter[np.linspace(0, len(quarter) - 1, n).astype(int)]


def load_flat(model, flat):
    flat = torch.tensor(flat, dtype=torch.float32, device=DEV)
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(flat[i:i + p.numel()].view(p.shape))
            i += p.numel()


model = make_model()
out = {}
for name, files in CHAINS.items():
    preds = []
    with torch.no_grad():
        for s in members(files):
            load_flat(model, s)
            preds.append(model(X).squeeze(-1).cpu().numpy())
    out[f"member_preds_{name}"] = np.stack(preds).astype(np.float32)
    print(f"{name}: member_preds {out[f'member_preds_{name}'].shape}")

# member-mean d(prediction)/d(standardized coefficient), per test star,
# from the chain of record
grad_sum = torch.zeros_like(X)
for s in members(CHAINS["gated5e6"]):
    load_flat(model, s)
    x = X.clone().requires_grad_(True)
    model(x).squeeze(-1).sum().backward()
    grad_sum += x.grad
grad_mean = (grad_sum / 50).cpu().numpy().astype(np.float32)
print(f"grad_mean {grad_mean.shape}")

np.savez_compressed(
    TAB / "exp7_analysis_pack.npz",
    **out,
    grad_mean_std_coords=grad_mean,
    norm_mu=d["norm_mu"], norm_sd=d["norm_sd"],
    test_y=d["test_y"], test_yerr=d["test_yerr"],
    teff=sp["teff"], mh=sp["mh"], sids=sp["sids"],
    split_seed=2003,
    note="members = 50 evenly thinned over each chain's final quarter; "
         "gated5e6 is the chain of record (Eq. 33 gate on; its member "
         "spread is a LOWER BOUND, the gate slows mixing); normal is the "
         "N(0,1) prior ablation; grad in standardized coords, divide by "
         "norm_sd for raw; spectra live in exp7_gaia_spectra_payload.npz")
size = (TAB / "exp7_analysis_pack.npz").stat().st_size / 1e6
print(f"pack written: {size:.1f} MB")
