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
    "scatter": [TAB / "exp7s_rt33_dt2e-05.npz"],   # THE CHAIN OF RECORD
    "gated5e6": [TAB / "exp7_rt33_dt5e-06.npz"],   # previous record
    "normal": [TAB / "exp7n_rt33_dt0.0001.npz"],   # N(0,1) prior ablation
}
LNS0 = -3.0  # scatter chain: ln s = LNS0 + u, u is the final parameter


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
D_net = sum(p.numel() for p in model.parameters())
for name, files in CHAINS.items():
    preds, s_vals = [], []
    with torch.no_grad():
        for st in members(files):
            if len(st) == D_net + 1:            # scatter chain carries u last
                s_vals.append(math.exp(LNS0 + float(st[-1])))
                st = st[:-1]
            load_flat(model, st)
            preds.append(model(X).squeeze(-1).cpu().numpy())
    out[f"member_preds_{name}"] = np.stack(preds).astype(np.float32)
    if s_vals:
        out[f"member_s_{name}"] = np.array(s_vals, dtype=np.float32)
    print(f"{name}: member_preds {out[f'member_preds_{name}'].shape}"
          + (f", s median {np.median(s_vals):.4f}" if s_vals else ""))

# member-mean d(prediction)/d(standardized coefficient), per test star,
# from the chain of record
grad_sum = torch.zeros_like(X)
for st in members(CHAINS["scatter"]):
    load_flat(model, st[:-1])
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
         "scatter is THE CHAIN OF RECORD (corrected noise model, converged "
         "on all drift checks, z std 0.95; member_s_scatter carries each "
         "member's intrinsic scatter s); gated5e6 is the previous record "
         "(collapsed spread), normal the N(0,1) ablation; grad in "
         "standardized coords, divide by norm_sd for raw; spectra live in "
         "exp7_gaia_spectra_payload.npz")
size = (TAB / "exp7_analysis_pack.npz").stat().st_size / 1e6
print(f"pack written: {size:.1f} MB")
