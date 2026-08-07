"""Retrain the MAP and ensemble baselines and SAVE THE WEIGHTS, so their
OOD alarms can be evaluated (the original packs stored predictions only).
Deterministic reruns of exp7_gaia_map (best lr) and exp7_gaia_ensemble.

    python exp7_baseline_weights.py
        -> results/tables/exp7_map_weights.npz      (flat state)
        -> results/tables/exp7_ensemble_weights.npz (10 flat states)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp7_gaia_hetero as H
from exp7_gaia_map import train_one as map_train
from exp7_gaia_ensemble import N_SEEDS, train_one as ens_train

TAB = H.TAB


def flat(model):
    return torch.cat([p.detach().flatten()
                      for p in model.parameters()]).cpu().numpy()


if __name__ == "__main__":
    Xs, y, yerr, *_ = H.load_data()
    model, loss = map_train(1e-3, Xs, y, yerr)   # the Task 1 winner lr
    np.savez(TAB / "exp7_map_weights.npz", state=flat(model),
             train_loss=loss, lr=1e-3)
    print(f"MAP weights saved (train loss {loss:,.0f})", flush=True)
    states = []
    for seed in range(N_SEEDS):
        model, loss = ens_train(seed, Xs, y, yerr)
        states.append(flat(model))
        print(f"ensemble seed {seed} done ({loss:,.0f})", flush=True)
    np.savez(TAB / "exp7_ensemble_weights.npz", states=np.stack(states))
    print("EXP7W-DONE", flush=True)
