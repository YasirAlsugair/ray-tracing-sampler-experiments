#!/usr/bin/env python3
"""Regenerate the frozen data behind the poster's minibatch-gradient figure.

This provenance step requires the AutoRT repository, its Python environment,
and its raw UCI Superconductivity archive.  It is not part of a normal poster
build.  The output is a compact, plot-ready NPZ consumed by
``minibatch_gradient_figure.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np


SEED = 20260814
N_TRAIN = 17010
BATCHES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 17010)
N_DRAWS = 300
DRAWN_BATCH = 2048
N_LEFT_ARROWS = 42


def _fan(target, theta, batch, key, n_draws):
    keys = jax.random.split(key, n_draws)
    rows = jax.vmap(lambda k: jax.random.choice(
        k, target.n_train, shape=(batch,), replace=False))(keys)
    return np.asarray(jax.vmap(lambda r: target.grad_on_rows(theta, r))(rows))


def _angles(fan, exact):
    cosine = ((fan @ exact)
              / (np.linalg.norm(fan, axis=1) * np.linalg.norm(exact)))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _polar(fan, exact, side):
    scale = np.linalg.norm(exact)
    radius = np.linalg.norm(fan, axis=1) / scale
    angle = np.radians(_angles(fan, exact))
    angle = np.where(fan @ side >= 0, angle, -angle)
    return radius * np.cos(angle), radius * np.sin(angle)


def prepare(autort_root: Path, output: Path):
    autort_root = autort_root.resolve()
    if str(autort_root) not in sys.path:
        sys.path.insert(0, str(autort_root))
    from rtbench.targets.blr import blr_superconductor

    target = blr_superconductor(n=N_TRAIN)
    draw_key, fan_key, arrow_key = jax.random.split(jax.random.PRNGKey(SEED), 3)
    theta = target.ref_sample(draw_key, 1)[0]
    exact = np.asarray(target.grad(theta))

    probe = np.asarray(jax.random.normal(jax.random.PRNGKey(SEED + 1),
                                         (target.dim,)))
    unit = exact / np.linalg.norm(exact)
    side = probe - (probe @ unit) * unit
    side /= np.linalg.norm(side)

    keys = jax.random.split(fan_key, len(BATCHES))
    angles = {
        batch: _angles(_fan(target, theta, batch, key, N_DRAWS), exact)
        for batch, key in zip(BATCHES, keys)
    }
    fan = _fan(target, theta, DRAWN_BATCH, arrow_key, N_LEFT_ARROWS)
    left_x, left_y = _polar(fan, exact, side)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        batches=np.asarray(BATCHES, dtype=np.int64),
        mean_angle=np.asarray([angles[b].mean() for b in BATCHES]),
        q10_angle=np.asarray([np.quantile(angles[b], 0.1) for b in BATCHES]),
        q90_angle=np.asarray([np.quantile(angles[b], 0.9) for b in BATCHES]),
        left_arrow_x=left_x,
        left_arrow_y=left_y,
        drawn_batch=np.asarray(DRAWN_BATCH, dtype=np.int64),
        drawn_mean_angle=np.asarray(angles[DRAWN_BATCH].mean()),
        n_train=np.asarray(target.n_train, dtype=np.int64),
        dimension=np.asarray(target.dim, dtype=np.int64),
        seed=np.asarray(SEED, dtype=np.int64),
        n_draws=np.asarray(N_DRAWS, dtype=np.int64),
        n_left_arrows=np.asarray(N_LEFT_ARROWS, dtype=np.int64),
    )
    print(f"wrote {output}")


def main():
    poster = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--autort-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=poster / "artifacts" / "data_v6" / "minibatch_gradient_data.npz",
    )
    args = parser.parse_args()
    prepare(args.autort_root, args.output)


if __name__ == "__main__":
    main()
