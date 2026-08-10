#!/usr/bin/env python3
"""Regenerate every figure consumed by the poster v6 candidate."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BUILDERS = (
    "data_bands_figure.py",
    "typical_set_figure.py",
    "minibatch_gradient_figure.py",
    "noise_geometry_figure.py",
    "tolerance_erosion_frontier_candidate_figure.py",
    "gaia_panels_v6.py",
)


def main():
    for builder in BUILDERS:
        print(f"building with {builder}", flush=True)
        subprocess.run([sys.executable, str(HERE / builder)], check=True)


if __name__ == "__main__":
    main()
