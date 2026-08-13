#!/usr/bin/env python3
"""Regenerate the code-built figures pinned by poster v9."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BUILDERS = (
    "data_bands_figure.py",
    "typical_set_figure.py",
    "intro_dynamics_figure.py",
    "minibatch_gradient_figure.py",
    "noise_geometry_figure.py",
    "tolerance_erosion_frontier_figure.py",
)


def main() -> None:
    for builder in BUILDERS:
        print(f"building with {builder}", flush=True)
        subprocess.run([sys.executable, str(HERE / builder)], check=True)


if __name__ == "__main__":
    main()
