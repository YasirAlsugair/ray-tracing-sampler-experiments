#!/usr/bin/env python3
"""Regenerate the portable code-built figures used by poster v8."""
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
    "tolerance_erosion_frontier_figure.py",
)


def main():
    for builder in BUILDERS:
        print(f"building with {builder}", flush=True)
        subprocess.run([sys.executable, str(HERE / builder)], check=True)


if __name__ == "__main__":
    main()
