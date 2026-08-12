#!/usr/bin/env python3
"""Regenerate the code-built figure overrides for poster v7."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BUILDERS = (
    "noise_geometry_figure.py",
    "tolerance_erosion_frontier_figure.py",
)


def main():
    for builder in BUILDERS:
        print(f"building with {builder}", flush=True)
        subprocess.run([sys.executable, str(HERE / builder)], check=True)


if __name__ == "__main__":
    main()
