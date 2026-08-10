#!/usr/bin/env python3
"""Create v6 Gaia panels from the locked v5 vector PDFs.

The upstream numerical artifacts needed for a full regeneration are intentionally
not duplicated here. This style-only pass preserves every plotted value and vector
mark, removes Matplotlib's two painted white background rectangles, recolors
the four methods into a balanced Gaia-local palette, and shortens the two
ray-tracing legend labels from RTS to RT.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

import sources


POSTER = Path(__file__).resolve().parents[1]
OUT = POSTER / "artifacts" / "figures_v6"
SOURCE_PDFS = {
    "gaia_calibration_by_group": sources.GAIA_CALIBRATION_SOURCE,
    "gaia_nll_vs_members": sources.GAIA_NLL_SOURCE,
}
SOURCE_MAP_PS = "[0.5412 0.5529 0.6] sc"
SOURCE_ENSEMBLE_PS = "[0.1216 0.4353 0.7059] sc"
SOURCE_HMC_PS = "[0.8196 0.2863 0.3569] sc"
SOURCE_RT_GAUSSIAN_PS = "[0.851 0.6039 0.1059] sc"
SOURCE_RT_STUDENT_PS = "[0.5412 0.3725 0.0235] sc"

GAIA_MAP_PS = "[0.6039 0.6275 0.6745] sc"       # #9AA0AC
GAIA_ENSEMBLE_PS = "[0.2235 0.4627 0.7255] sc"  # #3976B9
GAIA_HMC_PS = "[0.502 0.3922 0.6353] sc"        # #8064A2
GAIA_RT_PS = "[0.902 0.6314 0.102] sc"          # #E6A11A
GAIA_RT_GAUSSIAN_PS = "[0.902 0.6314 0.102] sc" # #E6A11A
GAIA_RT_STUDENT_PS = "[0.6039 0.4039 0] sc"     # #9A6700
ENSEMBLE_50 = sources.GAIA_ENSEMBLE_50

# Coordinates in the level-3 PostScript emitted from the locked Matplotlib PDFs.
# Keeping the three non-ensemble curves in that PDF avoids reconstructing them
# from numerical artifacts that were not retained after the cloud run.
NLL_CLIP = "877.625 592 5356.8 1663.2 re"
NLL_X1, NLL_LOG_SCALE = 1121.12, 1244.75
NLL_Y0, NLL_Y_SCALE = 592.0, 2772.0  # -2.0 maps to Y0
CAL_Y0, CAL_Y_SCALE = 552.0, 1108.8


def _run(*args: str) -> None:
    subprocess.run(args, check=True)


def _remove_painted_backgrounds(postscript: str) -> str:
    page = postscript.index("%%Page: 1 1")
    head, body = postscript[:page], postscript[page:]
    # In Matplotlib output the first two filled rectangles on the page are the
    # figure and axes backgrounds. Replace painting with path disposal, retaining
    # the original page box, clipping paths, typography, and all plotted data.
    body, count = re.subn(
        r"(^[-+0-9. ]+ re\n)f$", r"\1newpath", body,
        count=2, flags=re.MULTILINE,
    )
    if count != 2:
        raise RuntimeError(f"expected two background rectangles, replaced {count}")
    return head + body


def _rename_rt_legend(postscript: str) -> str:
    old = "-0.179893 TJm\n(S)\n[12.065\n0] Tj\n-0.102796 TJm\n(,)"
    new = "-0.179893 TJm\n(,)"
    updated = postscript.replace(old, new)
    if updated.count(new) - postscript.count(new) != 2:
        raise RuntimeError("expected to shorten exactly two RTS legend labels")
    return updated


def _remove_tuned_legend(postscript: str) -> str:
    """Shorten the NLL legend from ``SGHMC (tuned)`` to ``SGHMC``."""
    suffix = """0.051398 TJm
( )
[6.042
0] Tj
-0.385485 TJm
(\\()
[7.41
0] Tj
0.334087 TJm
(t)
[7.448
0] Tj
-0.179893 TJm
(u)
[12.046
0] Tj
-0.25699 TJm
(n)
[12.046
0] Tj
-0.25699 TJm
(e)
[11.685
0] Tj
0.102796 TJm
(d)
[12.065
0] Tj
-0.102796 TJm
(\\))
[7.41
0] Tj"""
    if postscript.count(suffix) != 1:
        raise RuntimeError("expected exactly one '(tuned)' SGHMC legend suffix")
    return postscript.replace(suffix, "")


def _recolor_methods(postscript: str, *, nll: bool) -> str:
    """Apply the Gaia-local method palette without disturbing geometry or text."""
    replacements = {
        SOURCE_MAP_PS: GAIA_MAP_PS,
        SOURCE_ENSEMBLE_PS: GAIA_ENSEMBLE_PS,
        SOURCE_HMC_PS: GAIA_HMC_PS,
        SOURCE_RT_GAUSSIAN_PS: GAIA_RT_GAUSSIAN_PS if nll else GAIA_RT_PS,
    }
    if nll:
        replacements[SOURCE_RT_STUDENT_PS] = GAIA_RT_STUDENT_PS
    for old, new in replacements.items():
        old_stroke = old[:-2] + "SC"
        new_stroke = new[:-2] + "SC"
        count = postscript.count(old) + postscript.count(old_stroke)
        if count == 0:
            raise RuntimeError(f"expected at least one PostScript colour command: {old}")
        postscript = postscript.replace(old, new)
        postscript = postscript.replace(old_stroke, new_stroke)
    return postscript


def _emphasize_reference_rule(postscript: str, *, nll: bool) -> str:
    """Strengthen the visual reference without changing its colour or dash."""
    old = "1.4 w\n[5.18 2.24] 0 d" if nll else "1.2 w\n[1.2 1.98] 0 d"
    new = "1.7 w\n[5.18 2.24] 0 d" if nll else "1.7 w\n[1.2 1.98] 0 d"
    if postscript.count(old) != 1:
        raise RuntimeError(f"expected exactly one Gaia reference rule: {old!r}")
    return postscript.replace(old, new)


def convert(name: str, *, rename_rt: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="suds-v6-gaia-") as temp:
        temp_dir = Path(temp)
        source_ps = temp_dir / f"{name}.ps"
        styled_ps = temp_dir / f"{name}_v6.ps"
        _run("pdftops", "-level3", str(SOURCE_PDFS[name]), str(source_ps))
        text = _remove_painted_backgrounds(source_ps.read_text())
        text = _recolor_methods(text, nll=rename_rt)
        text = _emphasize_reference_rule(text, nll=rename_rt)
        if rename_rt:
            text = _rename_rt_legend(text)
            text = _remove_tuned_legend(text)
        styled_ps.write_text(text)
        _run("ps2pdf", "-dEPSCrop", str(styled_ps), str(OUT / f"{name}.pdf"))
    print("wrote", OUT / f"{name}.pdf")


def _ensemble_statistics():
    if not ENSEMBLE_50.exists():
        raise FileNotFoundError(
            f"missing {ENSEMBLE_50}; run train_gaia_ensemble.py first")
    ensemble = np.load(ENSEMBLE_50)
    mus, sig2s = ensemble["mus"], ensemble["sig2s"]
    y, yerr = ensemble["test_y"], ensemble["test_yerr"]
    if len(mus) != 50:
        raise RuntimeError(f"expected 50 ensemble members, found {len(mus)}")

    z = (y - mus.mean(0)) / np.sqrt(yerr**2 + sig2s.mean(0) + mus.var(0))
    edges = (0.0, 0.004, 0.006, 0.01, 0.02)
    z_bins = np.array([
        z[(yerr >= lo) & (yerr < hi)].std()
        for lo, hi in zip(edges[:-1], edges[1:])
    ])

    var = yerr[None, :]**2 + sig2s
    lp = (-0.5 * (y[None, :] - mus)**2 / var
          - 0.5 * np.log(var) - 0.5 * np.log(2.0 * np.pi))
    ks = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50])
    rng = np.random.default_rng(7)
    nll = []
    for k in ks:
        values = []
        for _ in range(25):
            subset = rng.choice(len(lp), size=int(k), replace=False)
            mixture = np.logaddexp.reduce(lp[subset], axis=0) - np.log(k)
            values.append(-mixture.mean())
        nll.append(np.mean(values))
    return ks, np.asarray(nll), z_bins


def _blue_curve_ps(ks: np.ndarray, nll: np.ndarray) -> str:
    colour = "[0.2234 0.4626 0.7246]"
    xy = [(NLL_X1 + NLL_LOG_SCALE * np.log(float(k)),
           NLL_Y0 + NLL_Y_SCALE * (float(value) + 2.0))
          for k, value in zip(ks, nll)]
    lines = ["q", NLL_CLIP, "W", "20 w", "1 J", "/DeviceRGB {} CS",
             f"{colour} SC", f"{xy[0][0]:.3f} {xy[0][1]:.3f} m"]
    lines.extend(f"{x:.3f} {y:.3f} l" for x, y in xy[1:])
    lines.extend(["S", "Q"])
    for x, y in xy:
        lines.extend(["q", "/DeviceRGB {} cs", f"{colour} sc", "newpath",
                      f"{x:.3f} {y:.3f} 25 0 360 arc", "closepath", "f",
                      "10 w", "/DeviceRGB {} CS", f"{colour} SC", "newpath",
                      f"{x:.3f} {y:.3f} 25 0 360 arc", "closepath", "S", "Q"])
    return "\n".join(lines) + "\n"


def _patch_nll_curve(postscript: str, ks: np.ndarray, nll: np.ndarray) -> str:
    blue = r"\[0\.2234 0\.4626 0\.7246\] SC"
    pattern = re.compile(
        rf"q\n{re.escape(NLL_CLIP)}\nW\n20 w\n2 J\n/DeviceRGB \{{\}} CS\n"
        rf"{blue}.*?(?=q\n{re.escape(NLL_CLIP)}\nW\n\[51\.8 22\.4\] 0 d)",
        flags=re.DOTALL,
    )
    updated, count = pattern.subn(_blue_curve_ps(ks, nll), postscript, count=1)
    if count != 1:
        raise RuntimeError(f"expected one deep-ensemble NLL curve, replaced {count}")
    return updated


def _patch_calibration_bars(postscript: str, z_bins: np.ndarray) -> str:
    blue = "[0.2234 0.4626 0.7246] sc"
    purple = "[0.5019 0.3921 0.6348] sc"
    starts = (1219.78, 2487.96, 3756.14, 5024.32)
    widths = (266.32, 266.316, 266.32, 266.316)
    bars = "\n".join(
        f"{x:g} {CAL_Y0:g} {width:g} {CAL_Y_SCALE * value:.3f} re\nf"
        for x, width, value in zip(starts, widths, z_bins)
    )
    pattern = re.compile(
        rf"(/DeviceRGB \{{\}} cs\n{re.escape(blue)}\n).*?"
        rf"(?=/DeviceRGB \{{\}} cs\n{re.escape(purple)})",
        flags=re.DOTALL,
    )
    updated, count = pattern.subn(lambda match: match.group(1) + bars + "\n",
                                  postscript, count=1)
    if count != 1:
        raise RuntimeError(f"expected one set of deep-ensemble bars, replaced {count}")
    return updated


def apply_ensemble_50() -> None:
    ks, nll, z_bins = _ensemble_statistics()
    with tempfile.TemporaryDirectory(prefix="suds-v6-ensemble50-") as temp:
        temp_dir = Path(temp)
        for name, patcher, values in (
            ("gaia_nll_vs_members", _patch_nll_curve, (ks, nll)),
            ("gaia_calibration_by_group", _patch_calibration_bars, (z_bins,)),
        ):
            source_ps = temp_dir / f"{name}.ps"
            styled_ps = temp_dir / f"{name}_50.ps"
            _run("pdftops", "-level3", str(OUT / f"{name}.pdf"), str(source_ps))
            text = patcher(source_ps.read_text(), *values)
            styled_ps.write_text(text)
            _run("ps2pdf", "-dEPSCrop", str(styled_ps), str(OUT / f"{name}.pdf"))
    print("updated Gaia panels with 50 trained deep-ensemble members")


if __name__ == "__main__":
    convert("gaia_calibration_by_group")
    convert("gaia_nll_vs_members", rename_rt=True)
    apply_ensemble_50()
