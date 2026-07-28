"""Build the out-of-distribution payload for exp7: real stars the pristine
cut rejected, as three physics classes, standardized with the training
split's own vectors. The exp6 fake-images test with zero fake data.

    dwarfs   logg > 4.2, Teff 4500-6500 (main sequence, never seen)
    hot      Teff > 6000 (wrong temperature regime entirely)
    flagged  cool giants by Teff/logg but STARFLAG != 0 (bad spectra)

Output: results/tables/exp7_ood_payload.npz with standardized coefficient
matrices (up to 5000 stars per class) plus Teff/logg for bookkeeping.
"""
from pathlib import Path

import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"
rng = np.random.default_rng(7)

t = Table.read(ROOT / "data/gaia_xp/xp_apogee_cat.h5", path="__astropy_table__")
d = np.load(TAB / "exp7_gaia_pristine.npz")
in_sample = np.zeros(len(t), dtype=bool)
in_sample[d["keep_rows"]] = True

coeffs = np.asarray(t["coeffs"], dtype=np.float32)
teff = np.asarray(t["TEFF"], dtype=np.float64)
logg = np.asarray(t["LOGG"], dtype=np.float64)
starflag = np.asarray(t["STARFLAG"], dtype=np.int64)
clean = ~in_sample & np.isfinite(coeffs).all(axis=1) & np.isfinite(teff) & np.isfinite(logg)

classes = {
    "dwarfs": clean & (logg > 4.2) & (teff > 4500) & (teff < 6500),
    "hot": clean & (teff > 6000),
    "flagged": clean & (teff < 5000) & (logg < 3.5) & (starflag != 0),
}

out = {}
for name, mask in classes.items():
    rows = np.flatnonzero(mask)
    if len(rows) > 5000:
        rows = rng.choice(rows, 5000, replace=False)
    Xs = (coeffs[rows] - d["norm_mu"]) / d["norm_sd"]
    out[f"Xs_{name}"] = Xs.astype(np.float32)
    out[f"teff_{name}"] = teff[rows].astype(np.float32)
    out[f"logg_{name}"] = logg[rows].astype(np.float32)
    print(f"{name}: {mask.sum():,} available, kept {len(rows):,}, "
          f"median |z-coeff| {np.median(np.abs(Xs)):.2f}")

np.savez_compressed(TAB / "exp7_ood_payload.npz", **out,
                    note="cut-away stars, standardized with the training vectors")
print("saved", TAB / "exp7_ood_payload.npz")
