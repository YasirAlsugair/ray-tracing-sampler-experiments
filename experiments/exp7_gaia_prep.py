"""exp7 step 0: reproduce the paper's pristine cut and pack the pod payload.

Laroche & Speagle 2025 (arXiv:2404.07316) Sec 2.2. Good labels: Teff/sig(Teff)
> 30, sig(logg) < 0.4, sig([M/H]) < 0.2, 0 < BP-RP < 4, 6 < G < 17.5,
STARFLAG == 0, ASPCAPFLAG bits 19 (M_H_BAD) and 23 (STAR_BAD) clear. Pristine
tightens the three label cuts to > 100, < 0.1, < 0.05. No Teff or logg cut:
cool giants emerge. Paper counts: 502,311 -> 202,970 -> 123,804.

Output: results/tables/exp7_gaia_pristine.npz with standardized train/test
splits (seed 2003, 80/20), the norm vectors, and the raw index bookkeeping.
"""
from pathlib import Path

import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"

t = Table.read(ROOT / "data" / "gaia_xp" / "xp_apogee_cat.h5",
               path="__astropy_table__")

X = np.asarray(t["coeffs"], dtype=np.float32)
y = np.asarray(t["ALPHA_M"], dtype=np.float32)
y_err = np.asarray(t["ALPHA_M_ERR"], dtype=np.float32)
teff = np.asarray(t["TEFF"], dtype=np.float64)
teff_err = np.asarray(t["TEFF_ERR"], dtype=np.float64)
logg_err = np.asarray(t["LOGG_ERR"], dtype=np.float64)
mh_err = np.asarray(t["M_H_ERR"], dtype=np.float64)
starflag = np.asarray(t["STARFLAG"], dtype=np.int64)
aspcapflag = np.asarray(t["ASPCAPFLAG"], dtype=np.int64)
gmag = np.asarray(t["GAIAEDR3_PHOT_G_MEAN_MAG"], dtype=np.float64)
bp_rp = (np.asarray(t["GAIAEDR3_PHOT_BP_MEAN_MAG"], dtype=np.float64)
         - np.asarray(t["GAIAEDR3_PHOT_RP_MEAN_MAG"], dtype=np.float64))

print(f"crossmatch: {len(X):,} (paper 502,311)")

with np.errstate(invalid="ignore", divide="ignore"):
    shared = ((bp_rp > 0) & (bp_rp < 4) & (gmag > 6) & (gmag < 17.5)
              & (starflag == 0)
              & (aspcapflag & (1 << 19) == 0) & (aspcapflag & (1 << 23) == 0))
    teff_snr = np.where(teff_err > 0, teff / teff_err, 0.0)
    good = shared & (teff_snr > 30) & (logg_err < 0.4) & (mh_err < 0.2)
    pristine = shared & (teff_snr > 100) & (logg_err < 0.1) & (mh_err < 0.05)

print(f"good:     {good.sum():,} (paper 202,970)")
print(f"pristine: {pristine.sum():,} (paper 123,804)")

usable = (pristine & np.isfinite(y) & np.isfinite(y_err) & (y_err > 0)
          & np.isfinite(X).all(axis=1))
print(f"pristine & finite alpha/coeffs: {usable.sum():,}")

keep = np.flatnonzero(usable)
Xc, yc, ec = X[keep], y[keep], y_err[keep]

rng = np.random.default_rng(2003)
idx = rng.permutation(len(keep))
k = int(0.8 * len(keep))
train, test = idx[:k], idx[k:]

norm_mu = Xc[train].mean(axis=0)
norm_sd = Xc[train].std(axis=0)
train_Xs = (Xc[train] - norm_mu) / norm_sd
test_Xs = (Xc[test] - norm_mu) / norm_sd

print(f"train {len(train):,}  test {len(test):,}")
print(f"train_Xs mean|max {np.abs(train_Xs.mean(axis=0)).max():.2e} "
      f"sd [{train_Xs.std(axis=0).min():.4f}, {train_Xs.std(axis=0).max():.4f}]")
print(f"test_Xs  sd [{test_Xs.std(axis=0).min():.4f}, "
      f"{test_Xs.std(axis=0).max():.4f}]  <- outlier smoke detector")
print(f"y range [{yc.min():.3f}, {yc.max():.3f}]  spread {yc.std():.4f}  "
      f"median err {np.median(ec):.4f}")

out = TAB / "exp7_gaia_pristine.npz"
np.savez_compressed(
    out,
    train_Xs=train_Xs.astype(np.float32), test_Xs=test_Xs.astype(np.float32),
    train_y=yc[train], test_y=yc[test],
    train_yerr=ec[train], test_yerr=ec[test],
    norm_mu=norm_mu, norm_sd=norm_sd,
    keep_rows=keep, train_idx=train, test_idx=test, split_seed=2003,
    cut="pristine: Teff/sig>100, sig(logg)<0.1, sig(M_H)<0.05, 0<BP-RP<4, "
        "6<G<17.5, STARFLAG==0, ASPCAPFLAG bits 19&23 clear, finite alpha")
print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
