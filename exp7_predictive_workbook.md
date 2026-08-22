# Workbook: per-star posterior predictive (Josh's whiteboard, 2026-08 meeting)

You write every line of code yourself. Claude reviews against the checkpoints.
Rule of the exercise: before you code each task, say out loud (or write in one
sentence) what the plot will show and why. If you cannot, go back to the
explanation first.

## The one equation

For one star x, with M posterior draws theta_1..theta_M:

    p(y | x, D)  =  integral of  p(y | mu, sigma, nu) p(mu, sigma, nu | x, D)
                 ~=  (1/M) sum_m  t_{nu_m}( y ; mu_m(x), s_m(x) )

MAP plug-in keeps one curve, p(y | x, theta_MAP), and drops the cloud.

## Data inventory (verified on disk 2026-08-09)

- `results/tables/exp7t_pack_final.npz`: `members` (50, 11395), the t chain.
  Layout is [w, net]: index 0 is theta_nu, the rest is the network.
  Caution: use the driver's own flatten/unflatten helpers from
  `experiments/exp7_gaia_t.py`. Do not hand-slice; the [w, net] ordering
  already bit us once (the 2026-07-28 ordering bug).
- `results/tables/exp7h_pack3.npz`: `members` (50, 11394), the Gaussian chain
  (nu = infinity comparison). Members are clean; `final_state` is not (NaN-era,
  never use it as a start or a plug-in).
- `results/tables/exp7_map_weights.npz`: the Adam MAP fit. NOTE: it was fit to
  the GAUSSIAN hetero target, not the t target. Decide (Task B, step 0) what
  the honest plug-in comparison is and write your choice down.
- Test split: same seed-2003 80/20 split as everything else
  (`exp7_gaia_pristine.npz`).

## Task A: the 1-star scatterplot

Goal: for one chosen star, scatter the 50 points (mu_m, s_m), colored by nu_m.
Mark the MAP prediction as a single distinct point.

Hints, in order (stop reading when you don't need the next one):
1. Rebuild TModel, load one member vector, forward the star, read off
   mu, then s from yerr^2 + exp(2*(LNS0 + r)), then nu from model.nu().
2. Loop over the 50 members; a (50, 3) array is all you need.
3. Pick three stars: one typical member, one high catalog-error star, one
   from the OOD panel (hot star or dwarf). Three panels, shared axes if you
   can, so the cloud sizes are comparable.

Checkpoints:
- [ ] nu_m across members lands inside roughly [5.4, 5.6] (the certified
      interval was 5.48 [5.43, 5.53]). If you see nu near 7, you loaded the
      prior center, not the posterior.
- [ ] For the typical star, s_m clusters near ~0.02-0.03 and the mu cloud
      spread is several times SMALLER than s_m.
- [ ] For the OOD star, the mu cloud spread grows toward the same order as
      s_m. If all three clouds look identical, something is wrong.
- [ ] The MAP point sits inside or near the cloud (it is a fit to the same
      data), but it is one point: that is the visual argument.

## Task B: MAP plug-in vs full RTS marginal

Goal: for the same three stars, overlay on a y-grid:
(a) the MAP plug-in density (one curve),
(b) the RTS marginal, the average of the 50 member t densities (one mixture
    curve), and optionally (c) a few faint individual member curves so the
    mixture is visibly their average.
Mark the true catalog y as a vertical line.

Step 0 (decision, write it in the notebook): the stored MAP is Gaussian-target.
Either compare Gaussian-MAP-plug-in vs Gaussian-chain-marginal (clean
apples-to-apples) and show the t marginal as a third curve, or use a single
t-chain member as the plug-in stand-in. Say which you chose and why. This is
exactly the kind of subtlety Josh probes.

Hints:
1. Evaluate each member's log density on a y-grid (reuse the five-term NLL
   from `batch_nll_mean`, but per grid point, not averaged).
2. Average DENSITIES, not log densities: logsumexp over members minus log M.
   Averaging log densities is the classic mistake; it gives a geometric mean
   that is falsely narrow.
3. Sanity number: per-star NLL of the true y under each curve, printed on the
   panel.

Checkpoints:
- [ ] The mixture is at least as wide as the typical single-member curve,
      never narrower.
- [ ] For the typical star, plug-in and mixture nearly coincide (cloud is
      small). The visible separation happens on the OOD star.
- [ ] Averaged over a few hundred test stars, mixture NLL <= plug-in NLL.
      Direction must match the campaign result (chain beat MAP and ensemble
      on test NLL).
- [ ] logsumexp check: with M identical members the mixture must equal the
      single curve exactly.

## Toolbox: every function you need, by task

Setup (one cell): numpy, torch, matplotlib.pyplot, scipy.stats.t / scipy.stats.norm,
scipy.special.logsumexp. Import the drivers as libraries:
sys.path.insert(0, "experiments"); import exp7_gaia_t as T; import
exp7_gaia_hetero as H. Kernel = empirical/.venv (homebrew python has no torch).

Driver API (never rebuild or hand-slice):
- T.make_model()                  the TModel
- T.load_flat(model, member_row)  loads one flat vector, handles [w, net] order
- model.mu_r(X) -> (mu, r);  sigma(x) = np.exp(T.LNS0 + r)
- model.nu()                      scalar nu of the loaded member
- H.make_model(), H.load_flat, H.LNS0   Gaussian versions (no nu)
- exp7_map.npz already stores per-test-star mu and sig; no MAP model needed.

Task A: torch.no_grad() around all forwards; loop members -> arrays shaped
(50, N) for mu and sigma, (50,) for nu; .cpu().numpy() to leave the device.
Star picking: .std(axis=0), np.argmax, np.median, np.quantile, np.argsort.
Plot: plt.subplots(1, 3); ax.scatter(mu, sigma, c=nu, cmap=...);
fig.colorbar; ax.axvline for catalog y; marker="X" scatter for MAP.

Task B: np.linspace for the y grid. scipy.stats.t.logpdf(y, df, loc, scale)
IS the five-term batch_nll_mean formula (validated to 1e-8 vs the code);
norm.logpdf for Gaussian and MAP. Scale = np.sqrt(yerr**2 + sigma**2), total,
not sigma alone. Broadcast df as (50, 1) and loc/scale as (50, 1) or (50, N)
to evaluate all members at once. Mixture = logsumexp(lp, axis=0) - np.log(50):
averages densities safely. lp.mean(axis=0) is the WRONG geometric-mean
version. Unit check: 50 identical members must reproduce the single curve.

## Say-it-yourself questions (answer aloud before the next meeting)

1. What exactly is one dot in the Task A scatterplot? (One posterior draw's
   belief about this one star.)
2. Which uncertainty shrinks if Gaia doubles the training set, the cloud or
   the curve width? Why?
3. Why is nu one number per draw but sigma one number per star per draw?
4. Is the RTS marginal still a Student-t? What could make it skewed?
5. What does the MAP curve get wrong even when its mu is perfect?
6. Why average densities and not log densities?
