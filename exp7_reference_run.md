# exp7 reference run (2026-07-26/27, overnight)

Minibatch ray tracing on the Gaia [alpha/M] regression posterior, RunPod
5090s, about 6 GPU hours and $6 across two pods (both stopped). This is the
reference for the exp7 workbook; do the workbook by hand, then compare.
The presentation notebook is exp7_gaia_posterior.ipynb (fully executed).

## Setup

Sample: the paper's pristine cut reproduced at 126,156 stars (paper
123,804, within 1.9 percent; the "good" stage lands 263,722 vs 202,970
because this table's error columns are not exactly the astroNN ones). Split
seed 2003, 80/20 = 100,924 / 25,232, X standardized on train. Payload:
results/tables/exp7_gaia_pristine.npz. Cut script:
experiments/exp7_gaia_prep.py (workbook section 0 answer key, do not peek
early).

Target: Gaussian NLL with sigma = ALPHA_M_ERR per star, T = 1. Model: MLP
110-64-64-1 tanh, D = 11,329. Priors: Glorot per layer (bias sigma 0.1),
and N(0,1) for the head-to-head. Sampler: vendor minibatch Raytracer,
batch 1024, Eq. 33 gate every 30 steps. Driver:
experiments/exp7_gaia_driver.py.

## Scorecard (held-out 25,232 stars)

    RMSE   global mean            0.0927
    RMSE   kNN (k=10)             0.0742
    RMSE   N(0,1) chain           0.0704   (500k steps, prior ablation)
    RMSE   THE CHAIN (gated 5e-6) 0.0492   (2M steps, Eq. 33 on, acc 0.38)
    z std: chain 6.2 (spread is a lower bound), N(0,1) 7.8  (1.0 = calibrated)
    member spread (median): chain 0.0020 (lower bound), vs label err 0.0064

Validation: an unadjusted control arm (dt 1e-4, 500k steps, local file
exp7_rtub_dt0.0001.npz, not in the analysis pack) reproduces the chain's
predictions to one percent (RMSE 0.0486) with ~4x larger member spreads
(z std 4.4), so the gate changed the cost, not the answer. The chain
beats k-NN by a third. The uncertainties are at least 4x too small:
residuals ~0.049 vs claimed ~0.010. An intrinsic scatter s ~ 0.05 dex
(workbook section 2, choice 2) is the obvious next run.

## The findings

1. The prior decides what the budget buys. Glorot: born on its shell
   (weighted norm 11,235 of D = 11,329), whole budget spent fitting, RMSE
   0.049. N(0,1), same sampler and budget: starts 99 percent saturated,
   ends at 2.9x the misfit (4.78M vs 1.64M nats), RMSE 0.070, barely at
   k-NN level, z std 7.8, pinned to its own shell (11,233) throughout.
   exp6's 20,000-trajectory walk was a prior problem, not a sampler
   problem.

2. Eq. 33 is a diagnostic, not a filter, on this heteroscedastic target.
   Blind at the prior draw (sigma_sto 26M nats for Glorot, 3.5B for
   N(0,1); every dt accepts ~1.0, so the tuning ladder must run from a
   FITTED state). At the fit (sigma_sto 135k) it correctly all-rejects
   the dt 1e-4 dynamics: the luminosity ledger claims ~10M spurious nats
   per 30-step window. Tuned by the paper's recipe the ladder reads
   0.04 / 0.20 / 0.36 / 0.38 / 0.38 for dt 2e-5 to 1e-6: recovery right
   where the dt^2 bias scaling predicts, but the plateau 0.38 is the
   gate's own noise ceiling, so the tuned gate's decisions are coin
   flips. The 2M-step production arm at dt 5e-6 held acceptance 0.38,
   still crept in norm (19.2k to 24.4k), and collapsed the member spread
   to 0.002. Echoes Behroozi dropping MH at GPT-2 scale, now with a
   mechanism. Never read a drift rule without acceptance next to it: a
   frozen all-reject chain passes any stationarity test.

3. The chain of record keeps the gate (per Yasir's call): its decisions
   at the tuned dt are noise, but the agreement with the unadjusted
   control (0.0492 vs 0.0486) is itself evidence the knob-controlled
   dynamics were clean. The cost is a 20x smaller step, so the member
   spread is a lower bound; the misfit was also still drifting slowly at
   the end, so posterior spreads are provisional twice over.

4. GaiaXpy: coefficients are stored BP-first (swapped halves produce a
   seam at 640 nm). At fixed Teff and [M/H], high-alpha giants are 20-30
   percent brighter below 450 nm (shape-normalized stacks), and the
   network's saliency, obtained by calibrating the input gradient
   (calibration is linear in coefficients), concentrates in the same blue
   region. Data and model agree where alpha lives.

## Files

    exp7_gaia_posterior.ipynb                     the presentation notebook
    results/tables/exp7_rt33_dt0.0001.npz         Glorot gated base arm, 500k
    results/tables/exp7_rt33_dt0.0001_part2.npz   frozen all-reject leg
    results/tables/exp7_rtub_dt0.0001.npz         Glorot ungated control, 500k
    results/tables/exp7_rt33_dt5e-06.npz          Glorot gated production, 2M
    results/tables/exp7n_rt33_dt0.0001.npz        N(0,1) chain, 500k
    results/tables/exp7_gaia_spectra_payload.npz  test-split coeffs for GaiaXpy

## The intrinsic-scatter run (2026-07-27 evening, pod exp7-scatter)

The corrected noise model, var_i = err_i^2 + s^2 with ln s = -3 + u
sampled (u ~ N(0,1) prior, s starts at 0.0498), D = 11,330. Intended to continue
from the gated chain's endpoint; a parameter-ordering bug (u registers
BEFORE the network weights in the scatter model, caught post hoc when the
exact finisher rejected everything at -3.3M nats) scrambled the warm start
into a same-scale shuffle. The chain repaired it within 100k steps, an
accidental robustness test, and the converged endpoint is unaffected: the
run's own save/load are self-consistent. Driver:
experiments/exp7_gaia_scatter.py (log_prob layout fixed in the same file).
Tuned on the corrected target from the seed state:

    acceptance ladder  dt 1e-4: 0.83   5e-5: 0.90   2e-5: 0.94   1e-5: 0.92
    chosen dt 2e-5 (4x the old step), production 2M steps, Eq. 33 on

Three predictions were registered before the run; the scoreboard
(CORRECTED 2026-07-30, see the sigma_sto note below):

    sigma_sto collapses      HIT       3,337 nats at the true fitted
                                       endpoint vs 135k on the first
                                       likelihood, a ~40x collapse. The
                                       162k originally recorded as a
                                       refutation was measured at the
                                       bug-scrambled warm start.
    s lands near 0.049       NEAR HIT  s = 0.0452 +- 0.0010; the shift is
                                       explained: s equalizes star weights,
                                       the fit improved, residuals shrank
    z std lands near 1       HIT       z std 0.948, calibrated

sigma_sto correction (2026-07-30): the production run measured sigma_sto
once, at its starting state, which the parameter-ordering bug had
scrambled; a garbage network's residual tails inflate the batch variance,
hence 162k. Re-measured at the true fitted endpoint the corrected model
gives 3,337 nats. Two consequences. The "tail stars own sigma_sto" story
was an artifact of the scrambled state (the |z| > 4 tail is real, 41
stars, but it is not what set sigma_sto). And the scatter production's
Eq. 33 gate ran with softening 1/162k, about 49x too lenient, so its 0.99
acceptance and the revival ladder overstate the gate's health; the
chain's validity rests on the exact full-batch finisher (accepts the
endpoint at 0.98) and the drift checks, which are unaffected. Caught by
the heteroscedastic run's smoke test measuring sigma_sto at a clean
warm start.

And the run delivered more than predicted. RMSE 0.0466, the campaign
best (equal weighting freed the fit from the tiny-error stars). The
weighted norm fell back from 24.4k to 15.0k against the shell's 11.3k:
the corrected likelihood no longer demands oversized weights. Member
spread recovered to 0.0120 (real mixing at the 4x step). Acceptance held
at 0.99 with the gate on throughout. And ALL THREE drift checks pass
(misfit +90 vs noise 2517, wnorm -13 vs 304, s +7e-5 vs 1e-3) with
healthy acceptance: the first legitimately converged chain of the
campaign, 2M steps in 7.7 h on a slow-CPU host. THE CHAIN OF RECORD is
now this one. Chain file: results/tables/exp7s_rt33_dt2e-05.npz
(gitignored like the others).

## The exact full-batch finisher (2026-07-28, same pod)

A noiseless finisher launched from the converged endpoint: full-batch
gradients, exact Metropolis test every trajectory, no Eq. 33
approximation. Pilot found a cliff, not a slope: every dt from 2e-4 down
to 5e-6 rejects at 0.00, and 2e-6 accepts at 1.00, so the exact test
forces a step 10x smaller than the minibatch production step. Production:
5,000 trajectories, L = 30, 14.9 minutes, acceptance 0.98.

    RMSE 0.0469 (record: 0.0466)   z std 1.055   s median 0.0436

The verdict has two parts. First, the full-batch posterior ACCEPTS the
minibatch chain's endpoint at 0.98 and reproduces its predictions, so
the converged state is not a minibatch artifact. Second, the stamp is on
the location, not the spread: each trajectory covers a path of only
6e-5, the exact chain barely moves (member spread median 0.0007, s
spread 0.0000, both lower bounds by construction), and ln_post was still
climbing at the end (160,871 to 165,061, drift +58 vs noise 17). Spreads
still come from the chain of record (tau 0.012). Exactness at D = 11,330
cost a 10x smaller step, which is the scaling argument for minibatch
gates at scale, now measured on our own target. Chain file:
results/tables/exp7s_exact_dt2e-06.npz (gitignored).

## The heteroscedastic run (2026-07-30/31, pods exp7-hetero / exp7-hetero-3)

Motivated by the z-binning of the scatter chain: z std runs 0.39 to 1.43
across label-error bins (0.64 to 1.22 across [M/H]), so one global s
calibrates the average, not the structure. The model gains a second
output channel: sigma(x) = exp(-3 + r(x)), var_i = err_i^2 + sigma(x)^2,
D = 11,394, warm-started exactly at the scatter solution (mu head
copied, sigma head flat at the endpoint's s; verified bit-identical).
Driver: experiments/exp7_gaia_hetero.py.

With the honest gate softening (sigma_sto 3,337 nats at the clean seed,
see the correction above) the tuning ladder reads 0.04 / 0.23 / 0.39 /
0.45 / 0.45 for dt 1e-4 down to 5e-6: a plateau at a NEW noise ceiling
of 0.45, knee at dt 1e-5. So the scatter run's "gate revival"
(0.83-0.94) was mostly the over-softening artifact; honestly calibrated,
the corrected-target gate behaves like the first-likelihood one with a
higher ceiling and a 2x larger knee.

Production 2M steps at dt 1e-5 plus three 500k convergence legs
(3.5M total; the first pod died to a depleted balance mid-converge and
the run was reproduced from its deterministic seeds):

    RMSE 0.0473 (scatter 0.0466)   z std 1.060   acceptance ~0.5
    sigma(x): median 0.035, 16/84 pct [0.016, 0.061], max 0.48
    z std by label-error bin: 0.95 / 1.04 / 1.06 / 1.10
      (was 0.39 / 0.66 / 1.04 / 1.43 under the global s)

The calibration STRUCTURE is fixed: near-flat z std across the bins
where the global s failed, at a cost of 0.0007 in RMSE (a model allowed
to call stars noisy stops over-fitting them).

Convergence was a campaign of its own (2026-08-03/04). At batch 1024
the norm marched indefinitely (+600/leg through nine legs, no
deceleration). Escalating to batch 4096 at the same dt halved sigma_sto
(3,743 to ~1,940 nats), lifted acceptance to ~0.51, and did NOT slow
the march at first, which ruled out gradient-noise heating: the chain
was genuinely still descending, and with cleaner gradients it descended
faster. Over 19 further 500k legs (16M steps total) the march
decelerated (+1,773 to +947 to +542 per leg), plateaued, and reversed;
the drift rule then declared ALL THREE series level (misfit +843 vs
noise 3,242; wnorm -1,563 vs 1,316; sig_med level). CONVERGED, at 16M
steps and roughly $45 of pod time end to end.

Converged-chain numbers (50 members from the final quarter,
exp7h_pack3.npz, verified locally): RMSE 0.0494, z std 1.092, bins
0.99 / 1.09 / 1.09 / 1.07, sigma(x) median 0.0322 [16/84: 0.0141,
0.0620], tau 0.0049, tail 118 stars past |z| of 4, z kurtosis 8.5.
Note the honest RMSE cost of full convergence: 0.0466 (scatter) to
0.0494; as the chain settled into the posterior bulk at honest
per-star weights, the point accuracy relaxed to the intrinsic ~0.049
residual floor. Calibration flatness held throughout every snapshot.

CHAIN OF RECORD: the hetero chain, now that both chains pass all
drift checks (the both-must-converge rule). Calibration is the purpose
of the noise model and only the hetero chain keeps its per-star
promises. The scatter chain remains the best point predictor (RMSE
0.0466) and the global-s special case for comparisons. The Student-t
case stands at kurtosis 8.5.

The tail case for a Student-t likelihood STRENGTHENED as sigma(x)
sharpened: z kurtosis 8.1 (Gaussian 3), 127 stars beyond |z| of 4, most
of them modest misses on stars the model now declares precise. sigma(x)
does absorb the OLD tail (29 of the 41 global-s tail stars rescued,
their sigma(x) at the 91st percentile), so the remaining tail is a
shape problem, not a scale problem: the motivated rung three is
Student-t with scale sigma(x), a decision for Yasir and Josh.

Artifacts: results/tables/exp7h_pack.npz (3.5M-step members) and
exp7h_pack2.npz (6.5M-step members, traces, final state), pod logs
exp7h_pod.log / exp7legs.log; raw chain files reproducible from seeds.

## Open items

- Intrinsic scatter s as an 11,330th parameter, then recheck z std.
  DONE: the scatter run above; z std 0.948.
- Exact-chain finisher. DONE: stamps the location (acc 0.98, RMSE
  reproduced); spreads still quoted from the chain of record.
- Early NaNs in the window ledger at the saturated start (auto-accepted by
  an already-blind gate); harmless here, worth a guard.
- Heavy-tailed likelihood for the |z| > 4 tail stars (they own sigma_sto);
  a modeling decision for Yasir and Josh, out of scope for the reference
  run.
