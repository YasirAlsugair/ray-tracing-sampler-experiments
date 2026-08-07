# exp7 vs the BDL evaluation literature

Positioning the RTS Gaia campaign against the field's reference protocol
(Izmailov et al. 2021, "What Are Bayesian Neural Network Posteriors Really
Like?") and the 2026 position paper's asks. Written 2026-08-07; every exp7
number below is reproducible from the committed packs.

## Side by side

| Dimension | Izmailov et al. 2021 | 2026 position paper | exp7 (RTS) |
|---|---|---|---|
| Targets | UCI toys (1x50 MLP), CIFAR-10/100 (ResNet-20-FRN), IMDB | ResNet-18 11M, ViT 22M, LLM adapters 218M | Real survey regression: Gaia XP to [alpha/M], MLP 110-64-64-2, D 11.4k, 126k stars with real per-star errors |
| Noise model | Homoscedastic / classification | not its focus | Nested ladder: err-only, global s, per-star sigma(x), Student-t with sampled nu. Per-datum noise is absent from their protocols entirely |
| Ground truth | Full-batch HMC (240 TPU-days) | none proposed | Exact full-batch Metropolis finisher (acc 0.98, ~30 min); affordable at this scale |
| Predictive metrics | acc, LL, ECE; RMSE + test LL on UCI | LPPD/NLL, calibration error | test RMSE + test NLL, mixture predictive: matches the UCI protocol |
| Calibration | ECE (classification) | calibration error | Finer for regression: z std per label-error bin, PIT coverage, tail counts. Per-group structure is the exp7 headline |
| OOD | AUC-ROC, CIFAR to SVHN/CIFAR-100, vs ensembles/ODIN/Mahalanobis | not central | AUC-ROC of the tau alarm (chain of record): dwarfs 0.932, hot 0.998, survey-flagged 0.588 (near chance, a label-side problem no spectrum alarm can see) |
| Distribution shift | CIFAR-10-C, 16 corruptions x 5 | yes | No direct analog. Candidate: bin test stars by SNR or extinction as natural shift |
| Convergence diagnostics | R-hat in weight AND function space (multi-chain), BMA vs burn-in, 2D cross-sections | says parameter-space diagnostics uninformative | Drift rule on 3-4 observables read WITH acceptance, exact-finisher stamp, gate noise-ceiling analysis, kinetic thermometer (for SGHMC), IAT by observable. Single chains, so NO R-hat: the honest gap |
| Weight vs function space | Their headline: chains differ in weights, agree in function | warns function metrics can mislead | Analog finding: k-NN neighbor lists in the learned representation are posterior-invariant (Jaccard 0.90) across members |
| Baselines | SGD, deep ensembles, SGLD, MFVI, SWAG, subspace, DVI | ensembles, SWA, Laplace, MFVI | MAP (=SGD+prior), deep ensembles, SGHMC 2D-tuned with the grid on record. No MFVI/SWAG/Laplace yet |
| Compute honesty | compute-matched ablations | yes | Steps = gradient rows at equal batch; SGHMC arm matched; tuning grids recorded |

## Where exp7 already meets or exceeds the protocol

1. The predictive pair (RMSE + test NLL) matches the UCI convention and the
   position paper's LPPD ask.
2. Regression calibration is reported at a finer grain than ECE: per-group
   z std plus PIT coverage, with the noise model itself part of the study.
   No paper in their table varies the likelihood as a treatment.
3. OOD is now in their currency (AUC-ROC) and competitive: 0.93 to 1.00 for
   genuine covariate shift.
4. Exactness certificates they lack: the full-batch finisher, the gate's
   measured noise ceiling, and the kinetic thermometer are diagnostics the
   reference protocol has no analog for. The position paper explicitly says
   a tailored BNN evaluation framework is "critically needed" and declines
   to propose one. The exp7 certificate stack (drift + acceptance +
   exact-finisher + thermometer + noise ceiling + tuning-grid-on-record) is
   a concrete candidate. That is the open door the notes mention.

## Honest gaps, with effort estimates

1. Multi-chain R-hat (weight and function space), their central diagnostic.
   Needs 2-4 replicate chains from dispersed starts. At the measured ~150
   steps/s per chain this is a few hours and a few dollars per replicate.
   The natural next run if literature comparability matters.
2. BMA-performance vs burn-in curve: computable from existing snapshots,
   free, an afternoon.
3. Scale: their smallest image model (274k) is 24x our D; the position
   paper wants 11M+. The RTS paper itself reports GPT-2 scale, but OUR
   harness has only run D 11.4k. A ResNet-18 target would be a new
   campaign.
4. Missing baselines: MFVI, SWAG, Laplace. SWAG is the cheapest (it reuses
   the MAP trajectory); MFVI is a moderate lift.
5. Distribution shift: no CIFAR-10-C analog yet; the SNR-binning idea is
   unexplored.

## One-line positioning for the poster or a paper intro

On a real heteroscedastic scientific target, exp7 reports the reference
protocol's predictive and OOD metrics, exceeds its calibration granularity,
and adds exactness certificates the protocol lacks; it gives up multi-chain
R-hat and large-model scale, both stated openly.
