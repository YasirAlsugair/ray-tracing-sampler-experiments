# Tuesday prep: the per-star predictive (Josh's whiteboard task)

Figures: plots/exp7_predictive/. Script: experiments/exp7_predictive_figures.py.
Study this sheet, then close it and explain each figure out loud. The words
must be yours; these notes give you the understanding, not a script.

## The one equation (know it cold)

For one star x, with M = 50 posterior draws from the chain:

    p(y | x, D) = integral of p(y | mu, sigma, nu) p(mu, sigma, nu | x, D)
                ~= (1/50) sum_m t_{nu_m}( y ; mu_m(x), s_m(x) ),
    s_m^2 = catalog err^2 + sigma_m(x)^2.

First factor: the noise model, one bell curve given one draw's triple.
Second factor: the cloud, the posterior over the triple that the 11,395-dim
weight posterior induces for this star. Marginalizing = averaging the 50
curves. MAP = keeping one curve and discarding the cloud.

## The two uncertainties (the concept Josh probed)

- Aleatoric = sigma_m(x): the scatter the model believes is IN the data.
  Lives inside each curve's width. More samples never shrink it.
- Epistemic = the spread of the cloud: how unsure we are about the function
  itself. More DATA shrinks it; more samples only measure it better.
- MAP carries only the first. The marginal carries both, and is wider exactly
  where the draws disagree.

## Numbers to have ready (all recomputed from the packs, cross-checked)

- Test-set NLL, 25,232 stars: MAP plug-in -1.733, Gaussian marginal -1.941,
  Student-t marginal -1.973. The -1.941 and -1.973 reproduce the certified
  campaign numbers exactly, which validates the mixture code.
- So marginalization buys +0.21 nats/star; the t upgrade buys +0.03 more.
  Averaging over the cloud matters about 7x more than fattening the tails.
- mu-cloud spread: median 0.0034 (small; median spread/scale ratio 0.13).
  But the 99th percentile ratio is above 1: about 1% of stars are
  epistemic-dominated, where the cloud exceeds the noise scale.
- Star 3717 (largest disagreement): mu spread 0.21 = 6x its noise scale;
  the marginal is visibly multimodal; MAP is one confident spike.
- nu across draws: 5.37 to 5.61 (the certified interval was 5.48 [5.43,5.53]).

## Answers to the six workbook questions

1. One dot in the scatter = one posterior draw's belief about this one star:
   its predicted center mu, its believed scatter sigma, its tail weight nu.
2. Doubling the data shrinks the CLOUD (epistemic), not the curve widths
   (aleatoric): sigma(x) is a property the model attributes to the sky,
   the cloud is a property of our ignorance about the model.
3. nu is one global parameter of the likelihood (one number per draw);
   sigma has a per-star network head, so it is one number per star per draw.
4. No, the marginal is a mixture of t's, not a t. It can be skewed or
   multimodal if the mu cloud is lopsided (star 3717 shows exactly that).
5. Even with a perfect mu, MAP reports only aleatoric width. It is silently
   overconfident wherever the posterior disagrees, and it has no mechanism
   to know where that is.
6. Average densities, not log densities: mean of logs is the geometric mean,
   which is falsely narrow. In code: logsumexp(lp, axis=0) - log(M).

## Questions Josh may ask, and honest answers

- "Why does the marginal LOSE on the disagreement star (+3.3 vs +1.3)?"
  Because it honestly spreads mass over the disagreement, and the catalog
  value happened to land near the MAP spike. Marginalization wins on
  average (+0.21 nats/star), not pointwise. Saying this unprompted is the
  credibility move.
- "Is that MAP comparison fair?" Partly: the stored MAP was fit to the
  GAUSSIAN target, so the clean pair is MAP vs Gaussian marginal (+0.21);
  the t marginal adds +0.03 on top. Both gaps are shown separately.
- "Is the cloud the full epistemic uncertainty?" It is a lower bound: 50
  members from one chain window. Between-window drift and multi-chain
  variation would only widen it.
- "Why does the MAP point sit outside the typical star's cloud?" MAP is the
  mode of an optimization run, not a posterior draw; its sigma is also
  miscalibrated (it failed the z-structure test, SD(z) 1.18, 209 tails).
- "What would you do next?" Flag the epistemic-dominated 1% (ratio > 1) as
  an uncertainty-aware OOD signal; compare against the flagged/imposter
  panels.

## Self-test (do this Monday night, no notes)

Draw the whiteboard from memory: data, the M-row table, the 1-star scatter,
the two nested bell curves, the factorized formula, MAP vs marginal. Then
say the one equation and the three test-set NLLs. If you can, you are ready.
