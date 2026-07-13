# Step 2: from training loss to log-posterior

Setup: MNIST training set D = {(x_i, y_i)}, i = 1..N with N = 60,000. The network
defines a categorical likelihood per image, p(y | x, theta) = softmax(f_theta(x))_y.
Training minimizes the mean cross-entropy in nats (torch `cross_entropy` uses ln):

    CE(theta) = -(1/N) sum_i ln p(y_i | x_i, theta)

## The identity

Because the labels are discrete, the log-likelihood needs no additive constant
(unlike regression, where the Gaussian noise term contributes one):

    ln L(theta) = sum_i ln p(y_i | x_i, theta) = -N * CE(theta)     (exact)

So the training loss IS the log-likelihood, up to the factor -N. The temperature of
the posterior is not a free dial: sampling at "SCALE = N" is the Bayes posterior,
anything else is a tempered variant. (The shemagh run used the paper's heuristic
SCALE = N / (2 * 0.1); from now on SCALE = N = 60,000 is derived, not tuned.)

## Adding the prior

    ln p(theta | D) = ln L(theta) + ln p(theta) + const(D)

- **N(0, 1) prior on every weight:** ln p(theta) = -||theta||^2 / 2 + const.
  MAP for this posterior is exactly cross-entropy + L2 with coefficient 1/N in
  mean-loss units, i.e. ordinary weight decay of 1.67e-5.
- **Flat prior p(theta) = const:** ln posterior = -N * CE + const. Improper but
  usable locally; the run without weight decay targets this one.

Default for step 4 is the N(0, 1) prior (proper posterior, standard correspondence);
flat is the ablation.

## What the sampler sees

The ray tracing target density is the posterior itself, L_rts(theta) = p(theta | D),
so the refractive index and kick are

    n(theta) = p(theta | D)^(1/(D-1))
    grad ln n = [ -N * grad CE(theta) - theta ] / (D - 1)      (N(0,1) prior)

The Metropolis test (step 4) uses full-data Delta ln posterior, which is affordable
here because the models are small. A minibatch CE gradient times N stays an unbiased
estimate of grad ln L, which is the stochastic-gradient story from the earlier
experiments; with these models we do not need it.

## Concrete numbers at the step-1 point estimates

| model | D | N * CE (train) | ln L at point | prior term -||theta||^2/2 |
|---|---|---|---|---|
| MLP 784-64-10 | 50,890 | 0.0664 * 60k | -3,982 nats | -87 nats |
| CNN 16-32-fc | 12,810 | 0.0396 * 60k | -2,377 nats | -40 nats |

The likelihood term dominates the prior by ~46x (MLP) and ~59x (CNN) at the trained
point, so the N(0,1) prior is a gentle regularizer here, not a straitjacket. That is
the regime we want: the posterior geometry is set by the data.
