# Poster walkthrough (SUDS Showcase, Aug 14)

The whole poster is one story. If you remember nothing else, remember this line:

> Training gives one answer. Sampling gives all the answers the data allows.
> Cheap gradients are noisy, and noise heats HMC but only turns a ray.
> That is why we could sample an 11,000-parameter network on real Gaia data.

---

## The 30-second pitch (for anyone who stops)

Point at the fan of curves (top left), then the circle diagram (middle), then the two clouds (top right).

"When you train a neural network you get one set of weights, one guess. It cannot
tell you when it is unsure. Bayesian sampling gives you many networks instead of
one, so you get error bars. The problem is that sampling needs gradients, and
cheap minibatch gradients are noisy. Noise makes the standard sampler, HMC, speed
up and fly off course, like heating a gas. The ray tracing sampler moves at a
fixed speed, so noise can only turn it, never heat it. We used it to sample an
11,000-parameter network on 126,000 Gaia stars with minibatches. The draws give
honest error bars, and averaging them beat the single best fit by 27 percent on
held-out stars."

If they nod along, offer the full tour. If they are technical, jump to the
mechanism block (middle column).

---

## The full tour (3-4 minutes, left to right)

### Block 1: A trained network is a single guess
What to point at: the gold fan of curves, the single blue line, the NO DATA bands.

Say: "The blue line is a normally trained network, one function. The gold curves
are draws from the posterior over weights. Where there is data they all agree.
In the gaps they fan out. That fan is the uncertainty a single network can never
give you."

The equation: posterior = fit times prior. Do not linger, just name it.

Transition: "So we want to sample this distribution over weights. In high
dimensions it lives in a strange place."

### Block 2: Probability lives on a thin shell
What to point at: the density, volume, and mass curves, then the D = 100 bump.

Say: "Density is highest at the center, but there is almost no volume there.
Volume grows fast with radius. The product, the actual probability mass, piles
up on a thin shell at radius about square root of D. A correct sampler spends
almost all its time on this shell, not at the peak."

This block earns its place later: heating in block 4 means "thrown off the
shell." Plant that now: "Keep this shell in mind."

### Block 3: Minibatch gradients are cheap but noisy
What to point at: the arrow fan (exact gradient vs minibatch arrows), then the
angle chart.

Say: "Samplers use the gradient of log probability to know where to go. The
exact gradient touches all N data points. A minibatch is N over B times cheaper
and right on average, but any single one is very noisy. On this test problem,
even a batch of 2,048 points is off by 63 degrees on average. 90 degrees would
be a random direction. So the question is: what does a sampler do with
directions that are almost random?"

Transition: "This is the whole poster in one block." (point at block 4)

### Block 4: Noise heats HMC but only turns a ray
What to point at: the three circle diagrams, left to right.

Say, in three beats:

1. One step: "HMC carries a velocity. A noise kick adds to the velocity, and on
   average kicks make it faster. That extra speed is heat. The ray carries a
   direction of length one. A kick gets projected onto the sphere, so it can
   only rotate the direction. The speed cannot change, by construction."
2. Many steps: "Over a trajectory the kicks accumulate. For HMC the heat builds
   up and the sampler leaves the shell from block 2."
3. Long run: "For the ray, random rotations do not prefer any direction, so the
   leading noise effect creates no bias at all. The math on the poster says: if
   you halve the step size, HMC's noise tolerance grows by root 2, but the
   ray's tolerance doubles."

The QR code: "There is a live demo of this, you can run it in your browser."

### Block 5: Measured noise tolerance and sampling cost
What to point at: panel (a) slopes, then panel (c).

Say: "We measured the tolerance, not just derived it. Panel (a): across
dimensions 16 to 1024, the slopes match the theory, minus one for ray tracing,
minus one half for HMC. Panel (b) is the honest part: with real minibatch noise
instead of idealized noise, ray tracing's exponent weakens, to around 0.6. It
keeps its advantage but the clean scaling degrades. Panel (c) is the cost:
dataset passes per effective sample. Minibatch HMC needs around 30 passes.
Ray tracing with minibatches sits below the full-batch line."

If someone asks for a single speedup number, see the Q&A section. Do not quote
3.3x.

### Block 6: Empirical study on real Gaia data
What to point at: the bar chart first, then the two clouds, then the two density
panels.

Say: "Everything so far was the mechanism. Here it is on real data. We predict
each star's alpha abundance, an element ratio, from its Gaia spectrum, 110
numbers per star. The network has 11,394 weights and we sample it with minibatch
ray tracing on 126,000 stars."

Bar chart: "z std asks: are the error bars honest? For each star, take the miss
divided by the claimed error. If the error bars are honest that ratio has spread
one. Above one means overconfident, below one means underconfident. Gray is the
single trained network: overconfident. The gold bars are ray tracing." (Let them
read that gold sits closest to the line; the chart shows it.)

The two clouds: "Each gold dot is one draw of the network predicting the same
star. Left star: 50 dots on top of each other, the posterior agrees, trust it.
Right star: the dots scatter everywhere. The model is guessing, and it tells
you. The X is the single trained network. It gives you one point for both stars,
with no way to know which star is which kind."

The density panels: "Same two stars seen in the prediction itself. On the
typical star the sampled posterior and the single fit agree. On the hard star
the single fit puts almost no probability near the true value, dotted line,
while the posterior spreads out and covers it."

### Block 7: Conclusion
Read the three bullets almost as written. They are the summary:
1. Fixed speed means noise can only change direction. That is the robustness.
2. We took the sampler from its paper to a real survey posterior. The draws are
   the product: averaging beat the single best fit, and the spread warns you
   when the model is guessing.
3. On average, held-out stars were 27 percent more probable under the posterior
   average than under the single best fit, most of it from the averaging alone.

---

## Hard questions, honest answers

**"Is every star 27 percent more probable?"**
No. It is the average over 25,232 held-out stars. About two thirds of stars
improve individually; the median improvement is around 21 percent. The 27 is
the mean of the log score gain, exponentiated.

**"Where does the 27 percent come from?"**
Mostly from averaging: mixing the 50 draws' predictions gives about 23 of the
27 points. The rest comes from heavier tails in the predictive distribution.
That is why the bullet says "most of it from the averaging alone."

**"Are the point predictions better too?"**
No, and we say so. RMSE is a tie (0.0489 for the single fit, 0.0494 for the
chain). The gain is in the error bars and the log score, not the point
prediction. The single fit claims errors that are about 18 percent too small.

**"How much faster is it?"**
Careful here, we removed a big number from the poster on purpose. The honest
answer: "Minibatch gradients are N over B times cheaper per step, and panel (c)
shows ray tracing stays below the full-batch cost line while minibatch HMC blows
up. At matched accuracy in our linear regression test the saving was about 1.5x
at batch 256. Smaller batches look much faster but fail our accuracy check, so
we do not quote them."

**"Doesn't SGHMC already do minibatch MCMC?"**
Yes, and it is on our bar chart as a baseline (purple). SGHMC handles the noise
by adding friction to soak up the injected heat, which needs a noise estimate.
Ray tracing removes the heating mechanism itself: the speed is fixed by
geometry, so there is nothing for the noise to heat.

**"Why is it called ray tracing?"**
The sampler treats the probability density like an optical medium with a
refractive index. Snell's law bends a constant-speed light ray toward high
probability. It comes from the optics literature, and the name stuck.

**"Is it exact?"**
With full-batch gradients and the Metropolis correction, yes, it targets the
posterior exactly. With minibatches every method has some bias. The point of the
theory block is that ray tracing's bias has no leading-order term, so for the
same tolerance it can absorb far more noise as the step shrinks.

**"What is D = 11,394?"**
The number of network weights. It is a small MLP: 110 spectrum coefficients in,
a hidden layer, and two outputs per star, the predicted value and a per-star
scatter.

**"Where do the 50 draws come from?"**
Fifty saved states of one chain after warm-up, spaced out along the run. Each
one is a full set of network weights, so each one is a working network.

**"What is z std exactly?"**
For each star: (prediction minus catalog value) divided by the total claimed
error. If the error bars are honest, that quantity has standard deviation one
in every bin. We bin by the catalog's own label error so easy and hard stars
are judged separately.

**"Why does noise heat HMC? Really?"**
Kinetic energy is quadratic in velocity. Kicks that are zero on average still
increase the average of the square. So mean-zero noise pumps energy in, one
direction only. There is no matching mechanism that removes it, unless you add
friction (that is SGHMC) or a Metropolis test (which needs the full dataset).

**"What breaks? What are the limits?"**
Real minibatch noise is not isotropic, and panel (b) shows the clean scaling
degrades, exponent around 0.6 instead of 1. Also tuning: step size and
trajectory length still need care. And our cost comparison is against
full-batch sampling on one regression problem, not against every method on
every problem.

**"Who did what?"** (if asked)
Joint work with Chuxuan Ai, supervised by Prof. Speagle and Prof. Baptista.
The theory blocks and measured-tolerance study, and the Gaia study, are the
summer project; the sampler itself is from Behroozi's paper, reference 2.

---

## Numbers you can say out loud (all verified against the repo)

- 126,156 giants, 25,232 held-out test stars, 110 XP coefficients, D = 11,394
  weights, 50 draws.
- 27 percent average held-out probability gain; about 23 points of it from
  averaging alone; roughly two thirds of stars improve; median gain about 21
  percent.
- RMSE 0.0489 (single fit) vs 0.0494 (chain): a tie.
- Single fit's z std is about 1.18: claimed errors about 18 percent too small.
- Gradient example: D = 81, N = 17,010; batch 2,048 is off by 63 degrees on
  average.
- Scalings: noise tolerance grows like 1/h for ray tracing, 1/root-h for HMC;
  real minibatch noise weakens ray tracing's exponent to about 0.6.
- Cost: about 1.5x cheaper than full batch at matched accuracy (batch 256, the
  linear regression test). Minibatch HMC needs about 30 dataset passes per
  effective sample.

## Numbers to avoid

- 3.3x (and 5.1x). The cheap configurations behind them fail our own accuracy
  check, and the full-batch reference was assumed, not measured. If pressed,
  give the 1.5x with the caveat above.
- Any claim that the method is the first minibatch sampler. SGHMC is on the
  poster. The claim is the mechanism: no heating, by construction.

---

## Tips for the session

- Start people at the figure nearest to where they are standing, then pull them
  to block 4. Every path through the poster goes through "heats HMC, turns a ray."
- Say "error bars" to general visitors and "calibration" or "log score" to
  technical ones. Same content, different word.
- The two clouds figure is the best single figure for a general audience. The
  circle diagram in block 4 is the best one for a technical audience.
- When someone asks a question you cannot answer, the honest move: "That is a
  good question and I do not want to guess. My read is ... but I would have to
  check." This lands much better than improvising.
