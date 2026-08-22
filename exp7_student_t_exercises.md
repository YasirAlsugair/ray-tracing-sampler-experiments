# Student-t paper exercises

Do everything by hand (calculator allowed for exp, ln, powers). Send me photos
and I grade. Check values for the numeric ones are at the bottom; look only
after you commit to an answer.

Given lookups so you never need to compute gamma functions:
Gamma(3.25) = 2.549, Gamma(2.75) = 1.608, sqrt(5.5 * pi) = 4.157.
Tail factor table for nu = 5.5, i.e. (1 + z^2/5.5)^(-3.25):
z = 0: 1.000   z = 1: 0.581   z = 2: 0.169   z = 3: 0.043   z = 4: 0.012

## Part 1: the formula (from memory, then check against the sheet)

1. Write the Student-t density t(y; mu, s, nu) in full, including the
   normalizing constant. Label the three parts: standardize, tail, normalize.
2. Write the five-term NLL (the batch_nll_mean expression). Circle every
   term that contains nu.
3. Write s^2 = (catalog error)^2 + sigma(x)^2 and say in one sentence what
   each of the two pieces is.
4. Write nu = 2 + exp(1.6 + theta_nu). One sentence each: what the exp does,
   what the +2 does, what the 1.6 does.

## Part 2: numbers

5. Compute the density at y = 0.10 for mu = 0.096, s = 0.020, nu = 5.5.
   Steps: z, then tail factor (interpolate the table or compute), then the
   constant C = Gamma(3.25) / (Gamma(2.75) * 4.157 * s), then C * tail.
6. Outlier cost. At z = 4 with s = 1: t density = C * 0.012 with C = 0.3813.
   Gaussian density at z = 4 is 0.000134. What is the ratio t/Gaussian?
   How many nats cheaper is one z = 4 outlier under the t? (nats = ln ratio)
7. Gaussian limit. At z = 2 the Gaussian tail factor e^(-z^2/2) = 0.135.
   The nu = 5.5 table says 0.169. Compute the nu = 100 tail factor
   (1 + 4/100)^(-50.5) and confirm it moves toward 0.135.
8. Variance. For nu = 5.5, variance = nu/(nu-2) * s^2. By what factor is the
   t's standard deviation bigger than its scale s?
9. Kurtosis check (how nu was predicted before sampling). Excess kurtosis of
   a t is 6/(nu-4). The Gaussian chain's residuals had total kurtosis 8.5
   (excess 5.5). Solve for nu. Compare with the sampled 5.48.
10. theta_nu to nu. Compute nu for theta_nu = -1, 0, +1. Which one is the
    prior center?

## Part 3: the marginal by hand

Three posterior draws for one star, all with s = 0.05, nu = 5.5:
draw A: mu = 0.10,  draw B: mu = 0.20,  draw C: mu = 0.60.
The constant is C = 0.3813 / 0.05 = 7.626.

11. Build the 3 x 3 table of densities at y = 0.10, 0.20, 0.60.
    (Each cell: z = (y - mu)/0.05, tail from the table, times 7.626.)
12. Average each column. Write the three marginal values.
13. Sketch the marginal curve by hand from those three points plus what you
    know happens in between. How many bumps and why?
14. Probability question: from your marginal value at y = 0.60, what is the
    probability the star lies in [0.575, 0.625]? Why is it about one third
    of what draw C alone would say?
15. The trap: compute the average mu (one number) and sketch the single t
    curve centered there on the same sketch. Mark where it puts confident
    mass that no draw believes.

## Part 4: concepts, one or two sentences each

16. Write the exact marginal (integral) and the Monte Carlo estimate (sum)
    side by side. Index on the right letter. No double weighting.
17. Why are the 50 weights equal in the sum?
18. nu is global. What does global mean here, and why do the 50 members
    still show 50 different values?
19. Where does nu live relative to the MLP, and how does it get a gradient?
20. The same likelihood formula is used twice in this project with two
    different readings. Name both.

## Check values (numeric exercises only)

5: z = 0.2, tail = 0.977, C = 19.07, density = 18.6
6: ratio = 34, about 3.5 nats cheaper
7: 0.138
8: factor sqrt(5.5/3.5) = 1.25
9: nu = 4 + 6/5.5 = 5.09
10: nu = 4.20, 6.95, 15.5; theta_nu = 0 is the prior center
12: y = 0.10 column: (7.626 + 1.289 + about 0)/3 = 2.97; the other two
    columns are yours to verify against me
14: about 9 to 10 percent, and one third because only draw C puts mass there
