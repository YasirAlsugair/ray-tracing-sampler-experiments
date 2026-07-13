"""Stage-1 analytic ground truth and scorecard for 2D Boltzmann targets.

For a 2D target with known log L = log_prob(x) (= -U/kT) the Boltzmann density
is exactly  p(x) = exp(log_prob(x)) / Z, so we can score a sampler against truth
instead of against another sampler.  Everything here is derived from
`target.log_prob` alone (no separate U), so it is correct for double_well,
muller_brown, or any future 2D `boltzmann2d` target.

What this provides:
  * grid_eval        - normalised analytic cell-probabilities on a grid.
  * find_minima      - basin minima via normalised gradient-ascent flow on log L.
  * basin_labels     - assign every grid cell to a basin by steepest-ascent flow.
  * true_weights     - exact basin occupancy by integrating p over each basin.
  * core_mask        - basin cores (near the bottom) for no-recrossing transitions.
  * density_distances- TV / KL / JS of sampled vs analytic density on a grid.
  * sample_occupancy - sampled basin fractions + autocorrelation-aware CIs.
  * transitions_mfpt - committed (core-to-core) transition counts and MFPT.
  * split_rhat       - Gelman-Rubin split-Rhat for a scalar across chains.

The honest line (mirrors the project spine): the Metropolis test makes the chain
exact for any dynamics, so a density match validates the implementation and the
chain's ergodicity, not asymptotic bias.  Occupancy across overdispersed chains
is therefore the load-bearing test, and every error bar is autocorrelation-aware.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .metrics import tau_sokal


# --- analytic density on a grid -------------------------------------------------
def grid_eval(target, n=240):
    """Evaluate log L on an n x n grid over target.meta['box'].

    Returns xs, ys (axes), lp (n,n log-density), Pc (n,n cell-probabilities that
    sum to 1; equal-area cells so this is also the discretised Boltzmann measure),
    and dx, dy.  Pc is what TV/KL and basin integrals consume.
    """
    (x0, x1), (y0, y1) = target.meta["box"]
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")  # (n, n), row index = y
    pts = jnp.asarray(np.stack([XX.ravel(), YY.ravel()], axis=1))
    lp = np.asarray(jax.vmap(target.log_prob)(pts)).reshape(n, n)
    p_un = np.exp(lp - lp.max())
    Pc = p_un / p_un.sum()
    dx = float((x1 - x0) / (n - 1))
    dy = float((y1 - y0) / (n - 1))
    return {"xs": xs, "ys": ys, "lp": lp, "Pc": Pc, "dx": dx, "dy": dy,
            "box": target.meta["box"], "n": n}


# --- basins via gradient-ascent flow on log L ----------------------------------
def _ascent_flow(target, pts, box, n_steps=600, lr_frac=0.004):
    """Normalised gradient-ascent flow of log L (= steepest descent of U).

    Constant-speed normalised steps avoid blow-up where |grad log L| is huge near
    steep walls.  lr is lr_frac of the box diagonal.  Returns the endpoints.
    """
    (x0, x1), (y0, y1) = box
    diag = float(np.hypot(x1 - x0, y1 - y0))
    lr = lr_frac * diag
    g = jax.vmap(jax.grad(target.log_prob))

    def body(_, X):
        G = g(X)
        norm = jnp.sqrt(jnp.sum(G ** 2, axis=1, keepdims=True))
        step = G / jnp.maximum(norm, 1e-12)
        Xn = X + lr * step
        # keep inside the box (reflect softly by clipping)
        Xn = jnp.clip(Xn, jnp.array([x0, y0]), jnp.array([x1, y1]))
        return Xn

    X = jax.lax.fori_loop(0, n_steps, body, jnp.asarray(pts))
    return np.asarray(X)


def find_minima(target, n_seed=28, tol_frac=0.04, **flow):
    """Locate basin minima (= maxima of log L) by flowing a coarse seed grid and
    greedily clustering the endpoints.  Returns minima (K, 2) sorted deepest
    first (highest log L), and their log L values.
    """
    box = target.meta["box"]
    (x0, x1), (y0, y1) = box
    sx = np.linspace(x0, x1, n_seed)
    sy = np.linspace(y0, y1, n_seed)
    SX, SY = np.meshgrid(sx, sy, indexing="xy")
    seeds = np.stack([SX.ravel(), SY.ravel()], axis=1)
    ends = _ascent_flow(target, seeds, box, **flow)
    lp_ends = np.asarray(jax.vmap(target.log_prob)(jnp.asarray(ends)))

    diag = float(np.hypot(x1 - x0, y1 - y0))
    tol = tol_frac * diag
    order = np.argsort(-lp_ends)  # highest log L first
    centers, cen_lp = [], []
    for i in order:
        p = ends[i]
        if all(np.hypot(*(p - c)) > tol for c in centers):
            centers.append(p)
            cen_lp.append(float(lp_ends[i]))
    centers = np.array(centers)
    cen_lp = np.array(cen_lp)
    return centers, cen_lp


def basin_labels(target, grid, minima, **flow):
    """Label every grid cell by the basin (nearest minimum after ascent flow).
    Returns labels (n, n) int in [0, K).
    """
    xs, ys = grid["xs"], grid["ys"]
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    ends = _ascent_flow(target, pts, grid["box"], **flow)
    d2 = ((ends[:, None, :] - minima[None, :, :]) ** 2).sum(-1)  # (n*n, K)
    return np.argmin(d2, axis=1).reshape(len(ys), len(xs))


def true_weights(grid, labels, K):
    """Exact basin occupancy: integral of p over each basin (sum of cell-probs)."""
    Pc = grid["Pc"]
    w = np.array([Pc[labels == k].sum() for k in range(K)])
    return w / w.sum()


def core_mask(target, grid, labels, minima, core_depth=1.0):
    """Boolean grid of basin cores: cells within `core_depth` (in log-L units,
    i.e. core_depth*kT in energy) of their own basin minimum.  Used so that
    transition counting commits to a basin only at its bottom, not on the barrier.
    """
    lp = grid["lp"]
    lp_min = np.asarray(jax.vmap(target.log_prob)(jnp.asarray(minima)))  # (K,)
    core = np.zeros_like(labels, dtype=bool)
    for k in range(len(minima)):
        core |= (labels == k) & (lp >= lp_min[k] - core_depth)
    return core


# --- mapping samples onto the grid ---------------------------------------------
def _cell_index(samples, grid):
    """Map (N,2) samples to nearest grid node (ix, iy), clipped to the box."""
    xs, ys = grid["xs"], grid["ys"]
    ix = np.clip(np.round((samples[:, 0] - xs[0]) / grid["dx"]).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round((samples[:, 1] - ys[0]) / grid["dy"]).astype(int), 0, len(ys) - 1)
    return ix, iy


def density_distances(samples, target, nbin=60):
    """TV, KL(q||p), and Jensen-Shannon between the sampled density q and the
    analytic Boltzmann p, binned on an nbin x nbin grid.  Coarser than the basin
    grid so cells are populated.  Samples outside the box are dropped (reported).
    """
    (x0, x1), (y0, y1) = target.meta["box"]
    coarse = grid_eval(target, n=nbin)
    p = coarse["Pc"].ravel()  # (nbin*nbin,), sums to 1
    inside = ((samples[:, 0] >= x0) & (samples[:, 0] <= x1) &
              (samples[:, 1] >= y0) & (samples[:, 1] <= y1))
    s = samples[inside]
    H, _, _ = np.histogram2d(s[:, 1], s[:, 0], bins=[nbin, nbin],
                             range=[[y0, y1], [x0, x1]])  # row=y to match meshgrid
    q = H.ravel()
    q = q / q.sum()
    eps = 1e-12
    ps = p + eps; ps /= ps.sum()
    qs = q + eps; qs /= qs.sum()
    tv = 0.5 * np.sum(np.abs(ps - qs))
    kl = float(np.sum(qs * np.log(qs / ps)))
    m = 0.5 * (ps + qs)
    js = float(0.5 * np.sum(ps * np.log(ps / m)) + 0.5 * np.sum(qs * np.log(qs / m)))
    return {"tv": float(tv), "kl_q_p": kl, "js": js,
            "frac_outside_box": float(1.0 - inside.mean()), "nbin": nbin}


def sample_truth(target, key, n, nbin=120):
    """Approximate i.i.d. draws from the analytic Boltzmann density, by sampling
    grid cells in proportion to Pc and jittering uniformly within each cell.

    Used for the TV FLOOR: TV between N such i.i.d. draws and the analytic density
    is the irreducible finite-sample noise, so a sampler whose TV sits near this
    floor has matched the target as well as i.i.d. sampling would.
    """
    g = grid_eval(target, n=nbin)
    Pc = g["Pc"].ravel()
    idx = np.asarray(jax.random.choice(key, Pc.size, shape=(n,), p=jnp.asarray(Pc)))
    iy, ix = np.divmod(idx, nbin)
    jit = np.asarray(jax.random.uniform(jax.random.fold_in(key, 1), (n, 2))) - 0.5
    x = g["xs"][ix] + jit[:, 0] * g["dx"]
    y = g["ys"][iy] + jit[:, 1] * g["dy"]
    return np.stack([x, y], axis=1)


# --- occupancy with autocorrelation-aware CIs ----------------------------------
def sample_occupancy(X, target, grid, labels, z=1.96):
    """Sampled basin fractions f_k with autocorrelation-corrected CIs.

    X : (n_steps, n_walkers, 2) post-burn samples.  Each sample is assigned the
    basin of its grid cell (respects the energy landscape, unlike nearest-centre
    across a saddle).  CI uses N_eff = N/tau of the basin-indicator series
    (pooled over walkers via tau_sokal), so it is honest for correlated MCMC.
    """
    n_steps, n_walkers, _ = X.shape
    K = labels.max() + 1
    flat = X.reshape(-1, 2)
    ix, iy = _cell_index(flat, grid)
    lab = labels[iy, ix].reshape(n_steps, n_walkers)  # (T, M)
    f = np.array([(lab == k).mean() for k in range(K)])
    cis = []
    for k in range(K):
        ind = (lab == k).astype(float)  # (T, M)
        tau, _ = tau_sokal(ind, c=5.0)
        n_eff = ind.size / max(tau, 1.0)
        se = np.sqrt(max(f[k] * (1 - f[k]), 1e-12) / n_eff)
        cis.append((max(0.0, f[k] - z * se), min(1.0, f[k] + z * se), float(tau), float(n_eff)))
    return f, cis, lab


# --- committed (core-to-core) transitions and MFPT -----------------------------
def committed_transitions(X, grid, labels, core):
    """Core-to-core transition counts + MFPT (steps).  See transitions_mfpt doc.

    X : (n_steps, n_walkers, 2).  Returns dict with count matrix (K,K), mfpt
    matrix (K,K), and total committed transitions.
    """
    n_steps, n_walkers, _ = X.shape
    K = int(labels.max() + 1)
    counts = np.zeros((K, K))
    fpts = {(i, j): [] for i in range(K) for j in range(K) if i != j}

    for w in range(n_walkers):
        ix, iy = _cell_index(X[:, w, :], grid)
        lab_w = labels[iy, ix]
        core_w = core[iy, ix]
        committed = -1
        commit_time = 0
        for t in range(n_steps):
            if core_w[t]:
                b = lab_w[t]
                if b != committed:
                    if committed != -1:
                        counts[committed, b] += 1
                        fpts[(committed, b)].append(t - commit_time)
                    committed = b
                    commit_time = t
    mfpt = np.full((K, K), np.nan)
    for (i, j), v in fpts.items():
        if v:
            mfpt[i, j] = float(np.mean(v))
    return {"counts": counts, "mfpt": mfpt, "total": int(counts.sum())}


# --- convergence ---------------------------------------------------------------
def split_rhat(scalar):
    """Gelman-Rubin split-Rhat for a scalar quantity.

    scalar : (n_steps, n_walkers).  Splits each walker in half -> 2M chains, then
    the standard between/within variance ratio.  Target < 1.01.
    """
    x = np.asarray(scalar, dtype=float)
    n_steps, m = x.shape
    half = n_steps // 2
    chains = np.concatenate([x[:half], x[half:2 * half]], axis=1).T  # (2M, half)
    Mc, N = chains.shape
    means = chains.mean(1)
    B = N * means.var(ddof=1)
    W = chains.var(axis=1, ddof=1).mean()
    var_hat = (N - 1) / N * W + B / N
    return float(np.sqrt(var_hat / W)) if W > 0 else np.nan
