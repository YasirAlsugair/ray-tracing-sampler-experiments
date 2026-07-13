"""Diagnostics: integrated autocorrelation time (Sokal), ESS, mode occupancy,
inter-mode switch rate, and efficiency. Pure numpy/scipy; operates on sampled
arrays so it is agnostic to RT vs HMC.

tau_sokal reimplements the estimator shipped in the repo's util/acor_estimate.c
(tau = 1 + 2*sum_k rho_k with automatic windowing when lag > c*tau). It is the
standard Sokal/emcee estimator; we validate it against emcee.autocorr in
experiments (AR(1) has a known tau).
"""
import numpy as np


def _next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x):
    """Normalised autocovariance of a 1-D series via FFT (Wiener-Khinchin)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = _next_pow_two(2 * len(x))
    f = np.fft.fft(x, n=n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    if acf[0] == 0:
        return np.zeros_like(acf)
    return acf / acf[0]


def _auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(~m):
        return int(np.argmin(m))
    return len(taus) - 1


def tau_sokal(x, c=5.0):
    """Integrated autocorrelation time of a scalar quantity.

    x : (n_steps,) or (n_steps, n_walkers). Multiple walkers are combined by
    averaging their autocorrelation functions (emcee convention), which is the
    right thing when each walker is an independent chain of the same target.
    Returns tau (float). ESS of the pooled draws is n_steps * n_walkers / tau.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n_steps, n_walkers = x.shape
    acf = np.mean([autocorr_func_1d(x[:, w]) for w in range(n_walkers)], axis=0)
    taus = 2.0 * np.cumsum(acf) - 1.0
    window = _auto_window(taus, c)
    tau = float(taus[window])
    return max(tau, 1.0), window


def ess(x, c=5.0):
    """Effective sample size of the pooled draws of a scalar quantity."""
    x = np.asarray(x, dtype=float)
    n_total = x.size
    tau, _ = tau_sokal(x, c=c)
    return n_total / tau


def tau_per_coordinate(X, c=5.0):
    """X : (n_steps, n_walkers, D). Returns array of D integrated times."""
    X = np.asarray(X, dtype=float)
    n_steps, n_walkers, D = X.shape
    return np.array([tau_sokal(X[:, :, d], c=c)[0] for d in range(D)])


def ess_summary(X, c=5.0):
    """Conservative scalar ESS diagnostics for a multivariate chain.

    X : (n_steps, n_walkers, D). Returns dict with the worst-coordinate tau/ESS
    and the ESS of the slowest linear (PCA-1) mode, which for multimodal targets
    typically tracks the mode-indicator (the genuinely slow direction).
    """
    X = np.asarray(X, dtype=float)
    n_steps, n_walkers, D = X.shape
    n_total = n_steps * n_walkers
    taus = tau_per_coordinate(X, c=c)
    tau_worst = float(np.max(taus))

    # slowest linear mode: project onto the top principal axis of the pooled chain
    flat = X.reshape(-1, D)
    flat = flat - flat.mean(0)
    cov = np.cov(flat.T) if D > 1 else np.array([[np.var(flat)]])
    w, V = np.linalg.eigh(np.atleast_2d(cov))
    pc1 = V[:, -1]
    proj = (X - X.reshape(-1, D).mean(0)) @ pc1  # (n_steps, n_walkers)
    tau_pca1, _ = tau_sokal(proj, c=c)

    return {
        "tau_mean": float(np.mean(taus)),
        "tau_worst": tau_worst,
        "tau_pca1": float(tau_pca1),
        "ess_worst_coord": n_total / tau_worst,
        "ess_pca1": n_total / tau_pca1,
        "n_total": n_total,
    }


# --- mode-coverage diagnostics -------------------------------------------------
def assign_modes(X, centers):
    """Nearest-centre assignment. X:(...,D), centers:(K,D) -> integer labels (...)."""
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    d2 = ((X[..., None, :] - centers) ** 2).sum(-1)  # (..., K)
    return np.argmin(d2, axis=-1)


def mode_occupancy(X, centers):
    """Fraction of samples assigned to each mode. Returns array of length K."""
    labels = assign_modes(X, centers)
    K = len(centers)
    counts = np.bincount(labels.reshape(-1), minlength=K)
    return counts / counts.sum()


def switch_rate(X, centers):
    """Mean per-step probability of changing mode along a walker.

    X : (n_steps, n_walkers, D). Returns (switch_rate, transition_matrix KxK)."""
    labels = assign_modes(X, centers)  # (n_steps, n_walkers)
    K = len(centers)
    a, b = labels[:-1], labels[1:]
    switches = (a != b).mean()
    T = np.zeros((K, K))
    for i, j in zip(a.reshape(-1), b.reshape(-1)):
        T[i, j] += 1
    row = T.sum(1, keepdims=True)
    T = np.divide(T, row, out=np.zeros_like(T), where=row > 0)
    return float(switches), T


def far_mode_coverage(X, centers, home=0):
    """Fraction of WALKERS that visit at least one mode other than `home`.

    The Ricardo init-variance test: do walkers started near the origin ever reach
    the far modes? X:(n_steps, n_walkers, D)."""
    labels = assign_modes(X, centers)  # (n_steps, n_walkers)
    visited_far = np.array([
        len(set(labels[:, w].tolist()) - {home}) > 0 for w in range(labels.shape[1])
    ])
    n_modes_visited = np.array([len(set(labels[:, w].tolist())) for w in range(labels.shape[1])])
    return float(visited_far.mean()), float(n_modes_visited.mean())


def efficiency(ess_value, n_grad_evals, seconds):
    """Two efficiency numbers that neutralise step-size / wall-clock differences."""
    return {
        "ess_per_grad_eval": ess_value / n_grad_evals,
        "ess_per_sec": ess_value / seconds,
    }
