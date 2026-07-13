"""Test targets. Every log_prob is a jax-differentiable function of a flat (D,)
array, so it drops straight into sample_raytrace / sample_hamiltonian.

Each Target also carries the analytic 'truth' needed to score the sampler:
mode centres, mixture weights (= occupancy in the well-separated limit), and an
exact sampler for KS / moment checks.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import jax
import jax.numpy as jnp


@dataclass
class Target:
    name: str
    D: int
    log_prob: Callable
    centers: Optional[np.ndarray] = None          # (K, D)
    weights: Optional[np.ndarray] = None           # (K,)
    analytic_occupancy: Optional[np.ndarray] = None
    sigma: float = 1.0
    sample: Optional[Callable] = None              # (key, n) -> (n, D) exact draws
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
def gaussian(D, cond=1.0):
    """Zero-mean Gaussian with condition number `cond` (eigenvalues log-spaced
    from 1 to cond along the axes)."""
    var = np.logspace(0.0, np.log10(cond), D) if D > 1 else np.array([1.0])
    inv_var = jnp.asarray(1.0 / var)
    std = jnp.asarray(np.sqrt(var))

    def log_prob(x):
        return -0.5 * jnp.sum(x ** 2 * inv_var)

    def sample(key, n):
        return np.asarray(jax.random.normal(key, (n, D)) * std)

    return Target(name=f"gaussian_D{D}_cond{cond:g}", D=D, log_prob=log_prob,
                  centers=np.zeros((1, D)), weights=np.array([1.0]),
                  analytic_occupancy=np.array([1.0]), sample=sample,
                  meta={"cond": cond, "var": var})


# ---------------------------------------------------------------------------
def gaussian_mixture(D, sep, K=2, weights=None, sigma=1.0):
    """K isotropic Gaussians spaced `sep` apart along axis 0, mode 0 at the
    origin (so walkers started near 0 have 'home' = mode 0 and the rest are the
    far modes). analytic_occupancy = weights (exact in the well-separated limit).
    """
    if weights is None:
        weights = np.ones(K) / K
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    centers = np.zeros((K, D))
    centers[:, 0] = np.arange(K) * sep

    centers_j = jnp.asarray(centers)
    logw_j = jnp.asarray(np.log(weights))
    inv2s2 = 1.0 / (2.0 * sigma ** 2)
    norm = -0.5 * D * np.log(2 * np.pi * sigma ** 2)

    def log_prob(x):
        d2 = jnp.sum((x[None, :] - centers_j) ** 2, axis=1)  # (K,)
        return jax.scipy.special.logsumexp(logw_j + norm - inv2s2 * d2)

    def sample(key, n):
        k1, k2 = jax.random.split(key)
        comp = np.asarray(jax.random.choice(k1, K, shape=(n,), p=jnp.asarray(weights)))
        z = np.asarray(jax.random.normal(k2, (n, D))) * sigma
        return z + centers[comp]

    return Target(name=f"mixture_D{D}_K{K}_sep{sep:g}", D=D, log_prob=log_prob,
                  centers=centers, weights=weights, analytic_occupancy=weights,
                  sigma=sigma, sample=sample, meta={"sep": sep, "K": K})


# ---------------------------------------------------------------------------
def disjoint_islands(D, sep=12.0, K=3, R=3.0, sigma=1.0):
    """K Gaussian 'balls' of radius R with L=0 (log L = -1e30) in the gaps
    between them. The released sampler has no re-emission, so a ray that steps
    into a gap is rejected -- this is the honest 'trapped' boundary case.
    Centres along axis 0 spaced `sep`; equal mass per ball -> uniform occupancy.
    """
    centers = np.zeros((K, D))
    centers[:, 0] = np.arange(K) * sep
    centers_j = jnp.asarray(centers)
    inv2s2 = 1.0 / (2.0 * sigma ** 2)
    R2 = R ** 2

    def log_prob(x):
        d2 = jnp.sum((x[None, :] - centers_j) ** 2, axis=1)  # (K,)
        ll = jnp.where(d2 <= R2, -inv2s2 * d2, -jnp.inf)
        inside = jnp.any(d2 <= R2)
        best = jnp.max(ll)
        return jnp.where(inside, best, -1e30)

    def sample(key, n):
        # truncated-ish: sample gaussian around a random ball, reject outside R
        k1, k2 = jax.random.split(key)
        comp = np.asarray(jax.random.choice(k1, K, shape=(n,)))
        z = np.asarray(jax.random.normal(k2, (n, D))) * sigma
        return z + centers[comp]

    return Target(name=f"islands_D{D}_K{K}_sep{sep:g}_R{R:g}", D=D, log_prob=log_prob,
                  centers=centers, weights=np.ones(K) / K,
                  analytic_occupancy=np.ones(K) / K, sigma=sigma, sample=sample,
                  meta={"sep": sep, "K": K, "R": R})


# ---------------------------------------------------------------------------
def rosenbrock(D, a=1.0, b=20.0):
    """Curved, non-Gaussian banana (D must be even). No closed-form occupancy;
    used purely as a hard autocorrelation target. b=20 mild, b=100 hard."""
    assert D % 2 == 0, "rosenbrock needs even D"

    def log_prob(x):
        x0 = x[0::2]
        x1 = x[1::2]
        return -jnp.sum(b * (x1 - x0 ** 2) ** 2 + (a - x0) ** 2)

    return Target(name=f"rosenbrock_D{D}_b{b:g}", D=D, log_prob=log_prob,
                  meta={"a": a, "b": b})


# ---------------------------------------------------------------------------
# Molecular-dynamics-style 2D Boltzmann targets.  P(x) propto exp(-U(x)/kT),
# so log L = -U/kT and grad(log_prob) = -grad U / kT = +force/kT.  This is the
# SAME seam the alanine-dipeptide backend will use; getting these right also
# validates the MD plumbing.  Analytic truth (Z, basins, true weights) is
# computed in rts/analytic2d.py straight from log_prob, so meta only needs the
# viewing box and kT.  meta["kind"]=="boltzmann2d" marks them for that module.
# ---------------------------------------------------------------------------
def _uniform_box_sampler(box):
    """Overdispersed pilot/init sampler: uniform over the viewing box. Feeds
    tune_step_size and the Ricardo init-variance test (wide init -> mode
    discovery)."""
    lo = np.array([b[0] for b in box], dtype=float)
    hi = np.array([b[1] for b in box], dtype=float)

    def sample(key, n):
        u = np.asarray(jax.random.uniform(key, (n, len(box))))
        return lo + u * (hi - lo)

    return sample


def double_well(kT=1.0, h=4.0, a=1.0, ky=1.0, box=((-2.2, 2.2), (-2.2, 2.2))):
    """Symmetric 2D double well  U = h*((x/a)^2 - 1)^2 + 0.5*ky*y^2.

    Minima at (+-a, 0); barrier height h at x=0; true occupancy 1/2 each by
    symmetry.  The controlled analytic sanity check to pass BEFORE Muller-Brown.
    Pick h so h/kT ~ 3-6 (a crossable-but-nontrivial barrier).
    """
    def log_prob(x):
        U = h * ((x[0] / a) ** 2 - 1.0) ** 2 + 0.5 * ky * x[1] ** 2
        return -U / kT

    centers = np.array([[-a, 0.0], [a, 0.0]])
    return Target(name=f"double_well_kT{kT:g}_h{h:g}", D=2, log_prob=log_prob,
                  centers=centers, weights=np.array([0.5, 0.5]),
                  analytic_occupancy=np.array([0.5, 0.5]),
                  sample=_uniform_box_sampler(box),
                  meta={"kind": "boltzmann2d", "kT": float(kT), "h": float(h),
                        "a": float(a), "ky": float(ky), "box": box})


_MB = dict(
    A=np.array([-200.0, -100.0, -170.0, 15.0]),
    a=np.array([-1.0, -1.0, -6.5, 0.7]),
    b=np.array([0.0, 0.0, 11.0, 0.6]),
    c=np.array([-10.0, -10.0, -6.5, 0.7]),
    x0=np.array([1.0, 0.0, -0.5, -1.0]),
    y0=np.array([0.0, 0.5, 1.5, 1.0]),
)


def muller_brown(kT=15.0, box=((-1.7, 1.3), (-0.5, 2.3)), scale=1.0):
    """Muller-Brown potential: 3 minima separated by 2 curved saddles, the
    standard low-D rare-event benchmark.  U is in the conventional energy units;
    choose kT so the dominant barrier is a few kT (kT~15 default -- kT=1 gives
    ~100 kT barriers and nothing crosses).  `scale` multiplies U to sweep barrier
    height at fixed kT.  log L = -scale*U/kT.
    """
    A = jnp.asarray(_MB["A"]); aa = jnp.asarray(_MB["a"]); bb = jnp.asarray(_MB["b"])
    cc = jnp.asarray(_MB["c"]); X0 = jnp.asarray(_MB["x0"]); Y0 = jnp.asarray(_MB["y0"])

    def log_prob(x):
        dx = x[0] - X0
        dy = x[1] - Y0
        U = jnp.sum(A * jnp.exp(aa * dx ** 2 + bb * dx * dy + cc * dy ** 2))
        return -scale * U / kT

    # approximate minima (deep -> shallow); analytic2d polishes these.
    centers = np.array([[-0.558, 1.442], [0.623, 0.028], [-0.050, 0.467]])
    return Target(name=f"muller_brown_kT{kT:g}_s{scale:g}", D=2, log_prob=log_prob,
                  centers=centers, sample=_uniform_box_sampler(box),
                  meta={"kind": "boltzmann2d", "kT": float(kT), "scale": float(scale),
                        "box": box})
