"""Multi-walker driver. The JAX sampler is single-chain, so we vmap it over
independent (key, init) pairs to get an ensemble. Also: fair step-size tuning to
a target acceptance rate, and exact gradient-eval / wall-clock accounting so the
efficiency metric can't be gamed by step size.

step_size is passed as a *traced* argument (not closed over), so sweeping it
during tuning reuses a single compiled function instead of recompiling per value.
refresh_rate / n_steps / n_leapfrog stay static (they pick code paths / loop
lengths) and key the compile cache.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

from .raytrace_jax_W import sample_raytrace

_FN_CACHE = {}


def _single_chain_fn(kind, log_prob, log_W, n_steps, n_leapfrog, refresh_rate):
    is_hmc = kind == "hmc"

    def f(key, x0, step_size):
        chain, ll, acc = sample_raytrace(
            key, x0, log_prob, n_steps, n_leapfrog, step_size,
            refresh_rate, is_hmc, 1, None if is_hmc else log_W, False)
        return chain, ll, acc

    return f


def _get_fn(kind, target, log_W_fn, n_steps, n_leapfrog, refresh_rate):
    key = (kind, id(target.log_prob), id(log_W_fn), int(n_steps), int(n_leapfrog), float(refresh_rate))
    if key not in _FN_CACHE:
        base = _single_chain_fn(kind, target.log_prob, log_W_fn, n_steps, n_leapfrog, refresh_rate)
        _FN_CACHE[key] = jax.jit(jax.vmap(base, in_axes=(0, 0, None)))
    return _FN_CACHE[key]


def run_many(kind, key, inits, target, *, step_size, n_steps, n_leapfrog,
             refresh_rate=0, log_W_fn=None):
    """Run M independent walkers. inits: (M, D).

    Returns dict with chain (M, n_steps, D), ll (M, n_steps), accept (M, n_steps),
    plus seconds and n_grad_evals (M * n_steps * n_leapfrog; one grad per leapfrog
    step, including the W-tilted grad which is a single jax.grad call).
    """
    inits = jnp.asarray(inits, dtype=jnp.float64)
    M = inits.shape[0]
    keys = jax.random.split(key, M)
    f = _get_fn(kind, target, log_W_fn, n_steps, n_leapfrog, refresh_rate)

    t0 = time.perf_counter()
    chain, ll, acc = f(keys, inits, jnp.float64(step_size))
    chain = jax.block_until_ready(chain)
    seconds = time.perf_counter() - t0

    return {
        "chain": np.asarray(chain),          # (M, n_steps, D)
        "ll": np.asarray(ll),                # (M, n_steps)
        "accept": np.asarray(acc),           # (M, n_steps)
        "seconds": seconds,
        "n_grad_evals": M * n_steps * n_leapfrog,
        "kind": kind,
        "step_size": float(step_size),
        "n_leapfrog": int(n_leapfrog),
    }


def chain_steps_walkers_dims(res):
    """Reshape (M, n_steps, D) -> (n_steps, M, D) for the metrics module."""
    return np.transpose(res["chain"], (1, 0, 2))


def _pilot_inits(key, target, M):
    if target.sample is not None:
        return target.sample(key, M)
    return np.asarray(jax.random.normal(key, (M, target.D)))


def tune_step_size(kind, key, target, *, n_leapfrog, target_accept=0.7, grid=None,
                   n_pilot_walkers=8, n_pilot_steps=300, refresh_rate=0, log_W_fn=None):
    """Pick the step size whose mean acceptance is closest to target_accept.

    Pilots start in the typical set (drawn from target.sample when available) so
    the tuned step size reflects mixing, not transient burn-in.
    """
    if grid is None:
        grid = np.geomspace(0.008, 3.0, 14)
    k_init, k_run = jax.random.split(key)
    inits = _pilot_inits(k_init, target, n_pilot_walkers)

    best = None
    table = []
    for ss in grid:
        res = run_many(kind, k_run, inits, target, step_size=float(ss),
                       n_steps=n_pilot_steps, n_leapfrog=n_leapfrog,
                       refresh_rate=refresh_rate, log_W_fn=log_W_fn)
        a = float(np.mean(res["accept"]))
        table.append((float(ss), a))
        score = abs(a - target_accept)
        if best is None or score < best[2]:
            best = (float(ss), a, score)
    return {"step_size": best[0], "accept": best[1], "target_accept": target_accept,
            "grid": table}
