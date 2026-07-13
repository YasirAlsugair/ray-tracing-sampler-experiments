"""Shared configuration. Import this BEFORE any other jax-using module so that
float64 is enabled before the first array is created.

float64 is not optional here: the ray-tracing kick accumulates
delta_ln_L = (1 - D) * log(f_v) every leapfrog step, and in D >~ 50 the float32
rounding of f_v = sin(theta_f)/sin(theta_i) makes that tally (and therefore the
Metropolis test) unstable.
"""

import os

# Keep JAX on CPU and quiet; we never need a GPU for these toy targets.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

assert jnp.zeros(1).dtype == jnp.float64, "float64 did not take effect"

# One master seed for the whole study; every experiment derives its keys from it
# so figures regenerate bit-for-bit.
MASTER_SEED = 20260619
