from typing import Any
from functools import partial

import jax
from jax import Array, random, numpy as jnp

from .base import Prior, Model


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def info_gain(
    key: Array,
    design: Array,
    prior: Prior,
    model: Model,
    num_outer: int,
    num_inner: int,
    alpha: float,
) -> Array:
    """
    Generic nested Monte Carlo estimator for Sibson information gain at a single design,
    using log‐domain importance weights for numerical stability.

    Args:
        key: JAX PRNGKey
        design: Design point (shape depends on problem)
        prior: Prior distribution object with sample() and logpdf() methods
        model: Likelihood model object with sample() and logpdf() methods
        num_outer: Number of outer samples
        num_inner: Number of inner samples per outer sample
        alpha: Sibson order parameter

    Returns:
        info: Estimated Sibson information gain (float)
    """

    # 1. Sample outer theta from prior
    key, outer_key = random.split(key, 2)
    outer_thetas = prior.sample(outer_key, num_outer)

    # 2. For each outer theta, sample x ~ p(x | xi, theta)
    key, sub_key = random.split(key)
    x_keys = random.split(sub_key, num_outer)
    x_samples = jax.vmap(lambda k, theta: model.sample(k, design, theta))(x_keys, outer_thetas)

    # 3. Inner integral: for each (x, theta), approximate
    #   g(x) = \int p(theta) [ p(x | xi, theta) / p(x | xi)]^alpha dtheta
    def inner_log_integral(key, x):
        key, inner_key = random.split(key, 2)
        inner_thetas = prior.sample(inner_key, num_inner)
        log_lklhd = jax.vmap(lambda theta: model.logpdf(x, design, theta))(inner_thetas)
        log_mrgnl = jax.nn.logsumexp(log_lklhd) - jnp.log(num_inner)
        log_ws = alpha * (log_lklhd - log_mrgnl)
        return jax.nn.logsumexp(log_ws) - jnp.log(num_inner)

    key, sub_key = random.split(key)
    inner_keys = random.split(sub_key, num_outer)
    log_g = jax.vmap(inner_log_integral)(inner_keys, x_samples)

    # 4. Outer expectation: \int p(x | xi) * g(x)^(1 / alpha) dx
    info = jax.nn.logsumexp(log_g / alpha) - jnp.log(num_outer)
    return (alpha / (alpha - 1)) * info
