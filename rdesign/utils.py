from typing import Any
from functools import partial
import math

import jax
from jax import Array, random, numpy as jnp

from rdesign.base import Prior, Model


def inv_sqrt_schedule(t: int, tau_init: float) -> float:
    """Temperature schedule decaying as 1/sqrt(iteration).

    Args:
        t: Current iteration (0-indexed).
        tau_init: Initial temperature value.

    Returns:
        Temperature at iteration t: tau_init / sqrt(t + 1).
    """
    return tau_init / jnp.sqrt(t + 1)


def batch_samples(
    rng_key: Array,
    sample_size: int,
    batch_size: int,
    skip_last: bool = True,
) -> list[Array]:
    """Generate randomized batch indices for mini-batch processing.

    Args:
        rng_key: JAX PRNG key for shuffling.
        sample_size: Total number of samples to batch.
        batch_size: Number of samples per batch.
        skip_last: If True, drop the last incomplete batch. If False, include it.

    Returns:
        List of index arrays, each of length batch_size.
    """
    batch_idx = random.permutation(rng_key, sample_size)

    if skip_last:
        # Skip incomplete batch
        num_batches = sample_size // batch_size
        batch_idx = batch_idx[: num_batches * batch_size]
    else:
        # Include incomplete batch
        num_batches = math.ceil(sample_size / batch_size)

    batch_idx = jnp.array_split(batch_idx, num_batches)
    return batch_idx


def log_progress(
    iteration: int,
    max_iterations: int,
    metrics: dict[str, float],
    log_interval: int = 1,
    verbose: bool = True,
) -> None:
    """Log optimization progress to console.

    Args:
        iteration: Current iteration number.
        max_iterations: Total number of iterations.
        metrics: Dictionary of metric names to values.
        log_interval: Iterations between logs.
        verbose: Whether to print at all.
    """
    if not verbose:
        return
    if iteration % log_interval != 0 and iteration != max_iterations - 1:
        return

    metrics_str = "  ".join(f"{k}: {v:+.4f}" for k, v in metrics.items())
    print(f"[{iteration:5d}/{max_iterations}]  {metrics_str}")


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def robust_info_gain(
    rng_key: Array,
    design: Array,
    prior: Prior,
    model: Model,
    num_outer: int,
    num_inner: int,
    alpha: float,
) -> Array:
    """
    Generic nested Monte Carlo estimator for Sibson information gain at a single design,
    using log-domain importance weights for numerical stability.

    Args:
        rng_key: JAX PRNGKey
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
    rng_key, outer_key = random.split(rng_key, 2)
    outer_thetas = prior.sample(outer_key, num_outer)

    # 2. For each outer theta, sample x ~ p(x | xi, theta)
    rng_key, sub_key = random.split(rng_key)
    x_keys = random.split(sub_key, num_outer)
    x_samples = jax.vmap(lambda k, theta: model.sample(k, design, theta))(x_keys, outer_thetas)

    # 3. Inner integral: for each (x, theta), approximate
    #   g(x) = \int p(theta) [ p(x | xi, theta) / p(x | xi)]^alpha dtheta
    def inner_log_integral(rng_key, x):
        rng_key, inner_key = random.split(rng_key, 2)
        inner_thetas = prior.sample(inner_key, num_inner)
        log_lklhd = jax.vmap(lambda theta: model.logpdf(x, design, theta))(inner_thetas)
        log_mrgnl = jax.nn.logsumexp(log_lklhd) - jnp.log(num_inner)
        log_ws = alpha * (log_lklhd - log_mrgnl)
        return jax.nn.logsumexp(log_ws) - jnp.log(num_inner)

    rng_key, sub_key = random.split(rng_key)
    inner_keys = random.split(sub_key, num_outer)
    log_g = jax.vmap(inner_log_integral)(inner_keys, x_samples)

    # 4. Outer expectation: \int p(x | xi) * g(x)^(1 / alpha) dx
    info = jax.nn.logsumexp(log_g / alpha) - jnp.log(num_outer)
    return (alpha / (alpha - 1)) * info
