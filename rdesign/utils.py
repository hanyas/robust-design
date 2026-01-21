from typing import Any
from functools import partial
import math

import jax
from jax import Array, random, numpy as jnp

from rdesign.base import Prior, Likelihood


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


@partial(jax.jit, static_argnums=(4, 5, 6))
def sibson_alpha_mutual_information(
    rng_key: Array,
    design: Array,
    prior: Prior,
    lklhd: Likelihood,
    num_outer: int,
    num_inner: int,
    alpha: float,
) -> Array:
    """
    Nested Monte Carlo estimator for Sibson's α-mutual information.
    
    Args:
        rng_key: JAX PRNGKey
        design: Design point (shape depends on problem)
        prior: Prior distribution
        lklhd: Likelihood model
        num_outer: Number of outer samples
        num_inner: Number of inner samples per outer sample
        alpha: Sibson order parameter α ∈ (0, 1)
    
    Returns:
        Estimated Sibson information gain (scalar)
    """
    # 1. Sample outer theta from prior
    rng_key, outer_key = random.split(rng_key)
    outer_thetas = prior.sample(outer_key, num_outer)
    
    # 2. For each outer theta, sample x ~ p(x | xi, theta)
    rng_key, sub_key = random.split(rng_key)
    x_keys = random.split(sub_key, num_outer)
    x_samples = jax.vmap(lambda k, theta: lklhd.sample(k, design, theta))(
        x_keys, outer_thetas
    )
    
    # 3. Inner integral: for each (x, theta), approximate
    #    g(x) = ∫ p(θ) [p(x|ξ,θ) / p(x|ξ)]^α dθ
    def inner_log_integral(rng_key, x):
        rng_key, inner_key = random.split(rng_key)
        inner_thetas = prior.sample(inner_key, num_inner)
        
        log_lklhd = jax.vmap(lambda theta: lklhd.logpdf(x, design, theta))(inner_thetas)
        log_mrgnl = jax.nn.logsumexp(log_lklhd) - jnp.log(num_inner)
        log_ws = alpha * (log_lklhd - log_mrgnl)
        
        return jax.nn.logsumexp(log_ws) - jnp.log(num_inner)
    
    rng_key, sub_key = random.split(rng_key)
    inner_keys = random.split(sub_key, num_outer)
    log_g = jax.vmap(inner_log_integral)(inner_keys, x_samples)
    
    # 4. Outer expectation: ∫ p(x|ξ) g(x)^(1/α) dx
    info = jax.nn.logsumexp(log_g / alpha) - jnp.log(num_outer)
    
    return (alpha / (alpha - 1)) * info
