from typing import Callable
from functools import partial

import jax

from jax import Array, random, numpy as jnp
from flax.training.train_state import TrainState

from rdesign.policy import Policy
from rdesign.base import REPSConfig, StepInfo, OptimizationResult
from rdesign.utils import batch_samples, log_progress, inv_sqrt_schedule


def compute_weights(values: Array, tau: float) -> Array:
    """Compute softmax importance weights from objective values.

    Args:
        values: Objective function values for each sample.
        tau: Temperature parameter controlling sharpness of weights.

    Returns:
        Normalized importance weights summing to 1.
    """
    return jax.nn.softmax(values / tau)


@partial(jax.jit, static_argnames="policy")
def update_policy(
    policy: Policy,
    state: TrainState,
    samples: Array,
    weights: Array,
) -> tuple[TrainState, Array]:
    """Update policy parameters using weighted maximum likelihood.

    Args:
        policy: Policy object with log_prob method.
        state: Current training state with parameters and optimizer.
        samples: Batch of samples to fit.
        weights: Importance weights for each sample.

    Returns:
        Tuple of (updated policy state, loss value).
    """

    def loss_fn(params):
        log_probs = jax.vmap(lambda s: policy.log_prob(s, params))(samples)
        return -1.0 * jnp.sum(weights * log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def optimize_policy(
    rng_key: Array,
    func: Callable,
    policy: Policy,
    state: TrainState,
    config: REPSConfig,
) -> OptimizationResult:
    """
    Optimize a policy using REPS as policy search method.

    Args:
        rng_key: JAX PRNG key for stochastic sampling.
        func: Objective function to maximize. Maps (key, sample) to scalar value.
        policy: Policy defining the search distribution family.
        state: Initial training state containing policy parameters.
        config: REPS configuration.

    Returns:
        OptimizationResult with optimized policy state and training history.
    """
    history: dict[str, list] = {"value": [], "entropy": [], "temperature": []}

    def optimization_step(
        rng_key: Array,
        state: TrainState,
        iteration: int,
    ) -> tuple[TrainState, StepInfo]:
        """Single REPS iteration."""
        sample_key, func_key, batch_key = random.split(rng_key, 3)

        # Sample from current policy
        sample_keys = random.split(sample_key, config.num_samples)
        samples, log_probs = jax.vmap(lambda k: policy.sample_and_log_prob(k, state.params))(sample_keys)

        # Evaluate objective
        func_keys = random.split(func_key, config.num_samples)
        values = jax.vmap(lambda k, s, lp: func(k, s, lp, config.lmbda))(func_keys, samples, log_probs)

        # Compute temperature and importance weights
        tau = inv_sqrt_schedule(t=iteration, tau_init=1.0)
        weights = compute_weights(values, tau)

        # Project to policy family via weighted MLE
        batch_indices = batch_samples(
            rng_key=batch_key,
            sample_size=config.num_samples,
            batch_size=config.batch_size,
        )
        for idx in batch_indices:
            samples_batch = jax.tree_util.tree_map(lambda x: x[idx, ...], samples)
            weights_batch = jax.tree_util.tree_map(lambda x: x[idx, ...], weights)
            state, _ = update_policy(
                policy=policy,
                state=state,
                samples=samples_batch,
                weights=weights_batch,
            )

        return state, StepInfo(
            mean_value=jnp.mean(values),
            entropy=policy.entropy(state.params),
            temperature=tau,
        )

    # Policy search loop
    for iteration in range(config.max_iterations):
        rng_key, step_key = random.split(rng_key)
        state, info = optimization_step(step_key, state, iteration)

        history["value"].append(float(info.mean_value))
        history["entropy"].append(float(info.entropy))
        history["temperature"].append(float(info.temperature))

        log_progress(
            iteration=iteration,
            max_iterations=config.max_iterations,
            metrics={
                "tau": info.temperature,
                "value": info.mean_value,
                "entropy": info.entropy,
            },
            log_interval=config.log_interval,
            verbose=config.verbose,
        )

    return OptimizationResult(state=state, history=history)
