from typing import Callable
from functools import partial

import jax
import optax

from jax import Array, random, numpy as jnp
from flax.training.train_state import TrainState
from distrax import Lambda

from rdesign.base import REPSConfig, StepInfo, OptimizationResult
from rdesign.policy import Policy, GaussianParams
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
    policy_state: TrainState,
    samples: Array,
    weights: Array,
) -> tuple[TrainState, Array]:
    """Update policy parameters using weighted maximum likelihood.

    Args:
        policy: Policy object with log_prob method.
        policy_state: Current training state with parameters and optimizer.
        samples: Batch of samples to fit.
        weights: Importance weights for each sample.

    Returns:
        Tuple of (updated policy state, loss value).
    """

    def loss_fn(params):
        log_probs = jax.vmap(lambda s: policy.log_prob(s, params))(samples)
        return -1.0 * jnp.sum(weights * log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(policy_state.params)
    policy_state = policy_state.apply_gradients(grads=grads)
    return policy_state, loss


def optimize_policy(
    rng_key: Array,
    func: Callable,
    policy: Policy,
    policy_state: TrainState,
    config: REPSConfig,
) -> OptimizationResult:
    """
    Optimize a policy using REPS as mirror descent.

    This implements policy optimization as mirror descent in the space of
    distributions, where each iteration:
        1. Samples from current policy
        2. Computes importance weights via softmax: w_i ∝ exp(f(x_i) / tau)
        3. Projects back to policy family via weighted MLE

    Args:
        rng_key: JAX PRNG key for stochastic sampling.
        func: Objective function to maximize. Maps samples to scalar values.
        policy: Policy defining the search distribution family.
        policy_state: Initial training state containing policy parameters.
        config: REPS configuration.

    Returns:
        OptimizationResult with optimized policy state and training history.
    """
    history: dict[str, list] = {"value": [], "entropy": [], "temperature": []}

    def mirror_descent_step(
        rng_key: Array,
        policy_state: TrainState,
        iteration: int,
    ) -> tuple[TrainState, StepInfo]:
        """Single mirror descent iteration."""
        sample_key, batch_key = random.split(rng_key)

        # Sample from current policy
        sample_keys = random.split(sample_key, config.num_samples)
        samples = jax.vmap(lambda k: policy.sample(k, policy_state.params))(sample_keys)

        # Evaluate objective
        values = jax.vmap(func)(samples)

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
            policy_state, _ = update_policy(
                policy=policy,
                policy_state=policy_state,
                samples=samples_batch,
                weights=weights_batch,
            )

        return policy_state, StepInfo(
            mean_value=jnp.mean(values),
            entropy=policy.entropy(policy_state.params),
            temperature=tau,
        )

    # Mirror descent loop
    for iteration in range(config.max_iterations):
        rng_key, step_key = random.split(rng_key)
        policy_state, info = mirror_descent_step(step_key, policy_state, iteration)

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

    return OptimizationResult(policy_state=policy_state, history=history)


def main():
    dim = 2
    rng_key = jax.random.PRNGKey(0)
    init_key, opt_key = jax.random.split(rng_key)

    # Configure REPS as mirror descent
    config = REPSConfig(
        num_samples=100,
        batch_size=16,
        max_iterations=500,
        log_interval=50,
    )

    # Create policy with custom initialization
    params_module = GaussianParams(
        dim=dim,
        init_mean=lambda key, shape: 2.5 * jnp.ones((dim,)),
        init_log_std=lambda key, shape: jnp.zeros((dim,)),
    )
    policy = GaussianPolicy(params_module, bijector=Lambda(lambda x: x))

    # Initialize policy state
    optimizer = optax.adam(learning_rate=0.01)
    policy_state = policy.init_state(init_key, optimizer)

    # Define objective (maximize negative quadratic -> find origin)
    objective = lambda x: -0.5 * jnp.dot(x, x)

    # Run optimization
    result = optimize_policy(
        rng_key=opt_key,
        func=objective,
        policy=policy,
        policy_state=policy_state,
        config=config,
    )

    print(f"\nFinal mean: {result.policy_state.params['mean']}")


if __name__ == "__main__":
    main()
