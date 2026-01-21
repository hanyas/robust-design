"""
Nested Monte Carlo optimization of Sibson's alpha-MI for linear regression.

This version optimizes a policy (distribution) over a single design point
in [-1, 1] using REPS with the objective:
    Sibson's alpha-MI (nested estimator) - lambda * log pi(xi).
"""

from typing import Tuple

import jax
import optax

from flax.training.train_state import TrainState
from jax import Array, random, numpy as jnp

from rdesign.base import Policy, REPSConfig
from rdesign.bijectors import Tanh
from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian
from rdesign.policy import GaussianParams, ParametricGaussian
from rdesign.reps import compute_weights, update_policy
from rdesign.utils import (
    batch_samples,
    inv_sqrt_schedule,
    log_progress,
    sibson_alpha_mutual_information,
)

jax.config.update("jax_enable_x64", True)

# Model parameters
prior_mean = jnp.zeros(2)
prior_cov = jnp.eye(2)
noise_stddev = 0.1

# Robustness / estimator settings
alpha = 0.9
num_outer = 16
num_inner = 16

# Policy optimization settings
learning_rate = 5e-2
num_steps = 200
num_runs = 10
success_tol = 1e-2
lambda_reg = 0.1
num_policy_samples = 4096

# REPS configuration
reps_config = REPSConfig(
    num_samples=128,
    batch_size=32,
    max_iterations=num_steps,
    log_interval=20,
    verbose=True,
)

prior = Gaussian(mean=prior_mean, cov=prior_cov)
likelihood = LinearGaussian(noise_stddev=noise_stddev)
optimizer = optax.adam(learning_rate)


def estimate_sibson_alpha_mi(
    rng_key: Array,
    xi: Array,
    num_outer_samples: int = num_outer,
    num_inner_samples: int = num_inner,
    order: float = alpha,
) -> Array:
    """Estimate Sibson's alpha-MI with the nested Monte Carlo estimator."""
    design = jnp.atleast_1d(xi)
    return sibson_alpha_mutual_information(
        rng_key,
        design,
        prior,
        likelihood,
        num_outer_samples,
        num_inner_samples,
        order,
    )


def optimize_policy_with_penalty(
    rng_key: Array,
    policy: Policy,
    policy_state: TrainState,
    config: REPSConfig,
    entropy_weight: float,
    num_outer_samples: int = num_outer,
    num_inner_samples: int = num_inner,
    order: float = alpha,
) -> Tuple[TrainState, dict[str, list]]:
    """Optimize policy with REPS and an entropy-regularized objective."""
    if not 0.0 <= entropy_weight <= 1.0:
        raise ValueError("entropy_weight must be between 0 and 1.")

    history: dict[str, list] = {"value": [], "entropy": [], "temperature": []}

    for iteration in range(config.max_iterations):
        rng_key, step_key = random.split(rng_key)
        sample_key, eval_key, batch_key = random.split(step_key, 3)

        sample_keys = random.split(sample_key, config.num_samples)
        samples, log_probs = jax.vmap(
            lambda k: policy.sample_and_log_prob(k, policy_state.params)
        )(sample_keys)

        eval_keys = random.split(eval_key, config.num_samples)
        mi_values = jax.vmap(
            lambda k, x: estimate_sibson_alpha_mi(
                k,
                x,
                num_outer_samples=num_outer_samples,
                num_inner_samples=num_inner_samples,
                order=order,
            )
        )(eval_keys, samples)

        values = mi_values - entropy_weight * log_probs
        tau = inv_sqrt_schedule(t=iteration, tau_init=1.0)
        weights = compute_weights(values, tau)

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

        history["value"].append(float(jnp.mean(values)))
        history["entropy"].append(float(policy.entropy(policy_state.params)))
        history["temperature"].append(float(tau))

        log_progress(
            iteration=iteration,
            max_iterations=config.max_iterations,
            metrics={
                "tau": tau,
                "value": jnp.mean(values),
                "entropy": policy.entropy(policy_state.params),
            },
            log_interval=config.log_interval,
            verbose=config.verbose,
        )

    return policy_state, history


def expected_sibson_under_policy(
    rng_key: Array,
    policy: Policy,
    policy_state: TrainState,
    num_samples: int,
    num_outer_samples: int = num_outer,
    num_inner_samples: int = num_inner,
    order: float = alpha,
) -> Array:
    """Estimate E_{xi~pi}[Sibson alpha-MI] via Monte Carlo."""
    sample_key, eval_key = random.split(rng_key)

    sample_keys = random.split(sample_key, num_samples)
    designs = jax.vmap(lambda k: policy.sample(k, policy_state.params))(sample_keys)

    eval_keys = random.split(eval_key, num_samples)
    mi_values = jax.vmap(
        lambda k, x: estimate_sibson_alpha_mi(
            k,
            x,
            num_outer_samples=num_outer_samples,
            num_inner_samples=num_inner_samples,
            order=order,
        )
    )(eval_keys, designs)

    return jnp.mean(mi_values)


if __name__ == "__main__":
    rng_key = random.PRNGKey(0)
    run_keys = random.split(rng_key, num_runs)

    run_means = []
    run_stds = []
    run_success = []
    run_expected_mi = []

    for key in run_keys:
        init_key, opt_key, eval_key = random.split(key, 3)

        params_module = GaussianParams(
            dim=1,
            init_mean=lambda k, s: jnp.zeros((1,)),
            init_log_std=lambda k, s: jnp.zeros((1,)),
        )
        policy = ParametricGaussian(params_module, bijector=Tanh())
        policy_state = policy.init_state(init_key, optimizer)

        policy_state, _ = optimize_policy_with_penalty(
            rng_key=opt_key,
            policy=policy,
            policy_state=policy_state,
            config=reps_config,
            entropy_weight=lambda_reg,
        )

        eval_key, sample_key, expected_key = random.split(eval_key, 3)
        sample_keys = random.split(sample_key, reps_config.num_samples)
        samples = jax.vmap(lambda k: policy.sample(k, policy_state.params))(sample_keys)
        samples = samples.reshape(-1)

        closest_opt = jnp.where(samples >= 0, 1.0, -1.0)
        distances = jnp.abs(samples - closest_opt)

        run_means.append(jnp.mean(samples))
        run_stds.append(jnp.std(samples))
        run_success.append(jnp.mean(distances < success_tol))

        expected_mi = expected_sibson_under_policy(
            rng_key=expected_key,
            policy=policy,
            policy_state=policy_state,
            num_samples=num_policy_samples,
        )
        run_expected_mi.append(expected_mi)

    run_means = jnp.array(run_means)
    run_stds = jnp.array(run_stds)
    run_success = jnp.array(run_success)
    run_expected_mi = jnp.array(run_expected_mi)

    print("=" * 80)
    print("NESTED MONTE CARLO SIBSON POLICY OPTIMIZATION (REPS)")
    print("=" * 80)
    print(f"alpha: {alpha}")
    print(f"num_outer: {num_outer}  num_inner: {num_inner}")
    print(f"num_steps: {num_steps}  learning_rate: {learning_rate}")
    print(f"lambda_reg: {lambda_reg}")
    print(f"num_policy_samples: {num_policy_samples}")
    print(f"num_runs: {num_runs}")
    print()

    print(f"Mean robust EIG policy: {float(jnp.mean(run_expected_mi)):+.6f}")
    print(f"Std robust EIG under policy: {float(jnp.std(run_expected_mi)):.6f}")

    print("=" * 80)
    print("DONE")
    print("=" * 80)
