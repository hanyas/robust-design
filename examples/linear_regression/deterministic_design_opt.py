"""
Nested Monte Carlo optimization of Sibson's alpha-MI for linear regression.

We use the nested Monte Carlo estimator from rdesign.utils for the robust EIG.
The analytic optimum is at xi in {+1, -1} by symmetry, so we compare against those endpoints.
"""

from typing import Tuple

import jax
import optax
from jax import Array, random, numpy as jnp

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian
from rdesign.utils import sibson_alpha_mutual_information

jax.config.update("jax_enable_x64", True)

# Model parameters
prior_mean = jnp.zeros(2)
prior_cov = jnp.eye(2)
noise_stddev = 0.1

# Robustness / estimator settings
alpha = 0.9
num_outer = 16
num_inner = 16

# Optimization settings
learning_rate = 5e-2
num_steps = 200
num_runs = 50
success_tol = 1e-2

prior = Gaussian(mean=prior_mean, cov=prior_cov)
likelihood = LinearGaussian(noise_stddev=noise_stddev)
optimizer = optax.adam(learning_rate)


def estimate_sibson_alpha_mi(rng_key: Array, xi: Array) -> Array:
    """Estimate Sibson's alpha-MI with the nested Monte Carlo estimator."""
    design = jnp.atleast_1d(xi)
    return sibson_alpha_mutual_information(
        rng_key,
        design,
        prior,
        likelihood,
        num_outer,
        num_inner,
        alpha,
    )


@jax.jit
def optimize_design(rng_key: Array) -> Tuple[Array, Array]:
    """Optimize the design and return (xi, estimated MI)."""
    init_key, step_key, eval_key = random.split(rng_key, 3)
    params = random.normal(init_key, (1,))
    opt_state = optimizer.init(params)

    def step(state, key):
        params, opt_state = state

        def loss_fn(p):
            xi = jnp.tanh(p)[0]
            return -estimate_sibson_alpha_mi(key, xi)

        loss, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    keys = random.split(step_key, num_steps)
    (params, _), _ = jax.lax.scan(step, (params, opt_state), keys)

    xi = jnp.tanh(params)[0]
    final_mi = estimate_sibson_alpha_mi(eval_key, xi)
    return xi, final_mi


if __name__ == "__main__":
    rng_key = random.key(0)
    run_keys = random.split(rng_key, num_runs)

    designs = []
    mi_values = []
    for key in run_keys:
        xi, mi = optimize_design(key)
        designs.append(xi)
        mi_values.append(mi)

    designs = jnp.array(designs)
    mi_values = jnp.array(mi_values)

    print("=" * 80)
    print("NESTED MONTE CARLO SIBSON OPTIMIZATION")
    print("=" * 80)
    print(f"alpha: {alpha}")
    print(f"num_outer: {num_outer}  num_inner: {num_inner}")
    print(f"num_steps: {num_steps}  learning_rate: {learning_rate}")
    print(f"num_runs: {num_runs}")
    print()

    print("Results vs optimal designs {+1, -1}:")
    print(f"Mean robust EIG: {jnp.mean(mi_values):.4f}")
    print(f"Std robust EIG: {jnp.std(mi_values):.4f}")

    print("=" * 80)
    print("DONE")
    print("=" * 80)
