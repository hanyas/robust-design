"""
Nested Monte Carlo optimization of Sibson's alpha-MI for linear regression using LBFGS.

We use the nested Monte Carlo estimator from rdesign.utils for the robust EIG.
The analytic optimum is at xi in {+1, -1} by symmetry, so we compare against those endpoints.
"""

from typing import Tuple

import jax
from jax import Array, random, numpy as jnp
from jaxopt import LBFGSB

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian
from rdesign.utils import sibson_alpha_mutual_information

jax.config.update("jax_enable_x64", True)

# Model parameters
prior_mean = jnp.zeros(2)
prior_cov = jnp.eye(2)
noise_stddev = 0.3

# Robustness / estimator settings
alpha = 0.95
num_outer = 16
num_inner = 16

# Optimization settings
num_runs = 50
maxiter = 100

prior = Gaussian(mean=prior_mean, cov=prior_cov)
likelihood = LinearGaussian(noise_stddev=noise_stddev)


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
    """Optimize the design using LBFGSB and return (xi, estimated MI)."""
    init_key, opt_key, eval_key = random.split(rng_key, 3)

    # Initial guess: random points between -1 and 1
    x0 = random.uniform(init_key, (1,), minval=-1.0, maxval=1.0)

    def objective(xi, key):
        return -estimate_sibson_alpha_mi(key, xi)

    # LBFGSB from jaxopt with parameters inspired by the example
    optimizer = LBFGSB(
        fun=objective,
        maxiter=maxiter,
        jit=True,
        linesearch="zoom",
        linesearch_init="increase",
        maxls=10,
        tol=1e-3
    )

    # We use a fixed key for the objective to make it deterministic for the optimizer.
    # This is a common trick when using deterministic optimizers on stochastic objectives.
    bounds = (jnp.array([-1.0]), jnp.array([1.0]))

    result = optimizer.run(
        init_params=x0,
        key=opt_key,
        bounds=bounds
    )

    xi = result.params[0]
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
    print("NESTED MONTE CARLO SIBSON OPTIMIZATION (LBFGS)")
    print("=" * 80)
    print(f"alpha: {alpha}")
    print(f"num_outer: {num_outer}  num_inner: {num_inner}")
    print(f"maxiter: {maxiter}")
    print(f"num_runs: {num_runs}")
    print()

    print("Results vs optimal designs {+1, -1}:")
    print(f"Mean robust EIG: {jnp.mean(mi_values):.4f}")
    print(f"Std robust EIG: {jnp.std(mi_values):.4f}")

    print("=" * 80)
    print("DONE")
    print("=" * 80)
