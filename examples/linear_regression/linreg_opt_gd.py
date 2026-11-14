"""
High-dimensional linear regression design optimization using gradient descent.
"""

import jax
import optax

from jax import Array
import jax.numpy as jnp
import jax.random as jr

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian
from rdesign.linear_regression import LinearRegression

from rdesign.utils import nested_mc_sibson_mi_with_marginal as nested_mc_sibson_mi
from rdesign.bijectors import Tanh

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class HighDimLinearGaussian(LinearGaussian):

    def design_matrix(self, design: Array) -> Array:
        return design.reshape(1, -1)


# Settings
DIM = 10
ALPHA = 0.1
NOISE_STDDEV = 1e-1

# Estimator settings
NUM_OUTER = 128
NUM_INNER = 32

# Optimization settings
LEARNING_RATE = 1e-2
NUM_ITERS = 1000
NUM_TRIALS = 128

# Create the high-dimensional linear regression model
vars = jnp.linspace(1.0, 10.0, DIM)
prior = Gaussian(mean=jnp.zeros(DIM), cov=jnp.diag(vars))
lklhd = HighDimLinearGaussian(noise_stddev=NOISE_STDDEV)
model = LinearRegression(prior=prior, likelihood=lklhd)

# Optimizer & Bijector
optimizer = optax.adam(learning_rate=LEARNING_RATE)
bijector = Tanh()


def optimize_design(rng_key: Array) -> Array:
    """Optimize the design vector using Adam with Tanh constraint."""
    init_key, opt_key = jr.split(rng_key)

    # Initialize unconstrained parameters randomly
    params = jr.normal(init_key, (DIM,))
    opt_state = optimizer.init(params)

    def objective_fn(p: Array, key: Array) -> float:
        xi = bijector.forward(p)
        return -nested_mc_sibson_mi(
            rng_key=key,
            model=model,
            design=xi,
            num_outer=NUM_OUTER,
            num_inner=NUM_INNER,
            alpha=ALPHA,
        )

    def step(carry, i):
        p, state, key = carry
        obj_key, next_key = jr.split(key)

        val, grads = jax.value_and_grad(objective_fn)(p, obj_key)
        updates, new_state = optimizer.update(grads, state)
        new_p = optax.apply_updates(p, updates)
        return (new_p, new_state, next_key), val

    (final_p, _, _), _ = jax.lax.scan(step, (params, opt_state, opt_key), jnp.arange(NUM_ITERS))

    # Return constrained final design
    final_xi, _ = bijector.forward_and_log_det(final_p)
    return final_xi


print("=" * 80)
print(f"Running {NUM_TRIALS} optimization trials in {DIM} dimensions...")
print("=" * 80)

keys = jr.split(jr.PRNGKey(42), NUM_TRIALS)
found_designs = jax.vmap(optimize_design)(keys)


true_opt = jnp.ones(DIM)

# 1. Check if designs are pushed to boundaries (Mean abs value should be close to 1)
mean_abs_val = jnp.mean(jnp.abs(found_designs))
print(f"Mean Absolute Value of Designs (target -> 1.0): {mean_abs_val:.4f}")

# 2. Check Value Regret
true_max_val = model.sibson_mi(true_opt, ALPHA)
found_vals = jax.vmap(lambda d: model.sibson_mi(d, ALPHA))(found_designs)
regrets = true_max_val - found_vals

print(f"True Optimum Value: {true_max_val:.4f}")
print(f"Mean Found Value:   {jnp.mean(found_vals):.4f}")
print(f"Mean Regret:        {jnp.mean(regrets):.4f}")

# Plot
plt.figure(figsize=(10, 5))

# Subplot 1: Histogram of coordinate values
plt.subplot(1, 2, 1)
plt.hist(found_designs.ravel(), bins=50, color="0.65", edgecolor="black", alpha=0.7)
plt.axvline(-1.0, color="0.15", linestyle="--", linewidth=1)
plt.axvline(1.0, color="0.15", linestyle="--", linewidth=1)
plt.xlabel("Design Coordinate Values")
plt.ylabel("Frequency")
plt.title("Constraint Saturation")

# Subplot 2: Histogram of Regret
plt.subplot(1, 2, 2)
plt.hist(regrets, bins=20, color="0.65", edgecolor="black", alpha=0.7)
plt.axvline(0.0, color="0.15", linestyle="--", linewidth=2, label="Zero Regret")
plt.xlabel("Regret (True Max - Found)")
plt.title("Optimization Shortfall")
plt.legend()

plt.tight_layout()
plt.show()
