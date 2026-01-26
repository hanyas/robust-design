"""
High-dimensional linear regression design optimization using REPS.
"""

import jax
import optax

from jax import Array
import jax.numpy as jnp
import jax.random as jr

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian
from rdesign.linear_regression import LinearRegression

from rdesign.base import REPSConfig
from rdesign.utils import nested_mc_sibson_mi
from rdesign.policy import BoxGaussian
from rdesign.reps import optimize_policy
from rdesign.bijectors import Tanh

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class HighDimLinearGaussian(LinearGaussian):

    def design_matrix(self, design: Array) -> Array:
        return design.reshape(1, -1)



# Settings
DIM = 10
NOISE_STDDEV = 1e-1
ALPHA = 0.5
LAMBDA = 1e6

# Estimator settings
NUM_OUTER = 16
NUM_INNER = 16

# Setup Model
vars = jnp.linspace(1.0, 10.0, DIM)
prior = Gaussian(mean=jnp.zeros(DIM), cov=jnp.diag(vars))
lklhd = HighDimLinearGaussian(noise_stddev=NOISE_STDDEV)
model = LinearRegression(prior=prior, likelihood=lklhd)

# Setup Policy using rdesign.policy.BoxGaussian
policy = BoxGaussian(
    dim=DIM,
    bijector=Tanh(),
    init_mean=lambda k, s: jr.normal(k, s) * 0.1,
    init_log_std=lambda k, s: jnp.zeros(s)
)

# Setup REPS
config = REPSConfig(
    num_samples=256,
    batch_size=32,
    max_iterations=1000,
    log_interval=20,
    lmbda=LAMBDA
)

# Define objective function for REPS
def objective_fn(key: Array, design: Array, log_prob: Array, lmbda: float) -> float:
    mi = nested_mc_sibson_mi(
        rng_key=key,
        model=model,
        design=design,
        num_outer=NUM_OUTER,
        num_inner=NUM_INNER,
        alpha=ALPHA
    )
    return mi - (1.0 / lmbda) * log_prob

# Optimization
print(f"Running REPS optimization in {DIM} dimensions...")
rng_key = jr.PRNGKey(0)
init_key, run_key = jr.split(rng_key)

optimizer = optax.adam(learning_rate=1e-2)
policy_state = policy.init_state(init_key, optimizer)

result = optimize_policy(
    rng_key=run_key,
    func=objective_fn,
    policy=policy,
    policy_state=policy_state,
    config=config
)

# --- Analysis ---

# Extract final approximate mode design
final_mu = result.policy_state.params["mean"]
final_design = jnp.tanh(final_mu)

# True Optimum: All ones
true_opt = jnp.ones(DIM)

# 1. Check Saturation
mean_abs_val = jnp.mean(jnp.abs(final_design))
print(f"Mean Absolute Value of Final Design (target -> 1.0): {mean_abs_val:.4f}")

# 2. Value Regret
true_max_val = model.sibson_mi(true_opt, ALPHA)
found_val = model.sibson_mi(final_design, ALPHA)
regret = true_max_val - found_val

print("-" * 30)
print(f"Final Results (Single Run):")
print(f"True Optimum Value: {true_max_val:.4f}")
print(f"Found Value:        {found_val:.4f}")
print(f"Regret:             {regret:.4f}")

# Plot History
hist = result.history
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].plot(hist["value"])
axes[0].set_title("Mean Reward (MI)")
axes[0].set_xlabel("Iteration")

axes[1].plot(hist["temperature"])
axes[1].set_title("Temperature (tau)")
axes[1].set_xlabel("Iteration")

axes[2].plot(hist["entropy"])
axes[2].set_title("Policy Entropy")
axes[2].set_xlabel("Iteration")

# 4. Histogram of Policy Samples
sample_key, _ = jr.split(run_key)
sample_keys = jr.split(sample_key, 1024)
many_samples = jax.vmap(lambda k: policy.sample(k, result.policy_state.params))(sample_keys)

axes[3].hist(many_samples.ravel(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[3].axvline(-1.0, color='r', linestyle='--', linewidth=1)
axes[3].axvline(1.0, color='r', linestyle='--', linewidth=1)
axes[3].set_title("Policy Distribution")
axes[3].set_xlabel("Design Value")

plt.tight_layout()
plt.show()
