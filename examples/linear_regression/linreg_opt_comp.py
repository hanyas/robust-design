"""
Comparative Evaluation: Deterministic Gradient Descent vs. PAC-Bayes REPS
"""
from typing import Tuple
import csv

import jax
import optax

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian
from rdesign.linear_regression import LinearRegression

from rdesign.base import REPSConfig
from rdesign.utils import nested_mc_sibson_mi_with_marginal as nested_mc_sibson_mi
from rdesign.policy import BoxGaussian
from rdesign.reps import optimize_policy
from rdesign.bijectors import Tanh

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class HighDimLinearGaussian(LinearGaussian):

    def design_matrix(self, design: Array) -> Array:
        return design.reshape(1, -1)


# Problem Dimensions
DIM = 10
NOISE_STDDEV = 1e-1
ALPHA = 1e-1
LAMBDA = 1e6

vars = jnp.linspace(1.0, 10.0, DIM)
prior = Gaussian(mean=jnp.zeros(DIM), cov=jnp.diag(vars))
lklhd = HighDimLinearGaussian(noise_stddev=NOISE_STDDEV)
model = LinearRegression(prior=prior, likelihood=lklhd)

# Estimator Settings
NUM_OUTER = 16
NUM_INNER = 16

# Evaluation Settings
NUM_TRIALS_GD = 128  # Number of independent GD runs
NUM_SAMPLES_REPS = 128  # Number of samples from the policy

# =============================================================================
# 2. Deterministic Optimizer (Gradient Descent)
# =============================================================================

def run_gradient_descent(num_trials: int, seed: int = 42):
    """
    Runs multiple independent trials of Adam on the noisy objective.
    """
    print(f"\n[Deterministic] Running {num_trials} independent GD trials...")

    optimizer = optax.adam(learning_rate=1e-2)
    bijector = Tanh()

    def optimize_single(rng_key):
        init_key, opt_key = jr.split(rng_key)
        params = jr.normal(init_key, (DIM,))
        opt_state = optimizer.init(params)

        def objective_fn(p, key):
            xi = bijector.forward(p)
            return -nested_mc_sibson_mi(
                rng_key=key,
                model=model,
                design=xi,
                num_outer=NUM_OUTER,
                num_inner=NUM_INNER,
                alpha=ALPHA
            )

        def step(carry, i):
            p, state, k = carry
            obj_key, next_key = jr.split(k)
            val, grads = jax.value_and_grad(objective_fn)(p, obj_key)
            updates, new_state = optimizer.update(grads, state)
            new_p = optax.apply_updates(p, updates)
            return (new_p, new_state, next_key), val

        (final_p, _, _), _ = jax.lax.scan(
            step, (params, opt_state, opt_key), jnp.arange(1000)
        )
        return bijector.forward(final_p)

    keys = jr.split(jr.PRNGKey(seed), num_trials)
    return jax.jit(jax.vmap(optimize_single))(keys)

# =============================================================================
# 3. Stochastic Policy Optimizer
# =============================================================================

def run_policy_search(seed: int = 0):
    """
    Trains a single stochastic policy using REPS.
    """
    print(f"\n[Stochastic] Training single REPS policy...")

    policy = BoxGaussian(
        dim=DIM,
        bijector=Tanh(),
        init_mean=lambda k, s: jr.normal(k, s) * 0.1,
        init_log_std=lambda k, s: jnp.zeros(s)
    )

    config = REPSConfig(
        num_samples=256,
        batch_size=32,
        max_iterations=2500,
        log_interval=100,
        verbose=True,
        lmbda=LAMBDA
    )

    def objective_fn(key, design, log_prob, lmbda):
        mi = nested_mc_sibson_mi(
            rng_key=key,
            model=model,
            design=design,
            num_outer=NUM_OUTER,
            num_inner=NUM_INNER,
            alpha=ALPHA
        )
        return mi - (1.0 / lmbda) * log_prob

    rng_key = jr.PRNGKey(seed)
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

    return policy, result.policy_state

# =============================================================================
# 4. Main Comparison Logic
# =============================================================================

def compute_regret(designs):
    """Compute regret against true optimal design."""
    true_opt = jnp.ones(DIM)
    true_max_val = model.sibson_mi(true_opt, ALPHA)
    found_vals = jax.vmap(lambda d: model.sibson_mi(d, ALPHA))(designs)

    return true_max_val - found_vals


def save_histogram_csv(filename, bin_centers, count1, count2, header):
    """Helper to save histogram data to CSV."""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for bc, c1, c2 in zip(bin_centers, count1, count2):
            writer.writerow([bc, c1, c2])
    print(f"Data saved to {filename}")


# A. Run Deterministic Optimizer
gd_designs = run_gradient_descent(NUM_TRIALS_GD)
gd_regrets = compute_regret(gd_designs)
gd_saturation = jnp.mean(jnp.abs(gd_designs), axis=1)

# B. Run Stochastic Optimizer
policy, policy_state = run_policy_search()

# Sample from the trained policy to simulate "Deployment Risk"
sample_key = jr.PRNGKey(42)
sample_keys = jr.split(sample_key, NUM_SAMPLES_REPS)
reps_designs = jax.vmap(lambda k: policy.sample(k, policy_state.params))(sample_keys)
reps_regrets = compute_regret(reps_designs)
reps_saturation = jnp.mean(jnp.abs(reps_designs), axis=1)

# C. Statistics
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

print(f"Deterministic (GD) | Mean Regret: {jnp.mean(gd_regrets):.4f} | Std: {jnp.std(gd_regrets):.4f} | Worst: {jnp.max(gd_regrets):.4f}")
print(f"Stochastic (REPS)  | Mean Regret: {jnp.mean(reps_regrets):.4f} | Std: {jnp.std(reps_regrets):.4f} | Worst: {jnp.max(reps_regrets):.4f}")

print(f"Deterministic (GD) | Mean Saturation: {jnp.mean(gd_saturation):.4f}")
print(f"Stochastic (REPS)  | Mean Saturation: {jnp.mean(reps_saturation):.4f}")

# D. Save Histograms to CSV

# 1. Regret Histogram
regret_bins = jnp.linspace(0, jnp.maximum(jnp.max(gd_regrets), jnp.max(reps_regrets)), 30)
gd_hist, _ = jnp.histogram(gd_regrets, bins=regret_bins)
reps_hist, _ = jnp.histogram(reps_regrets, bins=regret_bins)
regret_bin_centers = (regret_bins[:-1] + regret_bins[1:]) / 2

save_histogram_csv(
    "./data/linreg_regret.csv",
    regret_bin_centers, gd_hist, reps_hist,
    ["bin_center", "gd_density", "reps_density"]
)

# 2. Saturation Histogram
sat_bins = jnp.linspace(0, 1.0, 45)
gd_sat_hist, _ = jnp.histogram(gd_saturation, bins=sat_bins)
reps_sat_hist, _ = jnp.histogram(reps_saturation, bins=sat_bins)
sat_bin_centers = (sat_bins[:-1] + sat_bins[1:]) / 2

save_histogram_csv(
    "./data/linreg_saturation.csv",
    sat_bin_centers, gd_sat_hist, reps_sat_hist,
    ["bin_center", "gd_density", "reps_density"]
)

# E. Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Regret
axes[0].bar(regret_bin_centers, gd_hist, width=regret_bins[1]-regret_bins[0], alpha=0.6, label='Deterministic (GD)', color='0.65', edgecolor='black')
axes[0].bar(regret_bin_centers, reps_hist, width=regret_bins[1]-regret_bins[0], alpha=0.6, label='Stochastic Policy (REPS)', color='0.15', edgecolor='black')

# Add mean lines
axes[0].axvline(jnp.mean(gd_regrets), color='0.15', linestyle='--', linewidth=2, label=f'Mean GD ({jnp.mean(gd_regrets):.2f})')
axes[0].axvline(jnp.mean(reps_regrets), color='0.15', linestyle='-', linewidth=2, label=f'Mean REPS ({jnp.mean(reps_regrets):.2f})')

axes[0].set_xlabel('Regret')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Regret Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Saturation
axes[1].bar(sat_bin_centers, gd_sat_hist, width=sat_bins[1]-sat_bins[0], alpha=0.6, label='Deterministic (GD)', color='0.65', edgecolor='black')
axes[1].bar(sat_bin_centers, reps_sat_hist, width=sat_bins[1]-sat_bins[0], alpha=0.6, label='Stochastic Policy (REPS)', color='0.15', edgecolor='black')

axes[1].axvline(jnp.mean(gd_saturation), color='0.15', linestyle='--', linewidth=2, label=f'Mean GD ({jnp.mean(gd_saturation):.2f})')
axes[1].axvline(jnp.mean(reps_saturation), color='0.15', linestyle='-', linewidth=2, label=f'Mean REPS ({jnp.mean(reps_saturation):.2f})')

axes[1].set_xlabel('Mean Absolute Design Value (Saturation)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Constraint Saturation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
