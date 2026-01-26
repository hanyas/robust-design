"""
Comparative Evaluation: Enumeration vs. Discrete REPS for A/B Testing
"""
from typing import Tuple, Callable
import csv

import jax
import optax

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from rdesign.distributions import Beta
from rdesign.ab_testing import ABTesting, Binomial

from rdesign.base import REPSConfig, Policy
from rdesign.policy import Categorical
from rdesign.utils import nested_mc_sibson_mi
from rdesign.reps import optimize_policy

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. Problem Setup
# =============================================================================

NUM_PARTICIPANTS = 100
ALPHA = 1e-1
LAMBDA = 1e6

# Models: Beta-Binomial
prior = Beta(
    delta=jnp.array([2., 50.0]),
    gamma=jnp.array([2., 50.0])
)
lklhd = Binomial(num_participants=NUM_PARTICIPANTS)
model = ABTesting(prior=prior, likelihood=lklhd)

# Estimator Settings
NUM_OUTER = 16
NUM_INNER = 16

# Evaluation Settings
NUM_TRIALS_ENUM = 1024   # Number of independent enumeration runs
NUM_SAMPLES_REPS = 1024  # Number of samples from the policy


# =============================================================================
# 3. Deterministic Optimizer 
# =============================================================================

def run_enumeration(num_trials: int, seed: int = 42):
    """
    Runs multiple trials of enumeration. In each trial, finds the allocation
    that maximizes the stochastic MI estimate across all possibilities.
    """
    print(f"\n[Enumeration] Running {num_trials} enumeration trials...")

    designs = jnp.arange(NUM_PARTICIPANTS + 1, dtype=jnp.float64)

    def find_best_single(rng_key):        
        # Vectorize objective evaluation over all designs
        keys = jr.split(rng_key, NUM_PARTICIPANTS + 1)
        
        vals = jax.vmap(lambda k, d: nested_mc_sibson_mi(
            rng_key=k,
            model=model,
            design=d,
            num_outer=NUM_OUTER,
            num_inner=NUM_INNER,
            alpha=ALPHA
        ))(keys, designs)
        
        return designs[jnp.argmax(vals)]

    trial_keys = jr.split(jr.PRNGKey(seed), num_trials)
    return jax.jit(jax.vmap(find_best_single))(trial_keys)


# =============================================================================
# 4. Stochastic Policy Optimizer (REPS)
# =============================================================================

def run_policy_search(seed: int = 0):
    print(f"\n[REPS] Training discrete policy...")

    policy = Categorical(num_options=NUM_PARTICIPANTS + 1)

    config = REPSConfig(
        num_samples=256,
        batch_size=32,
        max_iterations=1000,
        log_interval=200,
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

    optimizer = optax.adam(learning_rate=1e-1)
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
# 5. Main Comparison Logic
# =============================================================================

print("\n" + "="*80)
print("AB TESTING OPTIMIZATION COMPARISON")
print("="*80)

# 0. Ground Truth (High precision enumeration)
print("\nComputing ground truth...")
gt_designs = jnp.arange(NUM_PARTICIPANTS + 1, dtype=jnp.float64)
gt_vals = jax.vmap(lambda d: model.sibson_mi(d, ALPHA))(gt_designs)

true_opt_idx = jnp.argmax(gt_vals)
true_opt_design = gt_designs[true_opt_idx]
true_max_val = gt_vals[true_opt_idx]
print(f"True Optimal Design: n_A = {true_opt_design} | MI = {true_max_val:.4f}")

def compute_regret(designs):
    """Compute regret against true optimal value."""
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

# A. Run Enumeration
enum_designs = run_enumeration(NUM_TRIALS_ENUM)
enum_regrets = compute_regret(enum_designs)

# B. Run REPS
policy, policy_state = run_policy_search()

# Sample from the trained policy
sample_key = jr.PRNGKey(42)
sample_keys = jr.split(sample_key, NUM_SAMPLES_REPS)
reps_designs = jax.vmap(lambda k: policy.sample(k, policy_state.params))(sample_keys)
reps_regrets = compute_regret(reps_designs)

# C. Statistics
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

enum_saturation = enum_designs / NUM_PARTICIPANTS
reps_saturation = reps_designs / NUM_PARTICIPANTS

print(f"Enumeration | Mean Regret: {jnp.mean(enum_regrets):.4f} | Std: {jnp.std(enum_regrets):.4f} | Worst: {jnp.max(enum_regrets):.4f}")
print(f"REPS        | Mean Regret: {jnp.mean(reps_regrets):.4f} | Std: {jnp.std(reps_regrets):.4f} | Worst: {jnp.max(reps_regrets):.4f}")

print(f"Enumeration | Mean Saturation: {jnp.mean(enum_saturation):.4f}")
print(f"REPS        | Mean Saturation: {jnp.mean(reps_saturation):.4f}")

# D. Save Histograms to CSV

# 1. Regret Histogram
regret_bins = jnp.linspace(0, float(jnp.maximum(jnp.max(enum_regrets), jnp.max(reps_regrets))), 20)
enum_hist, _ = jnp.histogram(enum_regrets, bins=regret_bins)
reps_hist, _ = jnp.histogram(reps_regrets, bins=regret_bins)
regret_bin_centers = (regret_bins[:-1] + regret_bins[1:]) / 2

save_histogram_csv(
    "examples/ab_testing/data/abtesting_regret.csv",
    regret_bin_centers, enum_hist, reps_hist,
    ["bin_center", "enum_density", "reps_density"]
)

# 2. Saturation Histogram
sat_bins = jnp.linspace(0, 1.0, NUM_PARTICIPANTS + 1)
enum_sat_hist, _ = jnp.histogram(enum_saturation, bins=sat_bins)
reps_sat_hist, _ = jnp.histogram(reps_saturation, bins=sat_bins)
sat_bin_centers = (sat_bins[:-1] + sat_bins[1:]) / 2

save_histogram_csv(
    "examples/ab_testing/data/abtesting_saturation.csv",
    sat_bin_centers, enum_sat_hist, reps_sat_hist,
    ["bin_center", "enum_density", "reps_density"]
)

# E. Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Design Distribution
axes[0].hist(enum_designs, bins=jnp.arange(NUM_PARTICIPANTS + 2) - 0.5, alpha=0.6, label='Enumeration', color='salmon')
axes[0].hist(reps_designs, bins=jnp.arange(NUM_PARTICIPANTS + 2) - 0.5, alpha=0.6, label='REPS', color='skyblue')
axes[0].axvline(true_opt_design, color='green', linestyle='--', linewidth=2, label=f'True Opt ({true_opt_design})')
axes[0].set_xlabel('Design (n_A)')
axes[0].set_ylabel('Count')
axes[0].set_title('Selected Design Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Regret Distribution
max_regret = float(jnp.maximum(jnp.max(enum_regrets), jnp.max(reps_regrets)))
regret_bins = jnp.linspace(0, max_regret, 30)
axes[1].hist(enum_regrets, bins=regret_bins, alpha=0.6, label='Enumeration', color='salmon')
axes[1].hist(reps_regrets, bins=regret_bins, alpha=0.6, label='REPS', color='skyblue')
axes[1].set_xlabel('Regret')
axes[1].set_ylabel('Count')
axes[1].set_title('Regret Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
