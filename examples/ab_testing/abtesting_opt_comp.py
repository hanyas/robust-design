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
from rdesign.utils import nested_mc_sibson_mi_with_marginal as nested_mc_sibson_mi
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
prior = Beta(delta=jnp.array([2.0, 50.0]), gamma=jnp.array([2.0, 50.0]))
lklhd = Binomial(num_participants=NUM_PARTICIPANTS)
model = ABTesting(prior=prior, likelihood=lklhd)

# Estimator Settings
NUM_OUTER = 16
NUM_INNER = 16

# Evaluation Settings
NUM_TRIALS_ENUM = 1024  # Number of independent enumeration runs
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

        vals = jax.vmap(
            lambda k, d: nested_mc_sibson_mi(
                rng_key=k, model=model, design=d, num_outer=NUM_OUTER, num_inner=NUM_INNER, alpha=ALPHA
            )
        )(keys, designs)

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
        num_samples=256, batch_size=32, max_iterations=2500, log_interval=200, verbose=True, lmbda=LAMBDA
    )

    def objective_fn(key, design, log_prob, lmbda):
        mi = nested_mc_sibson_mi(
            rng_key=key, model=model, design=design, num_outer=NUM_OUTER, num_inner=NUM_INNER, alpha=ALPHA
        )
        return mi - (1.0 / lmbda) * log_prob

    rng_key = jr.PRNGKey(seed)
    init_key, run_key = jr.split(rng_key)

    optimizer = optax.adam(learning_rate=1e-1)
    state = policy.init_state(init_key, optimizer)

    result = optimize_policy(
        rng_key=run_key, func=objective_fn, policy=policy, state=state, config=config
    )

    return policy, result.state


# =============================================================================
# 5. Main Comparison Logic
# =============================================================================

print("\n" + "=" * 80)
print("AB TESTING OPTIMIZATION COMPARISON")
print("=" * 80)

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
policy, state = run_policy_search()

# Sample from the trained policy
sample_key = jr.PRNGKey(42)
sample_keys = jr.split(sample_key, NUM_SAMPLES_REPS)
reps_designs = jax.vmap(lambda k: policy.sample(k, state.params))(sample_keys)
reps_regrets = compute_regret(reps_designs)


# TV similarity between the trial allocation [n_A, n_B] / N and the
# optimal allocation [n_A*, n_B*] / N. For a 2-arm split this simplifies to
# 1 - |n_A - n_A*| / N.
def tv_similarity(designs: Array) -> Array:
    return 1.0 - jnp.abs(designs - true_opt_design) / NUM_PARTICIPANTS


enum_similarity = tv_similarity(enum_designs)
reps_similarity = tv_similarity(reps_designs)

# C. Statistics
print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

print(
    f"Enumeration | Mean Regret: {jnp.mean(enum_regrets):.4f} | Std: {jnp.std(enum_regrets):.4f} | Worst: {jnp.max(enum_regrets):.4f}"
)
print(
    f"REPS        | Mean Regret: {jnp.mean(reps_regrets):.4f} | Std: {jnp.std(reps_regrets):.4f} | Worst: {jnp.max(reps_regrets):.4f}"
)

print(f"Enumeration | Mean Similarity: {jnp.mean(enum_similarity):.4f}")
print(f"REPS        | Mean Similarity: {jnp.mean(reps_similarity):.4f}")

# D. Save Histograms to CSV

NUM_BIN_EDGES = 30

# 1. Regret Histogram
regret_bins = jnp.linspace(0.0, 0.3, NUM_BIN_EDGES)
enum_hist, _ = jnp.histogram(enum_regrets, bins=regret_bins)
reps_hist, _ = jnp.histogram(reps_regrets, bins=regret_bins)
regret_bin_centers = (regret_bins[:-1] + regret_bins[1:]) / 2

save_histogram_csv(
    "examples/ab_testing/data/abtesting_regret.csv",
    regret_bin_centers,
    enum_hist,
    reps_hist,
    ["bin_center", "enum_density", "reps_density"],
)

# 2. Similarity Histogram
sim_bins = jnp.linspace(0.7, 1.0, NUM_BIN_EDGES)
enum_sim_hist, _ = jnp.histogram(enum_similarity, bins=sim_bins)
reps_sim_hist, _ = jnp.histogram(reps_similarity, bins=sim_bins)
sim_bin_centers = (sim_bins[:-1] + sim_bins[1:]) / 2

save_histogram_csv(
    "examples/ab_testing/data/abtesting_similarity.csv",
    sim_bin_centers,
    enum_sim_hist,
    reps_sim_hist,
    ["bin_center", "enum_density", "reps_density"],
)

# E. Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Regret
axes[0].bar(
    regret_bin_centers,
    enum_hist,
    width=regret_bins[1] - regret_bins[0],
    alpha=0.6,
    label="Enumeration",
    color="0.65",
    edgecolor="black",
)
axes[0].bar(
    regret_bin_centers,
    reps_hist,
    width=regret_bins[1] - regret_bins[0],
    alpha=0.6,
    label="REPS",
    color="0.15",
    edgecolor="black",
)
axes[0].set_xlabel("Regret")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Regret Distribution")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Similarity
axes[1].bar(
    sim_bin_centers,
    enum_sim_hist,
    width=sim_bins[1] - sim_bins[0],
    alpha=0.6,
    label="Enumeration",
    color="0.65",
    edgecolor="black",
)
axes[1].bar(
    sim_bin_centers,
    reps_sim_hist,
    width=sim_bins[1] - sim_bins[0],
    alpha=0.6,
    label="REPS",
    color="0.15",
    edgecolor="black",
)
axes[1].set_xlabel(r"TV Similarity: $1 - |n_A - n_A^\star| / N$")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Design Similarity to Optimal Allocation")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
