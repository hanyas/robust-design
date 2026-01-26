"""
Robust Experimental Design: Under-Promise, Over-Deliver Assessment (A/B Testing)
"""
from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln

import matplotlib.pyplot as plt
import csv

from rdesign.distributions import Beta
from rdesign.ab_testing import ABTesting, Binomial

jax.config.update("jax_enable_x64", True)


class TiltedBinomial(NamedTuple):
    """
    Generates data from the α-tilted Binomial likelihood p_α(x | θ).
    Used as the "True Process" for simulation.
    """
    num_participants: int
    alpha: float = 1.0

    def get_trials(self, design: Array) -> Array:
        """Get trial counts [n_A, n_B] from design (n_A)."""
        n_a = design
        n_b = self.num_participants - design
        return jnp.array([n_a, n_b], dtype=jnp.float32)

    def _log_tilted_prob(self, k: Array, n: Array, theta: Array) -> Array:
        """
        Computes unnormalized log probability: α * log(Binomial PMF)
        """
        # Safe masking for support [0, n]
        mask = (k >= 0) & (k <= n)
        safe_k = jnp.where(mask, k, 0.0)
        safe_nmk = jnp.where(mask, n - safe_k, 0.0)

        # Log Binomial Coefficient: log(nCx)
        log_comb = gammaln(n + 1) - gammaln(safe_k + 1) - gammaln(safe_nmk + 1)

        # Log Likelihood: x log(θ) + (n-x) log(1-θ)
        theta = jnp.clip(theta, 1e-6, 1.0 - 1e-6)
        log_lklhd = safe_k * jnp.log(theta) + safe_nmk * jnp.log(1.0 - theta)

        # Tilted Log Prob: α * (log_comb + log_lklhd)
        return jnp.where(mask, self.alpha * (log_comb + log_lklhd), -jnp.inf)

    def sample(self, rng_key: Array, design: Array, theta_true: Array) -> Array:
        """
        Sample observations x from the tilted likelihood.
        """
        trials = self.get_trials(design)

        # Grid of all possible outcomes [0, 1, ..., N]
        k_grid = jnp.arange(self.num_participants + 1, dtype=jnp.float32)

        def sample_group(key, n, theta):
            # 1. Compute unnormalized logits for all k
            logits = self._log_tilted_prob(k_grid, n, theta)
            # 2. Sample from Categorical (handles normalization internally)
            return jr.categorical(key, logits).astype(jnp.float32)

        # Vectorize sampling over the two groups (A and B)
        keys = jr.split(rng_key, 2)
        return jax.vmap(sample_group)(keys, trials, theta_true)


# ============================================================================
# Configuration
# ============================================================================

rng_key = jr.PRNGKey(42)

# Parameters
num_participants = 2500
num_trials = 10_000

# Alpha settings (Misspecification order)
alpha_nominal = 1.0
alpha_true = 1e-1
alpha_robust = 1e-2

# True Process (Tilted Binomial)
proc_prior = Beta(
    delta=jnp.array([1, 5000.0]),
    gamma=jnp.array([1, 5000.0])
)
proc_lklhd = TiltedBinomial(
    num_participants=num_participants,
    alpha=alpha_true
)

# Model (Beta-Binomial)
mdl_prior = Beta(
    delta=jnp.array([2., 50.0]),
    gamma=jnp.array([2., 50.0])
)
mdl_lklhd = Binomial(num_participants=num_participants)
model = ABTesting(prior=mdl_prior, likelihood=mdl_lklhd)

# Design
xi = model.optimal_design_sibson(alpha_robust)

# ============================================================================
# Calculate the Promise (Robust EIG under nominal model)
# ============================================================================

# Robust EIG under nominal model
promise_eig = model.sibson_mi(xi, alpha_robust)

print(f"\nTHE GUARANTEE (Promise)")
print(f"  Metric: Robust EIG (Sibson MI with α={alpha_robust})")
print(f"  Value:  {promise_eig:.4f} nats")

# ============================================================================
# Run Simulation
# ============================================================================

print(f"\nRunning {num_trials} trials...")

def run_trial(key):
    """Single trial: generate data from true process, compute oracle and subjective IG."""
    prior_key, process_key = jr.split(key)

    # 1. Sample True Parameter from Process Prior
    theta = proc_prior.sample(prior_key)

    # 2. Generate Data from True Process (Tilted Likelihood)
    y = proc_lklhd.sample(process_key, xi, theta)

    # Oracle: KL divergence using tilted posterior with the TRUE alpha
    oracle_posterior = model.get_tilted_posterior(y, xi, alpha_true)
    ig_oracle = oracle_posterior.kl_to(model.prior)

    # Nominal: KL divergence using standard Bayes posterior (alpha=1.0)
    bayes_posterior = model.get_bayes_posterior(y, xi)
    ig_nominal = bayes_posterior.kl_to(model.prior)

    # Robust: Rényi IG using nominal model with robust alpha
    ig_robust = model.renyi_ig(y, xi, alpha_robust)

    return jnp.array([ig_oracle, ig_nominal, ig_robust])

keys = jr.split(rng_key, num_trials)
results = jax.jit(jax.vmap(run_trial))(keys)

dist_oracle = results[:, 0]
dist_nominal = results[:, 1]
dist_robust = results[:, 2]

# Aggregation
avg_oracle = jnp.mean(dist_oracle)
avg_nominal = jnp.mean(dist_nominal)
avg_robust = jnp.mean(dist_robust)

print("✓ Complete")

# ============================================================================
# Results
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS (A/B Testing)")
print("=" * 80)
print(f"Promise  (guarantee):           {promise_eig:.4f} nats")
print(f"Oracle   (true utility):        {avg_oracle:.4f} nats")
print(f"Nominal  (perceived utility): {avg_nominal:.4f} nats")
print(f"Robust   (perceived utility): {avg_robust:.4f} nats")
print("-" * 80)
print(f"Objective gap  (Oracle - Promise):    {avg_oracle - promise_eig:+.4f} nats")
print(f"Nominal gap    (Oracle - Perceived):  {avg_oracle - avg_nominal:+.4f} nats")
print(f"Robust gap     (Oracle - Perceived):  {avg_oracle - avg_robust:+.4f} nats")
print("=" * 80)

# ============================================================================
# Visualization
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Histogram bins
all_data = jnp.concatenate([dist_oracle, dist_nominal, dist_robust])
bins = jnp.linspace(all_data.min(), all_data.max(), 45)

# # Oracle distribution
# hist_oracle, bin_edges_oracle, patches_oracle = ax.hist(
#     dist_oracle, bins=bins, color='0.15', alpha=0.7,
#     align='right', edgecolor='black', linewidth=0.5,
#     label=f'Oracle (Shannon IG)'
# )
# # Add hatching pattern
# for patch in patches_oracle:
#     patch.set_hatch('///')

# Nominal distribution
hist_nominal, bin_edges_nominal, patches_nominal = ax.hist(
    dist_nominal, bins=bins, color='0.65', alpha=0.7,
    edgecolor='black', linewidth=0.5, align='right',
    label=f'Nominal (Shannon IG)'
)

# Robust distribution
hist_robust, bin_edges_robust, patches_robust = ax.hist(
    dist_robust, bins=bins, align='right', color='0.15', alpha=0.7,
    edgecolor='black', linewidth=0.5,
    label=f'Robust (Rényi IG, α={alpha_robust})'
)

# Bin centers for plotting
bin_centers = (bin_edges_nominal[:-1] + bin_edges_nominal[1:]) / 2

# Save to CSV
csv_path = "examples/ab_testing/data/abtesting_infogain.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["bin_center", "nominal_count", "robust_count"])
    for bc, nom, rob in zip(bin_centers, hist_nominal, hist_robust):
        writer.writerow([bc, nom, rob])

print(f"\nHistogram data saved to {csv_path} ({len(bin_centers)} bins)")

# Vertical lines
ax.axvline(promise_eig, color='0.1', linestyle='--', lw=2.5,
           label=f'Sibson MI: {promise_eig:.2f}')

ax.set_xlabel('Information Gain (nats)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()