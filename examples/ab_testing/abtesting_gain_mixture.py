"""
Robust Experimental Design: Under-Promise, Over-Deliver Assessment (A/B Testing)
"""

from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt
import csv

from rdesign.distributions import Beta
from rdesign.ab_testing import ABTesting, Binomial

jax.config.update("jax_enable_x64", True)


class BetaMixtureBinomial(NamedTuple):
    """
    Beta-mixture Binomial true process (matches ``abtesting_coverage_mixture.py``).

    For each group, p ~ Beta(kappa * theta, kappa * (1 - theta)) and then
    x ~ Binomial(n, p). Smaller kappa values induce stronger overdispersion.
    """

    num_participants: int
    mixture_concentration: float = 10.0

    def get_trials(self, design: Array) -> Array:
        """Get trial counts [n_A, n_B] from design (n_A)."""
        n_a = design
        n_b = self.num_participants - design
        return jnp.array([n_a, n_b], dtype=jnp.float32)

    def sample(self, rng_key: Array, design: Array, theta_true: Array) -> Array:
        trials = self.get_trials(design)
        keys = jr.split(rng_key, 2)

        def sample_group(key, n, theta):
            key_rate, key_obs = jr.split(key)
            theta = jnp.clip(theta, 1e-6, 1.0 - 1e-6)
            alpha = jnp.maximum(self.mixture_concentration * theta, 1e-6)
            beta = jnp.maximum(self.mixture_concentration * (1.0 - theta), 1e-6)
            latent_rate = jr.beta(key_rate, alpha, beta)
            return jr.binomial(key_obs, n.astype(int), latent_rate).astype(jnp.float32)

        return jax.vmap(sample_group)(keys, trials, theta_true)


# ============================================================================
# Configuration
# ============================================================================

rng_key = jr.PRNGKey(42)

# Parameters
num_participants = 100
num_trials = 10_000

# Alpha settings (Misspecification order)
alpha_nominal = 1.0
alpha_robust = 0.056

# True Process (Beta-mixture Binomial) — matches abtesting_coverage_mixture.py
mixture_concentration = 10.0
proc_prior = Beta(
    delta=jnp.array([100.0, 1.0]),
    gamma=jnp.array([100.0, 1.0]),
)
proc_lklhd = BetaMixtureBinomial(
    num_participants=num_participants,
    mixture_concentration=mixture_concentration,
)

# Model (Beta-Binomial)
mdl_prior = Beta(delta=jnp.array([10.0, 1.0]), gamma=jnp.array([10.0, 1.0]))
mdl_lklhd = Binomial(num_participants=num_participants)
model = ABTesting(prior=mdl_prior, likelihood=mdl_lklhd)

# ============================================================================
# Run Simulation
# ============================================================================

print(f"\nRunning {num_trials} trials...")


def run_trial(key):
    """Single trial: random design, generate data, compute nominal and robust IG."""
    design_key, prior_key, process_key = jr.split(key, 3)

    # Random design (uniform allocation to group A) per trial
    xi = jr.randint(design_key, shape=(), minval=0, maxval=num_participants + 1)

    # 1. Sample True Parameter from Process Prior
    theta = proc_prior.sample(prior_key, 1)[0]

    # 2. Generate Data from True Process (Beta-mixture Binomial)
    y = proc_lklhd.sample(process_key, xi, theta)

    # Nominal: KL divergence using standard Bayes posterior (alpha=1.0)
    bayes_posterior = model.get_bayes_posterior(y, xi)
    ig_nominal = bayes_posterior.kl_to(model.prior)

    # Robust: Rényi IG using nominal model with robust alpha
    ig_robust = model.renyi_ig(y, xi, alpha_robust)

    return jnp.array([ig_nominal, ig_robust])


keys = jr.split(rng_key, num_trials)
results = jax.jit(jax.vmap(run_trial))(keys)

dist_nominal = results[:, 0]
dist_robust = results[:, 1]

# Aggregation
avg_nominal = jnp.mean(dist_nominal)
avg_robust = jnp.mean(dist_robust)

print("✓ Complete")

# ============================================================================
# Results
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS (A/B Testing)")
print("=" * 80)
print(f"Nominal  (perceived utility): {avg_nominal:.4f} nats")
print(f"Robust   (perceived utility): {avg_robust:.4f} nats")
print("-" * 80)

# ============================================================================
# Visualization
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Histogram bins — shared range across both distributions, clipped at the
# 99th percentile so a handful of mixture tail outliers don't blow out
# the bin range.
all_data = jnp.concatenate([dist_nominal, dist_robust])
lo = float(all_data.min())
hi = float(jnp.percentile(all_data, 99))
bins = jnp.linspace(lo, hi, 19)  # 19 edges => 18 bins

# Nominal distribution
hist_nominal, bin_edges_nominal, patches_nominal = ax.hist(
    dist_nominal,
    bins=bins,
    color="0.65",
    alpha=0.7,
    edgecolor="black",
    linewidth=0.5,
    align="right",
    label=f"Nominal (Shannon IG)",
)

# Robust distribution
hist_robust, bin_edges_robust, patches_robust = ax.hist(
    dist_robust,
    bins=bins,
    align="right",
    color="0.15",
    alpha=0.7,
    edgecolor="black",
    linewidth=0.5,
    label=f"Robust (Rényi IG, α={alpha_robust})",
)

# Bin centers for plotting
bin_centers = (bin_edges_nominal[:-1] + bin_edges_nominal[1:]) / 2

# Save to CSV
csv_path = "examples/ab_testing/data/abtesting_gain_mixture.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["bin_center", "nominal_count", "robust_count"])
    for bc, nom, rob in zip(bin_centers, hist_nominal, hist_robust):
        writer.writerow([bc, nom, rob])

print(f"\nHistogram data saved to {csv_path} ({len(bin_centers)} bins)")

ax.set_xlabel("Information Gain (nats)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_xlim(lo, hi)
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
