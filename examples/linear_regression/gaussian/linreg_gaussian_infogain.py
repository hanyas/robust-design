"""
Robust Experimental Design: Under-Promise, Over-Deliver Assessment

This script demonstrates that maximizing the Robust Expected Information Gain (Sibson MI)
provides a rigorous lower bound on the True Expected Information Gain (Shannon MI),
provided the true process is "nicer" (less misspecified) than the worst-case assumption.
"""
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt
import csv

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian, LinearRegression

jax.config.update("jax_enable_x64", True)


def generate_optimal_design(num_designs: int) -> Array:
    """Generate optimal design: alternating -1 and 1."""
    return jnp.array([(-1.0)**(i+1) for i in range(num_designs)])


# ============================================================================
# Configuration
# ============================================================================

rng_key = jr.PRNGKey(42)

alpha_nominal = 1.0
alpha_true = 5e-1
alpha_robust = 1e-1

num_designs = 2
num_trials = 10_000

# Noise scale
dim_theta = 2
base_stddev = jnp.sqrt(1e-2)

# True process
proc_prior_mean = jnp.array([1.5, -1.5])
proc_prior_cov = jnp.diag(jnp.array([1e-2, 1e-2]))
proc_prior = Gaussian(mean=proc_prior_mean, cov=proc_prior_cov)

proc_stddev = base_stddev / jnp.sqrt(alpha_true)
proc_lklhd = LinearGaussian(noise_stddev=proc_stddev)
process = LinearRegression(prior=proc_prior, likelihood=proc_lklhd)

# Model
mdl_prior_mean = jnp.zeros(2)
mdl_prior_cov = jnp.diag(jnp.array([2.0, 2.0]))
mdl_prior = Gaussian(mean=mdl_prior_mean, cov=mdl_prior_cov)
mdl_lklhd = LinearGaussian(noise_stddev=base_stddev)
model = LinearRegression(prior=mdl_prior, likelihood=mdl_lklhd)

# Design
xi = generate_optimal_design(num_designs)

# ============================================================================
# Calculate the Promise (Robust EIG under nominal model)
# ============================================================================

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

    # Sample theta from prior and generate data from true process
    theta = process.prior.sample(prior_key, 1)[0]
    y = process.likelihood.sample(process_key, xi, theta)

    # Oracle: KL divergence using tilted posterior with α_true
    oracle_posterior = model.get_tilted_posterior(y, xi, alpha_true)
    ig_oracle = oracle_posterior.kl_to(model.prior)

    # Bayes: KL divergence using bayes posterior
    bayes_posterior = model.get_bayes_posterior(y, xi)
    ig_nominal = bayes_posterior.kl_to(model.prior)

    # Subjective: Rényi IG using nominal model with α_robust
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
print("RESULTS")
print("=" * 80)
print(f"Promise  (guarantee):           {promise_eig:.4f} nats")
print(f"Oracle   (true utility):        {avg_oracle:.4f} nats")
print(f"Nominal  (perceived utility): {avg_nominal:.4f} nats")
print(f"Robust   (perceived utility): {avg_robust:.4f} nats")
print("-" * 80)
print(f"Objective gap  (Oracle - Promise):    {avg_oracle - promise_eig:+.4f} nats")
print(f"Nominal gap   (Oracle - Perceived):  {avg_oracle - avg_nominal:+.4f} nats")
print(f"Robust gap    (Oracle - Perceived):  {avg_oracle - avg_robust:+.4f} nats")
print("=" * 80)

# ============================================================================
# Save Results
# ============================================================================

# csv_path = "examples/linear_regression/data/linreg_gaussian_infogain.csv"
# with open(csv_path, "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["ig_oracle", "ig_nominal", "ig_robust"])
#     for row in zip(dist_oracle.tolist(), dist_nominal.tolist(), dist_robust.tolist()):
#         writer.writerow(row)
#
# print(f"\nData saved to {csv_path}")

# ============================================================================
# Visualization
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Histogram bins
all_data = jnp.concatenate([dist_oracle, dist_nominal, dist_robust])
bins = jnp.linspace(all_data.min(), all_data.max(), 30)

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
csv_path = "examples/linear_regression/data/linreg_gaussian_infogain_hist.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["bin_center", "nominal_count", "robust_count"])
    for bc, nom, rob in zip(bin_centers, hist_nominal, hist_robust):
        writer.writerow([bc, nom, rob])

print(f"\nHistogram data saved to {csv_path} ({len(bin_centers)} bins)")

# Vertical lines
ax.axvline(promise_eig, color='0.1', linestyle='--', lw=2.5,
           label=f'Sibson MI: {promise_eig:.2f}')
# ax.axvline(avg_oracle, color='darkgreen', linestyle='-', lw=2.5,
#            label=f'Avg Oracle: {avg_oracle:.2f}')
# ax.axvline(avg_nominal, color='steelblue', linestyle=':', lw=2.5,
#            label=f'Avg Nominal: {avg_nominal:.2f}')
# ax.axvline(avg_robust, color='darkred', linestyle=':', lw=2.5,
#            label=f'Sibson agg Robust: {avg_robust:.2f}')

ax.set_xlabel('Information Gain (nats)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_xlim(left=None, right=8)
ax.set_title(f'Robust Information Gain: Under-Promise, Over-Deliver\n'
             f'(True α={alpha_true}, Robust α={alpha_robust})',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
