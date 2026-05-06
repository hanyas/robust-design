"""
Robust Experimental Design: Under-Promise, Over-Deliver Assessment
"""

from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt
import csv

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian, LinearRegression

jax.config.update("jax_enable_x64", True)


class LinearStudentT(NamedTuple):
    """Student-t linear likelihood (matches ``linreg_coverage_studentt.py``)."""

    df: float
    noise_stddev: float

    def design_matrix(self, design: Array) -> Array:
        design = jnp.atleast_1d(design)
        size = design.shape[0]
        return jnp.column_stack([design, jnp.ones(size)])

    def mean(self, design: Array, theta: Array) -> Array:
        H = self.design_matrix(design)
        return H @ theta

    def sample(self, rng_key: Array, design: Array, theta: Array) -> Array:
        size = design.shape[0]
        loc = self.mean(design, theta)
        scale_matrix = self.noise_stddev**2 * jnp.eye(size)

        rng_key, key1, key2 = jr.split(rng_key, 3)
        chi2_samples = jr.gamma(key1, self.df / 2, shape=(size,)) * 2

        L = jnp.linalg.cholesky(scale_matrix)
        z = jr.normal(key2, (size,))
        normal_samples = loc + L @ z

        scale_factors = jnp.sqrt(self.df / chi2_samples)
        return loc + (normal_samples - loc) * scale_factors


def generate_random_design(rng_key: Array, num_designs: int) -> Array:
    """Uniform random design in [-1, 1]."""
    return jr.uniform(rng_key, shape=(num_designs,), minval=-1.0, maxval=1.0)


# ============================================================================
# Configuration
# ============================================================================

rng_key = jr.PRNGKey(42)

alpha_nominal = 1.0
alpha_robust = 0.056

num_designs = 2
num_trials = 10_000

# Noise scale
dim_theta = 2
base_stddev = jnp.sqrt(1e-2)

# True process — Student-t, matches linreg_coverage_studentt.py
df_true = 25.0
proc_prior_mean = jnp.array([1.5, -1.5])
proc_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
proc_prior = Gaussian(mean=proc_prior_mean, cov=proc_prior_cov)

proc_stddev = base_stddev * 2 / 5e-1
proc_lklhd = LinearStudentT(df=df_true, noise_stddev=proc_stddev)
process = LinearRegression(prior=proc_prior, likelihood=proc_lklhd)

# Model
mdl_prior_mean = jnp.array([1.5, -1.5])
mdl_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
mdl_prior = Gaussian(mean=mdl_prior_mean, cov=mdl_prior_cov)
mdl_lklhd = LinearGaussian(noise_stddev=base_stddev)
model = LinearRegression(prior=mdl_prior, likelihood=mdl_lklhd)

# ============================================================================
# Run Simulation
# ============================================================================

print(f"\nRunning {num_trials} trials...")


def run_trial(key):
    """Single trial: random design, generate data, compute nominal and robust IG."""
    design_key, prior_key, process_key = jr.split(key, 3)

    # Random design (uniform in [-1, 1]) per trial
    xi = generate_random_design(design_key, num_designs)

    # Sample theta from prior and generate data from true process
    theta = process.prior.sample(prior_key, 1)[0]
    y = process.likelihood.sample(process_key, xi, theta)

    # Bayes: KL divergence using bayes posterior
    bayes_posterior = model.get_bayes_posterior(y, xi)
    ig_nominal = bayes_posterior.kl_to(model.prior)

    # Subjective: Rényi IG using nominal model with α_robust
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
print("RESULTS")
print("=" * 80)
print(f"Nominal  (perceived utility): {avg_nominal:.4f} nats")
print(f"Robust   (perceived utility): {avg_robust:.4f} nats")
print("-" * 80)

# ============================================================================
# Visualization
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Histogram bins — shared range across both distributions, clipped at the
# 99th percentile so a handful of Student-t tail outliers don't blow out
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
csv_path = "examples/linear_regression/data/linreg_gain_studentt.csv"
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
