"""
Posterior Calibration Assessment for Misspecified Linear Regression

TRUE PROCESS: Student-t 
MODELS: LinearGaussian

This script evaluates posterior calibration when the true data-generating process
has heavy-tailed Student-t noise but we fit a Gaussian model.
"""
from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import chi2   # chi2.ppf is not supported in JAX
import csv

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian, LinearRegression

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


def compute_mahalanobis_distance(
    theta: Array,
    mu_post: Array,
    cov_post: Array,
) -> float:
    """Compute Mahalanobis distance between true parameter and posterior mean."""
    diff = theta - mu_post
    prec_post = jnp.linalg.inv(cov_post)
    return diff @ prec_post @ diff


def compute_credible_coverage(
    mahalanobis_distances: Array,
    coverage_level: float,
    dim: int,
) -> Array:
    """Compute fraction of samples within credible region."""
    threshold = chi2.ppf(coverage_level, df=dim)
    return jnp.mean(mahalanobis_distances <= threshold)


def compute_calibration_curve(
    mahalanobis_distances: Array,
    coverage_levels: Array,
    dim: int,
) -> Array:
    """Compute actual coverage at different credible levels."""    
    # Pre-compute thresholds using scipy outside of JAX transformations
    thresholds = chi2.ppf(coverage_levels, df=dim)
    return jax.vmap(lambda t: jnp.mean(mahalanobis_distances <= t))(thresholds)


def generate_uniform_design(rng_key: Array, num_designs: int) -> Array:
    """Generate uniform design: uniform in [-1, 1]."""
    return jr.uniform(rng_key, shape=(num_designs,), minval=-1.0, maxval=1.0)


def generate_optimal_design(num_designs: int) -> Array:
    """Generate optimal design: alternating -1 and 1."""
    return jnp.array([(-1.0)**(i+1) for i in range(num_designs)])


# Configuration
rng_key = jr.PRNGKey(42)

# Models: Gaussian
# jnp.logspace(-2, 0, 25)  # from 0.01 to 1.0
alpha_nominal = 1.0
alpha_true = 5e-1
alpha_robust = 1e-1

# Experiment parameters
num_designs = 2
num_trials = 10_000

# Noise scale
dim_theta = 2
base_stddev = jnp.sqrt(1e-2)

# Create true process
proc_prior_mean = jnp.array([1.5, -1.5])
proc_prior_cov = jnp.diag(jnp.array([1e-2, 1e-2]))
proc_prior = Gaussian(mean=proc_prior_mean, cov=proc_prior_cov)

proc_stddev = base_stddev / jnp.sqrt(alpha_true)
proc_lklhd = LinearGaussian(noise_stddev=proc_stddev)
process = LinearRegression(prior=proc_prior, likelihood=proc_lklhd)

# Create model (Gaussian)
mdl_prior_mean = jnp.zeros(2)
mdl_prior_cov = jnp.diag(jnp.array([2.0, 2.0]))
mdl_prior = Gaussian(mean=mdl_prior_mean, cov=mdl_prior_cov)

mdl_lklhd = LinearGaussian(noise_stddev=base_stddev)
model = LinearRegression(prior=mdl_prior, likelihood=mdl_lklhd)

print("=" * 80)
print("COVERAGE COMPARISON: GAUSSIAN TRUE PROCESS")
print("=" * 80)
print(f"TRUE PROCESS: Gaussian with σ² = {proc_stddev**2}")
print(f"\nMODELS: LinearGaussian")
print(f"  Nominal α={alpha_nominal:.1f}:    Standard Bayesian Update")
print(f"  Robust  α={alpha_robust:.2f}:    Tilted Posterior")
print(f"\nDesign points:     {num_designs}")
print(f"Number of trials:    {num_trials}")
print("=" * 80)

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
print(f"\nRunning {num_trials} trials...")

# Generate optimized design once
xi_optimal = generate_optimal_design(num_designs)

def run_single_trial(rng_key):
    key_theta, key_design_rand, key_obs_opt, key_obs_rand = jr.split(rng_key, 4)

    # Sample true parameter from prior
    theta_true = process.prior.sample(key_theta, 1)[0]

    # ========== OPTIMIZED DESIGN ==========
    y_optimal = process.likelihood.sample(key_obs_opt, xi_optimal, theta_true)
    posterior_opt_nominal = model.get_tilted_posterior(y_optimal, xi_optimal, alpha_nominal)
    posterior_opt_robust = model.get_tilted_posterior(y_optimal, xi_optimal, alpha_robust)

    # ========== Uniform DESIGN ==========
    xi_uniform = generate_uniform_design(key_design_rand, num_designs)
    y_uniform = process.likelihood.sample(key_obs_rand, xi_uniform, theta_true)
    posterior_rand_nominal = model.get_tilted_posterior(y_uniform, xi_uniform, alpha_nominal)
    posterior_rand_robust = model.get_tilted_posterior(y_uniform, xi_uniform, alpha_robust)

    # Compute Mahalanobis distances
    return jnp.array([
        compute_mahalanobis_distance(theta_true, posterior_opt_nominal.mean, posterior_opt_nominal.cov),
        compute_mahalanobis_distance(theta_true, posterior_opt_robust.mean, posterior_opt_robust.cov),
        compute_mahalanobis_distance(theta_true, posterior_rand_nominal.mean, posterior_rand_nominal.cov),
        compute_mahalanobis_distance(theta_true, posterior_rand_robust.mean, posterior_rand_robust.cov)
    ])

# Batch run all trials
trial_keys = jr.split(rng_key, num_trials)
results = jax.jit(jax.vmap(run_single_trial))(trial_keys)

# Extract results
mahal_opt_nominal = results[:, 0]
mahal_opt_robust = results[:, 1]
mahal_rand_nominal = results[:, 2]
mahal_rand_robust = results[:, 3]

print(f"Completed {num_trials} trials")

# ============================================================================
# COMPUTE CALIBRATION METRICS
# ============================================================================
print("\n" + "=" * 80)
print("COMPUTING CALIBRATION METRICS")
print("=" * 80)

# Coverage levels to evaluate
coverage_levels = jnp.linspace(0.05, 0.99, 25)

# Compute calibration curves for optimized design
cal_curve_opt_nominal = compute_calibration_curve(mahal_opt_nominal, coverage_levels, dim_theta)
cal_curve_opt_robust = compute_calibration_curve(mahal_opt_robust, coverage_levels, dim_theta)

# Compute calibration curves for random design
cal_curve_rand_nominal = compute_calibration_curve(mahal_rand_nominal, coverage_levels, dim_theta)
cal_curve_rand_robust = compute_calibration_curve(mahal_rand_robust, coverage_levels, dim_theta)

# Write plotted data to CSV
csv_path = "examples/linear_regression/data/linreg_gaussian_coverage.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "coverage_levels",
        "cal_curve_opt_nominal",
        "cal_curve_opt_robust",
        "cal_curve_rand_nominal",
        "cal_curve_rand_robust",
    ])
    for row in zip(
        coverage_levels.tolist(),
        cal_curve_opt_nominal.tolist(),
        cal_curve_opt_robust.tolist(),
        cal_curve_rand_nominal.tolist(),
        cal_curve_rand_robust.tolist(),
    ):
        writer.writerow(row)

# 95% coverage for reporting
coverage_95_opt_nominal = compute_credible_coverage(mahal_opt_nominal, 0.95, dim_theta)
coverage_95_opt_robust = compute_credible_coverage(mahal_opt_robust, 0.95, dim_theta)
coverage_95_rand_nominal = compute_credible_coverage(mahal_rand_nominal, 0.95, dim_theta)
coverage_95_rand_robust = compute_credible_coverage(mahal_rand_robust, 0.95, dim_theta)

print("Computed calibration curves")

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS: 95% Credible Region Coverage")
print("=" * 80)

print("\nOPTIMIZED DESIGN")
print("-" * 30)
print(f"  Nominal α={alpha_nominal:.1f}:     {100*coverage_95_opt_nominal:.1f}%")
print(f"  Robust  α={alpha_robust:.2f}:     {100*coverage_95_opt_robust:.1f}%")

print("\nRANDOM DESIGN")
print("-" * 30)
print(f"  Nominal α={alpha_nominal:.1f}:     {100*coverage_95_rand_nominal:.1f}%")
print(f"  Robust  α={alpha_robust:.2f}:     {100*coverage_95_rand_robust:.1f}%")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
table_levels = jnp.linspace(0.1, 1.0, 10)
print("\nCOVERAGE COMPARISON TABLE")
print("-" * 115)
print(f"{'Design':<18} | " + " | ".join([f"{l:8.1f}" for l in table_levels]))
print("-" * 115)

# Robust evaluations
rob_opt_covs = [compute_credible_coverage(mahal_opt_robust, l, dim_theta) for l in table_levels]
rob_rand_covs = [compute_credible_coverage(mahal_rand_robust, l, dim_theta) for l in table_levels]

# Nominal evaluations
nom_opt_covs = [compute_credible_coverage(mahal_opt_nominal, l, dim_theta) for l in table_levels]
nom_rand_covs = [compute_credible_coverage(mahal_rand_nominal, l, dim_theta) for l in table_levels]

print(f"{'Robust + Optimal':<18} | " + " | ".join([f"{100*c:8.1f}%" for c in rob_opt_covs]))
print(f"{'Robust + Uniform':<18} | " + " | ".join([f"{100*c:8.1f}%" for c in rob_rand_covs]))
print("-" * 115)
print(f"{'Nominal + Optimal':<18} | " + " | ".join([f"{100*c:8.1f}%" for c in nom_opt_covs]))
print(f"{'Nominal + Uniform':<18} | " + " | ".join([f"{100*c:8.1f}%" for c in nom_rand_covs]))
print("-" * 115)

print("\n" + "=" * 80)

# ============================================================================
# VISUALIZATION: CALIBRATION CURVES
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION")
print("=" * 80)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Perfect calibration reference
ax.plot([0, 1], [0, 1], ':', color='0.2', lw=2, label='Perfect')

# Optimal design - Nominal (dark grey, dashed)
ax.plot(coverage_levels, cal_curve_opt_nominal, '--', color='0.1', lw=2.5, 
        label=f'Nominal w/ Optimal (α={alpha_nominal:.1f})')

# Optimal design - Robust (dark grey, solid)
ax.plot(coverage_levels, cal_curve_opt_robust, '-', color='0.1', lw=2.5, 
        label=f'Robust w/ Optimal (α={alpha_robust:.2f})')

# Uniform design - Nominal (medium grey, dashed)
ax.plot(coverage_levels, cal_curve_rand_nominal, '--', color='0.5', lw=2.5, 
        label=f'Nominal w/ Uniform (α={alpha_nominal:.1f})')

# Uniform design - Robust (medium grey, solid)
ax.plot(coverage_levels, cal_curve_rand_robust, '-', color='0.5', lw=2.5, 
        label=f'Robust w/ Uniform (α={alpha_robust:.2f})')

ax.set_xlabel('Expected Coverage', fontsize=12)
ax.set_ylabel('Actual Coverage', fontsize=12)
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()
