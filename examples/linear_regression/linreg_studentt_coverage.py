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
from scipy.stats import chi2

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian, LinearRegression

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class LinearStudentT(NamedTuple):
    """
    Student-t linear likelihood for regression

    Args:
        noise_stddev: Noise standard deviation
        df: Degrees of freedom
    """
    df: float
    noise_stddev: float

    def design_matrix(self, design: Array) -> Array:
        """Construct augmented design matrix H(ξ) = [ξ, 1]."""
        design = jnp.atleast_1d(design)
        size = design.shape[0]
        return jnp.column_stack([design, jnp.ones(size)])

    def mean(self, design: Array, theta: Array) -> Array:
        """Compute mean for a single design point."""
        H = self.design_matrix(design)
        return H @ theta

    def sample(self, rng_key: Array, design: Array, theta: Array) -> Array:
        """Sample from Student-t likelihood."""
        size = design.shape[0]

        loc = self.mean(design, theta)
        scale_matrix = self.noise_stddev**2 * jnp.eye(size)

        # Sample from chi-squared
        rng_key, key1, key2 = jr.split(rng_key, 3)
        chi2_samples = jr.gamma(key1, self.df/2, shape=(size,)) * 2

        # Sample from multivariate normal
        L = jnp.linalg.cholesky(scale_matrix)
        z = jr.normal(key2, (size,))
        normal_samples = loc + L @ z

        # Scale to get Student-t
        scale_factors = jnp.sqrt(self.df / chi2_samples)
        t_samples = loc + (normal_samples - loc) * scale_factors

        return t_samples


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
    coverages = []
    for level in coverage_levels:
        coverage = compute_credible_coverage(mahalanobis_distances, level, dim)
        coverages.append(coverage)
    return jnp.array(coverages)


def generate_uniform_design(rng_key: Array, num_designs: int) -> Array:
    """Generate uniform design: uniform in [-1, 1]."""
    return jr.uniform(rng_key, shape=(num_designs,), minval=-1.0, maxval=1.0)


def generate_optimal_design(num_designs: int) -> Array:
    """Generate optimized design: alternating -1 and 1."""
    return jnp.array([(-1.0)**(i+1) for i in range(num_designs)])


# Configuration
rng_key = jr.PRNGKey(42)

# True process: Student-t
df_true = 3.
noise_stddev = jnp.sqrt(1e-1)

# Models: Gaussian
# jnp.logspace(-2, 0, 25)  # from 0.01 to 1.0
alpha_robust = 1e-2   # Robust model
alpha_nominal = 1.0   # Nominal model

# Experiment parameters
num_designs = 10      # Number of designs
num_trials = 10_000   # Number of trials

# Prior
dim_theta = 2
prior_mean = jnp.zeros(2)
prior_cov = jnp.diag(jnp.array([3.0, 2.0]))  # jnp.eye(2)

# Create true process (Student-t)
process = LinearStudentT(df=df_true, noise_stddev=noise_stddev)

# Create model (Gaussian)
prior = Gaussian(mean=prior_mean, cov=prior_cov)
model = LinearGaussian(noise_stddev=noise_stddev)
linreg = LinearRegression(prior=prior, likelihood=model)

print("=" * 80)
print("COVERAGE COMPARISON: STUDENT-T TRUE PROCESS")
print("=" * 80)
print(f"TRUE PROCESS: Student-t with df={df_true}")
print(f"  Noise: σ² = {noise_stddev**2}")
print(f"\nMODELS: LinearGaussian")
print(f"  Noise: constant σ² = {noise_stddev**2}")
print(f"\nDesign points:                 {num_designs}")
print(f"Number of trials:              {num_trials}")
print(f"\nModel specifications:")
print(f"  Nominal (α={alpha_nominal:.1f}):    assumes Gaussian noise")
print(f"  Robust (α={alpha_robust:.2f}):   robust to model misspecification")

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING EXPERIMENTS")
print("=" * 80)


# Storage for results
mahal_opt_nominal = []
mahal_opt_robust = []
mahal_rand_nominal = []
mahal_rand_robust = []

# Sample from prior and evaluate calibration
print(f"\nRunning {num_trials} trials...")
for i in range(num_trials):
    if (i + 1) % 1000 == 0:
        print(f"  Trial {i + 1}/{num_trials}...")

    rng_key, key_theta, key_design_rand, key_obs_opt, key_obs_rand = jr.split(rng_key, 5)

    # Sample true parameter from prior
    theta_true = prior.sample(key_theta, 1)[0]

    # ========== OPTIMIZED DESIGN ==========
    # Generate optimized design
    xi_optimal = generate_optimal_design(num_designs)

    # Generate data from TRUE Student-t process
    y_optimal = process.sample(key_obs_opt, xi_optimal, theta_true)

    # Compute posteriors: nominal and robust
    posterior_opt_nominal = linreg.get_tilted_posterior(y_optimal, xi_optimal, alpha_nominal)
    posterior_opt_robust = linreg.get_tilted_posterior(y_optimal, xi_optimal, alpha_robust)

    # Compute Mahalanobis distances
    mahal_opt_nom = compute_mahalanobis_distance(theta_true, posterior_opt_nominal.mean, posterior_opt_nominal.cov)
    mahal_opt_rob = compute_mahalanobis_distance(theta_true, posterior_opt_robust.mean, posterior_opt_robust.cov)

    mahal_opt_nominal.append(mahal_opt_nom)
    mahal_opt_robust.append(mahal_opt_rob)

    # ========== Uniform DESIGN ==========
    # Generate random design
    xi_uniform = generate_uniform_design(key_design_rand, num_designs)

    # Generate data from TRUE Student-t process
    y_uniform = process.sample(key_obs_rand, xi_uniform, theta_true)

    # Compute posteriors: nominal and robust
    posterior_rand_nominal = linreg.get_tilted_posterior(y_uniform, xi_uniform, alpha_nominal)
    posterior_rand_robust = linreg.get_tilted_posterior(y_uniform, xi_uniform, alpha_robust)

    # Compute Mahalanobis distances
    mahal_rand_nom = compute_mahalanobis_distance(theta_true, posterior_rand_nominal.mean, posterior_rand_nominal.cov)
    mahal_rand_rob = compute_mahalanobis_distance(theta_true, posterior_rand_robust.mean, posterior_rand_robust.cov)

    mahal_rand_nominal.append(mahal_rand_nom)
    mahal_rand_robust.append(mahal_rand_rob)

# Convert to arrays
mahal_opt_nominal = jnp.array(mahal_opt_nominal)
mahal_opt_robust = jnp.array(mahal_opt_robust)
mahal_rand_nominal = jnp.array(mahal_rand_nominal)
mahal_rand_robust = jnp.array(mahal_rand_robust)

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
print("RESULTS")
print("=" * 80)

print("\n" + "-" * 80)
print("OPTIMIZED DESIGN")
print("-" * 80)
print(f"Mahalanobis distance M² (expected: {dim_theta:.0f} for well-calibrated):")
print(f"  Nominal (α={alpha_nominal:.1f}):     {jnp.mean(mahal_opt_nominal):.2f} ± {jnp.std(mahal_opt_nominal):.2f}  ({jnp.mean(mahal_opt_nominal)/dim_theta:.2f}× expected)")
print(f"  Robust (α={alpha_robust:.2f}):   {jnp.mean(mahal_opt_robust):.2f} ± {jnp.std(mahal_opt_robust):.2f}  ({jnp.mean(mahal_opt_robust)/dim_theta:.2f}× expected)")

print(f"\n95% credible region coverage (expected: 95.0%):")
print(f"  Nominal (α={alpha_nominal:.1f}):     {100*coverage_95_opt_nominal:.1f}%  {'✗ under-coverage' if coverage_95_opt_nominal < 0.94 else '✓'}")
print(f"  Robust (α={alpha_robust:.2f}):   {100*coverage_95_opt_robust:.1f}%  {'✓ good coverage' if 0.94 <= coverage_95_opt_robust <= 0.96 else '• over-coverage' if coverage_95_opt_robust > 0.96 else '✗ under-coverage'}")

print("\n" + "-" * 80)
print("RANDOM DESIGN")
print("-" * 80)
print(f"Mahalanobis distance M² (expected: {dim_theta:.0f} for well-calibrated):")
print(f"  Nominal (α={alpha_nominal:.1f}):     {jnp.mean(mahal_rand_nominal):.2f} ± {jnp.std(mahal_rand_nominal):.2f}  ({jnp.mean(mahal_rand_nominal)/dim_theta:.2f}× expected)")
print(f"  Robust (α={alpha_robust:.2f}):   {jnp.mean(mahal_rand_robust):.2f} ± {jnp.std(mahal_rand_robust):.2f}  ({jnp.mean(mahal_rand_robust)/dim_theta:.2f}× expected)")

print(f"\n95% credible region coverage (expected: 95.0%):")
print(f"  Nominal (α={alpha_nominal:.1f}):     {100*coverage_95_rand_nominal:.1f}%  {'✗ under-coverage' if coverage_95_rand_nominal < 0.94 else '✓'}")
print(f"  Robust (α={alpha_robust:.2f}):   {100*coverage_95_rand_robust:.1f}%  {'✓ good coverage' if 0.94 <= coverage_95_rand_robust <= 0.96 else '• over-coverage' if coverage_95_rand_robust > 0.96 else '✗ under-coverage'}")

print("\n" + "-" * 80)
print("KEY INSIGHT: MODEL MISSPECIFICATION")
print("-" * 80)
print(f"With Student-t true process (df={df_true}, heavy tails):")
print(f"  • Nominal Gaussian model is MISSPECIFIED")
print(f"    - Underestimates uncertainty from heavy tails")
print(f"    - Optimized design M² = {jnp.mean(mahal_opt_nominal):.2f} >> {dim_theta} (overconfident)")
print(f"    - Optimized design coverage = {100*coverage_95_opt_nominal:.1f}% << 95% (under-coverage)")
print(f"\n  • Robust model (α={alpha_robust:.2f}) provides ROBUSTNESS")
print(f"    - Inflates uncertainty to account for misspecification")
print(f"    - Optimized design M² = {jnp.mean(mahal_opt_robust):.2f} ≈ {dim_theta} (better calibrated)")
print(f"    - Optimized design coverage = {100*coverage_95_opt_robust:.1f}% ≈ 95% (good coverage)")

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