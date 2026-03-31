"""
Posterior Calibration Assessment for Misspecified Linear Regression

This variant matches linreg_coverage_studentt.py except that the optimized
two-point design is labeled as power-GEIG optimal, which coincides with the
Sibson boundary optimum in this linear-Gaussian model.
"""
from typing import NamedTuple

import csv

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from scipy.stats import chi2   # chi2.ppf is not supported in JAX

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian, LinearRegression

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

        rng_key, key1, key2 = jr.split(rng_key, 3)
        chi2_samples = jr.gamma(key1, self.df / 2, shape=(size,)) * 2

        L = jnp.linalg.cholesky(scale_matrix)
        z = jr.normal(key2, (size,))
        normal_samples = loc + L @ z

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
    thresholds = chi2.ppf(coverage_levels, df=dim)
    return jax.vmap(lambda t: jnp.mean(mahalanobis_distances <= t))(thresholds)


def generate_uniform_design(rng_key: Array, num_designs: int) -> Array:
    """Generate uniform design: uniform in [-1, 1]."""
    return jr.uniform(rng_key, shape=(num_designs,), minval=-1.0, maxval=1.0)


def generate_sibson_optimal_design(num_designs: int) -> Array:
    """Known Sibson-optimal two-point boundary design."""
    return jnp.array([(-1.0) ** (i + 1) for i in range(num_designs)])


def genereate_geig_optimal_design(num_designs: int) -> Array:
    """Power-GEIG-optimal design, equal to the Sibson optimum here."""
    return generate_sibson_optimal_design(num_designs)


# Configuration
rng_key = jr.PRNGKey(42)

# Models: Gaussian
alpha_robust = 1e-2
alpha_nominal = 1.0

# Experiment parameters
num_designs = 2
num_trials = 10_000

# Prior
dim_theta = 2
base_stddev = jnp.sqrt(1e-2)

# Create true process (Student-t)
df_true = 5.0

proc_prior_mean = jnp.array([1.5, -1.5])
proc_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
proc_prior = Gaussian(mean=proc_prior_mean, cov=proc_prior_cov)

proc_stddev = base_stddev * 2 / 5e-1
proc_lkld = LinearStudentT(df=df_true, noise_stddev=proc_stddev)

# Create model (Gaussian)
mdl_prior_mean = jnp.array([1.5, -1.5])
mdl_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
mdl_prior = Gaussian(mean=mdl_prior_mean, cov=mdl_prior_cov)

mdl_lklhd = LinearGaussian(noise_stddev=base_stddev)
model = LinearRegression(prior=mdl_prior, likelihood=mdl_lklhd)

print("=" * 80)
print("COVERAGE COMPARISON: STUDENT-T TRUE PROCESS")
print("=" * 80)
print(f"TRUE PROCESS: Student-t with df={df_true}")
print(f"  Noise: σ² = {proc_stddev**2}")
print("\nMODELS: LinearGaussian")
print(f"  Nominal α={alpha_nominal:.1f}:    Standard Bayesian Update")
print(f"  Robust  α={alpha_robust:.2f}:    Tilted Posterior")
print(f"\nDesign points:     {num_designs}")
print(f"Number of trials:    {num_trials}")
print("=" * 80)

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
print(f"\nRunning {num_trials} trials...")

xi_optimal = genereate_geig_optimal_design(num_designs)
print(f"Power-GEIG optimized design: {xi_optimal}")


def run_single_trial(rng_key):
    key_theta, key_design_rand, key_obs_opt, key_obs_rand = jr.split(rng_key, 4)

    theta_true = proc_prior.sample(key_theta, 1)[0]

    y_optimal = proc_lkld.sample(key_obs_opt, xi_optimal, theta_true)
    posterior_opt_nominal = model.get_tilted_posterior(y_optimal, xi_optimal, alpha_nominal)
    posterior_opt_robust = model.get_tilted_posterior(y_optimal, xi_optimal, alpha_robust)

    xi_uniform = generate_uniform_design(key_design_rand, num_designs)
    y_uniform = proc_lkld.sample(key_obs_rand, xi_uniform, theta_true)
    posterior_rand_nominal = model.get_tilted_posterior(y_uniform, xi_uniform, alpha_nominal)
    posterior_rand_robust = model.get_tilted_posterior(y_uniform, xi_uniform, alpha_robust)

    return jnp.array([
        compute_mahalanobis_distance(theta_true, posterior_opt_nominal.mean, posterior_opt_nominal.cov),
        compute_mahalanobis_distance(theta_true, posterior_opt_robust.mean, posterior_opt_robust.cov),
        compute_mahalanobis_distance(theta_true, posterior_rand_nominal.mean, posterior_rand_nominal.cov),
        compute_mahalanobis_distance(theta_true, posterior_rand_robust.mean, posterior_rand_robust.cov)
    ])


trial_keys = jr.split(rng_key, num_trials)
results = jax.jit(jax.vmap(run_single_trial))(trial_keys)

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

coverage_levels = jnp.linspace(0.05, 0.99, 25)

cal_curve_opt_nominal = compute_calibration_curve(mahal_opt_nominal, coverage_levels, dim_theta)
cal_curve_opt_robust = compute_calibration_curve(mahal_opt_robust, coverage_levels, dim_theta)
cal_curve_rand_nominal = compute_calibration_curve(mahal_rand_nominal, coverage_levels, dim_theta)
cal_curve_rand_robust = compute_calibration_curve(mahal_rand_robust, coverage_levels, dim_theta)

csv_path = "examples/linear_regression/data/linreg_coverage_studentt_geig.csv"
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
print(f"  Nominal α={alpha_nominal:.1f}:     {100 * coverage_95_opt_nominal:.1f}%")
print(f"  Robust  α={alpha_robust:.2f}:     {100 * coverage_95_opt_robust:.1f}%")

print("\nRANDOM DESIGN")
print("-" * 30)
print(f"  Nominal α={alpha_nominal:.1f}:     {100 * coverage_95_rand_nominal:.1f}%")
print(f"  Robust  α={alpha_robust:.2f}:     {100 * coverage_95_rand_robust:.1f}%")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
table_levels = jnp.array([0.25, 0.50, 0.75, 0.95])
print("\nCOVERAGE COMPARISON TABLE")
print("-" * 77)
print(f"{'Design':<18} | " + " | ".join([f"{100 * l:7.0f}%" for l in table_levels]))
print("-" * 77)

rob_opt_covs = [compute_credible_coverage(mahal_opt_robust, l, dim_theta) for l in table_levels]
rob_rand_covs = [compute_credible_coverage(mahal_rand_robust, l, dim_theta) for l in table_levels]
nom_opt_covs = [compute_credible_coverage(mahal_opt_nominal, l, dim_theta) for l in table_levels]
nom_rand_covs = [compute_credible_coverage(mahal_rand_nominal, l, dim_theta) for l in table_levels]

print(f"{'Robust + Optimal':<18} | " + " | ".join([f"{100 * c:7.1f}%" for c in rob_opt_covs]))
print(f"{'Robust + Uniform':<18} | " + " | ".join([f"{100 * c:7.1f}%" for c in rob_rand_covs]))
print("-" * 77)
print(f"{'Nominal + Optimal':<18} | " + " | ".join([f"{100 * c:7.1f}%" for c in nom_opt_covs]))
print(f"{'Nominal + Uniform':<18} | " + " | ".join([f"{100 * c:7.1f}%" for c in nom_rand_covs]))
print("-" * 77)

print("\n" + "=" * 80)

# ============================================================================
# VISUALIZATION: CALIBRATION CURVES
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION")
print("=" * 80)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot([0, 1], [0, 1], ':', color='0.2', lw=2, label='Perfect')
ax.plot(coverage_levels, cal_curve_opt_nominal, '--', color='0.1', lw=2.5,
        label=f'Nominal w/ Optimal (α={alpha_nominal:.1f})')
ax.plot(coverage_levels, cal_curve_opt_robust, '-', color='0.1', lw=2.5,
        label=f'Robust w/ Power-GEIG Optimal (α={alpha_robust:.2f})')
ax.plot(coverage_levels, cal_curve_rand_nominal, '--', color='0.5', lw=2.5,
        label=f'Nominal w/ Uniform (α={alpha_nominal:.1f})')
ax.plot(coverage_levels, cal_curve_rand_robust, '-', color='0.5', lw=2.5,
        label=f'Robust w/ Uniform (α={alpha_robust:.2f})')

ax.set_xlabel('Expected Coverage', fontsize=12)
ax.set_ylabel('Actual Coverage', fontsize=12)
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()
