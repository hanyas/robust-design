"""
Posterior Calibration under Random Designs Across an α Sweep.
"""

from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import chi2  # chi2.ppf is not supported in JAX
import csv

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

        rng_key, key1, key2 = jr.split(rng_key, 3)
        chi2_samples = jr.gamma(key1, self.df / 2, shape=(size,)) * 2

        L = jnp.linalg.cholesky(scale_matrix)
        z = jr.normal(key2, (size,))
        normal_samples = loc + L @ z

        scale_factors = jnp.sqrt(self.df / chi2_samples)
        t_samples = loc + (normal_samples - loc) * scale_factors

        return t_samples


def compute_mahalanobis_distance(theta: Array, mu_post: Array, cov_post: Array) -> float:
    """Mahalanobis distance between true parameter and posterior mean."""
    diff = theta - mu_post
    prec_post = jnp.linalg.inv(cov_post)
    return diff @ prec_post @ diff


def compute_credible_coverage(mahalanobis_distances: Array, coverage_level: float, dim: int) -> Array:
    """Fraction of samples within credible region at coverage_level."""
    threshold = chi2.ppf(coverage_level, df=dim)
    return jnp.mean(mahalanobis_distances <= threshold)


def compute_calibration_curve(mahalanobis_distances: Array, coverage_levels: Array, dim: int) -> Array:
    """Actual coverage at a list of expected coverage levels."""
    thresholds = chi2.ppf(coverage_levels, df=dim)
    return jax.vmap(lambda t: jnp.mean(mahalanobis_distances <= t))(thresholds)


def generate_uniform_design(rng_key: Array, num_designs: int) -> Array:
    """Uniform random design in [-1, 1]."""
    return jr.uniform(rng_key, shape=(num_designs,), minval=-1.0, maxval=1.0)


rng_key = jr.PRNGKey(42)

num_designs = 2
num_trials = 10_000

# α grid
alpha_values = 10.0 ** jnp.linspace(-1.25, 0.0, 5)
num_alphas = alpha_values.shape[0]

# Prior
dim_theta = 2
base_stddev = jnp.sqrt(1e-2)

# True process (Student-t)
df_true = 25.0

proc_prior_mean = jnp.array([1.5, -1.5])
proc_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
proc_prior = Gaussian(mean=proc_prior_mean, cov=proc_prior_cov)

proc_stddev = base_stddev * 2 / 5e-1
proc_lkld = LinearStudentT(df=df_true, noise_stddev=proc_stddev)

# Model (Gaussian)
mdl_prior_mean = jnp.array([1.5, -1.5])
mdl_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
mdl_prior = Gaussian(mean=mdl_prior_mean, cov=mdl_prior_cov)

mdl_lklhd = LinearGaussian(noise_stddev=base_stddev)
model = LinearRegression(prior=mdl_prior, likelihood=mdl_lklhd)

print("=" * 80)
print("COVERAGE SWEEP UNDER RANDOM DESIGNS: LINEAR REGRESSION")
print("=" * 80)
print(f"TRUE PROCESS: Student-t with df={df_true}")
print(f"  Noise: σ² = {proc_stddev**2}")
print("\nMODEL: LinearGaussian (uniform random designs only)")
print(f"  α values:        {[float(a) for a in alpha_values]}")
print(f"\nDesign points:    {num_designs}")
print(f"Number of trials: {num_trials}")
print("=" * 80)


print(f"\nRunning {num_trials} trials...")


def run_single_trial(rng_key):
    """For one trial, return Mahalanobis distance per α (shape (num_alphas,))."""
    key_theta, key_design, key_obs = jr.split(rng_key, 3)

    theta_true = proc_prior.sample(key_theta, 1)[0]

    xi_uniform = generate_uniform_design(key_design, num_designs)
    y_uniform = proc_lkld.sample(key_obs, xi_uniform, theta_true)

    def mahal_for_alpha(alpha):
        posterior = model.get_tilted_posterior(y_uniform, xi_uniform, alpha)
        return compute_mahalanobis_distance(theta_true, posterior.mean, posterior.cov)

    return jax.vmap(mahal_for_alpha)(alpha_values)


trial_keys = jr.split(rng_key, num_trials)
results = jax.jit(jax.vmap(run_single_trial))(trial_keys)
# results shape: (num_trials, num_alphas)

mahal_per_alpha = [results[:, i] for i in range(num_alphas)]

print(f"Completed {num_trials} trials")

print("\n" + "=" * 80)
print("COMPUTING CALIBRATION METRICS")
print("=" * 80)

coverage_levels = jnp.linspace(0.05, 0.99, 25)
curves = [compute_calibration_curve(m, coverage_levels, dim_theta) for m in mahal_per_alpha]

csv_path = "examples/linear_regression/data/linreg_coverage_studentt.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = ["coverage_levels"] + [f"alpha_{float(a):.4f}" for a in alpha_values]
    writer.writerow(header)
    for level_idx, level in enumerate(coverage_levels.tolist()):
        row = [level] + [curves[i][level_idx].item() for i in range(num_alphas)]
        writer.writerow(row)

print("Computed calibration curves")

print("\n" + "=" * 80)
print("COVERAGE COMPARISON TABLE")
print("=" * 80)
table_levels = jnp.array([0.25, 0.50, 0.75, 0.95])
print(f"{'α':>8} | " + " | ".join([f"{100 * l:7.0f}%" for l in table_levels]))
print("-" * 73)
for i, alpha in enumerate(alpha_values):
    covs = [compute_credible_coverage(mahal_per_alpha[i], l, dim_theta) for l in table_levels]
    print(f"{float(alpha):>8.4f} | " + " | ".join([f"{100 * c:7.1f}%" for c in covs]))

print("\n" + "=" * 80)

print("\nVISUALIZATION")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot([0, 1], [0, 1], ":", color="0.2", lw=2, label="Perfect")

# Grayscale: small α (dark) -> large α (light)
for i, alpha in enumerate(alpha_values):
    shade = 0.10 + 0.65 * (i / max(num_alphas - 1, 1))
    ax.plot(
        coverage_levels,
        curves[i],
        "-",
        color=str(shade),
        lw=2.5,
        label=f"α = {float(alpha):.3f}",
    )

ax.set_xlabel("Expected Coverage", fontsize=12)
ax.set_ylabel("Actual Coverage", fontsize=12)
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
