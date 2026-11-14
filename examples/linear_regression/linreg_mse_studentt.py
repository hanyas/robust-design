"""
Posterior Mean MSE for Misspecified Linear Regression
"""

from typing import NamedTuple

import csv

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian, LinearRegression

jax.config.update("jax_enable_x64", True)


class LinearStudentT(NamedTuple):
    """
    Student-t linear likelihood for regression.

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


def generate_sibson_optimal_design(num_designs: int) -> Array:
    """Known Sibson-optimal two-point boundary design."""
    return jnp.array([(-1.0) ** (i + 1) for i in range(num_designs)])


def generate_uniform_design(rng_key: Array, num_designs: int) -> Array:
    """Generate uniform random design in [-1, 1]."""
    return jr.uniform(rng_key, shape=(num_designs,), minval=-1.0, maxval=1.0)


def summarize(values: Array) -> tuple[Array, Array, Array]:
    """Mean and central 80% interval (q10, q90) across trials."""
    mean = jnp.mean(values, axis=1)
    q10 = jnp.percentile(values, 10, axis=1)
    q90 = jnp.percentile(values, 90, axis=1)
    return mean, q10, q90


rng_key = jr.PRNGKey(42)

alpha_values = 10.0 ** jnp.linspace(-1.25, 0.0, 5)
num_alphas = len(alpha_values)

num_design_points = 2
num_trials = 10_000

dim_theta = 2
base_stddev = jnp.sqrt(1e-2)

# True process (Student-t)
df_true = 25.0

proc_prior_mean = jnp.array([1.5, -1.5])
proc_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
proc_prior = Gaussian(mean=proc_prior_mean, cov=proc_prior_cov)

proc_stddev = base_stddev * 2 / 5e-1
proc_lklhd = LinearStudentT(df=df_true, noise_stddev=proc_stddev)

# Model (Gaussian)
mdl_prior_mean = jnp.array([1.5, -1.5])
mdl_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
mdl_prior = Gaussian(mean=mdl_prior_mean, cov=mdl_prior_cov)

mdl_lklhd = LinearGaussian(noise_stddev=base_stddev)
model = LinearRegression(prior=mdl_prior, likelihood=mdl_lklhd)

print("=" * 80)
print("POSTERIOR MEAN RMSE EXPERIMENT: STUDENT-T TRUE PROCESS")
print("=" * 80)
print(f"TRUE PROCESS: Student-t with df={df_true}")
print(f"  Noise: σ² = {proc_stddev**2:.4f}")
print("\nMODEL: LinearGaussian")
print("  Robust learning:   q(θ|x) ∝ p(θ) p(x|θ)^α")
print("\nMetric")
print("  sqrt( mean_i (E[θ|x,ξ]_i − θ_true_i)² )  per trial  (RMSE)")
print("\nComparison rule")
print("  Sibson-optimized two-point boundary design vs uniform random design")
print(f"\nDesign points:     {num_design_points}")
print(f"Number of trials:  {num_trials}")
print("=" * 80)

print("\n" + "=" * 80)
print("RUNNING EXPERIMENTS")
print("=" * 80)

rmse_optimal = jnp.zeros((num_alphas, num_trials))
rmse_uniform = jnp.zeros((num_alphas, num_trials))

xi_optimal = generate_sibson_optimal_design(num_design_points)
print(f"Sibson-optimized design: {xi_optimal}")
print(f"\nRunning {num_trials} trials...")


@jax.jit
def evaluate_all_alphas(
    alpha_vals: Array,
    y_optimal: Array,
    y_uniform: Array,
    xi_uniform: Array,
    theta_true: Array,
) -> tuple[Array, Array]:
    """Per-trial RMSE of the posterior mean for both designs across α."""

    def evaluate_alpha(alpha):
        post_opt = model.get_tilted_posterior(y_optimal, xi_optimal, alpha)
        post_unif = model.get_tilted_posterior(y_uniform, xi_uniform, alpha)
        rmse_opt = jnp.sqrt(jnp.mean((post_opt.mean - theta_true) ** 2))
        rmse_unif = jnp.sqrt(jnp.mean((post_unif.mean - theta_true) ** 2))
        return rmse_opt, rmse_unif

    return jax.vmap(evaluate_alpha)(alpha_vals)


for trial in range(num_trials):
    if (trial + 1) % 200 == 0 or trial == 0:
        print(f"  Trial {trial + 1}/{num_trials}...")

    rng_key, sub_key = jr.split(rng_key)
    theta_true = proc_prior.sample(sub_key, 1)[0]

    rng_key, sub_key = jr.split(rng_key)
    y_optimal = proc_lklhd.sample(sub_key, xi_optimal, theta_true)

    rng_key, sub_key = jr.split(rng_key)
    xi_uniform = generate_uniform_design(sub_key, num_design_points)
    rng_key, sub_key = jr.split(rng_key)
    y_uniform = proc_lklhd.sample(sub_key, xi_uniform, theta_true)

    r_opt, r_unif = evaluate_all_alphas(alpha_values, y_optimal, y_uniform, xi_uniform, theta_true)
    rmse_optimal = rmse_optimal.at[:, trial].set(r_opt)
    rmse_uniform = rmse_uniform.at[:, trial].set(r_unif)

print("✓ Complete")

print("\n" + "=" * 80)
print("RESULTS: Posterior Mean RMSE across α")
print("=" * 80)

mean_opt, q10_opt, q90_opt = summarize(rmse_optimal)
mean_unif, q10_unif, q90_unif = summarize(rmse_uniform)

print(f"\n{'α':>8} | {'Optimal: mean [q10, q90]':>32} | {'Uniform: mean [q10, q90]':>32}")
print("-" * 82)
for idx in range(num_alphas):
    print(
        f"{float(alpha_values[idx]):>8.3f} | "
        f"{float(mean_opt[idx]):>8.4f} [{float(q10_opt[idx]):.4f}, {float(q90_opt[idx]):.4f}] | "
        f"{float(mean_unif[idx]):>8.4f} [{float(q10_unif[idx]):.4f}, {float(q90_unif[idx]):.4f}]"
    )

print("\n" + "=" * 80)

# ============================================================================
# CSV
# ============================================================================

csv_path = "examples/linear_regression/data/linreg_mse_studentt.csv"
with open(csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["alpha", "mean_opt", "q10_opt", "q90_opt", "mean_unif", "q10_unif", "q90_unif"])
    for idx in range(num_alphas):
        writer.writerow(
            [
                float(alpha_values[idx]),
                float(mean_opt[idx]),
                float(q10_opt[idx]),
                float(q90_opt[idx]),
                float(mean_unif[idx]),
                float(q10_unif[idx]),
                float(q90_unif[idx]),
            ]
        )
