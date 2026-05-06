"""
Posterior Predictive Performance for Misspecified Linear Regression
"""

from typing import NamedTuple

import csv
import numpy as np

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

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


def custom_split(rng_key: Array, num_splits: int) -> tuple[Array, Array]:
    """Split the RNG key into num_splits parts."""
    keys = jr.split(rng_key, num_splits)
    return keys[0], keys[1:]


def build_alpha_grid() -> Array:
    """Log-spaced robustness orders from 1e-2 to 1."""
    return jnp.asarray(np.geomspace(1e-2, 1.0, 21), dtype=jnp.float64)


rng_key = jr.PRNGKey(42)

# Alpha values to evaluate
alpha_values = build_alpha_grid()
num_alphas = len(alpha_values)

# Experiment parameters
num_design_points = 2
num_test_points = 32
num_trials = int(4 * 1024)

# Noise scale
dim_theta = 2
base_stddev = jnp.sqrt(1e-2)

# Create true process (Student-t)
df_true = 5.0

proc_prior_mean = jnp.array([1.5, -1.5])
proc_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
proc_prior = Gaussian(mean=proc_prior_mean, cov=proc_prior_cov)

proc_stddev = base_stddev * 2 / 5e-1
proc_lklhd = LinearStudentT(df=df_true, noise_stddev=proc_stddev)

# Create model (Gaussian)
mdl_prior_mean = jnp.array([1.5, -1.5])
mdl_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
mdl_prior = Gaussian(mean=mdl_prior_mean, cov=mdl_prior_cov)

mdl_lklhd = LinearGaussian(noise_stddev=base_stddev)
model = LinearRegression(prior=mdl_prior, likelihood=mdl_lklhd)

print("=" * 80)
print("PREDICTIVE PERFORMANCE EXPERIMENT: STUDENT-T TRUE PROCESS")
print("=" * 80)
print(f"TRUE PROCESS: Student-t with df={df_true}")
print(f"  Noise: σ² = {proc_stddev**2:.4f}")
print("\nMODEL: LinearGaussian")
print("  Robust learning:   q(θ|x) ∝ p(θ) p(x|θ)^α")
print("  Robust prediction: σ²/α")
print("\nComparison rule")
print("  Sibson-optimized two-point boundary design vs uniform random design")
print(f"\nDesign points:     {num_design_points}")
print(f"Test points:       {num_test_points}")
print(f"Number of trials:  {num_trials}")
print("=" * 80)

print("\n" + "=" * 80)
print("RUNNING EXPERIMENTS")
print("=" * 80)

elpd_optimal = jnp.zeros((num_alphas, num_trials))
elpd_uniform = jnp.zeros((num_alphas, num_trials))

xi_optimal = generate_sibson_optimal_design(num_design_points)
print(f"Sibson-optimized design: {xi_optimal}")
print(f"\nRunning {num_trials} trials...")

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

    rng_key, sub_keys = custom_split(rng_key, num_test_points + 1)
    xi_test = jax.vmap(lambda k: generate_uniform_design(k, num_design_points))(sub_keys)

    rng_key, sub_keys = custom_split(rng_key, num_test_points + 1)
    y_test = jax.vmap(lambda k, xi: proc_lklhd.sample(k, xi, theta_true))(sub_keys, xi_test)

    def evaluate_alpha(alpha):
        posterior_opt = model.get_tilted_posterior(y_optimal, xi_optimal, alpha)
        log_preds_opt = jax.vmap(
            lambda xi_i, y_i: model.get_posterior_predictive(posterior_opt, xi_i, alpha).logpdf(y_i)
        )(xi_test, y_test)

        posterior_unif = model.get_tilted_posterior(y_uniform, xi_uniform, alpha)
        log_preds_unif = jax.vmap(
            lambda xi_i, y_i: model.get_posterior_predictive(posterior_unif, xi_i, alpha).logpdf(y_i)
        )(xi_test, y_test)

        return jnp.mean(log_preds_opt), jnp.mean(log_preds_unif)

    elpds_opt, elpds_unif = jax.vmap(evaluate_alpha)(alpha_values)
    elpd_optimal = elpd_optimal.at[:, trial].set(elpds_opt)
    elpd_uniform = elpd_uniform.at[:, trial].set(elpds_unif)

print("✓ Complete")

print("\n" + "=" * 80)
print("RESULTS: ELPD across α")
print("=" * 80)

mean_elpd_opt = jnp.mean(elpd_optimal, axis=1)
std_elpd_opt = jnp.std(elpd_optimal, axis=1)
se_elpd_opt = std_elpd_opt / jnp.sqrt(num_trials)
mean_elpd_unif = jnp.mean(elpd_uniform, axis=1)
std_elpd_unif = jnp.std(elpd_uniform, axis=1)
se_elpd_unif = std_elpd_unif / jnp.sqrt(num_trials)

print(f"\n{'α':>8} | {'Sibson Opt (mean ± SE)':>23} | {'Uniform (mean ± SE)':>20}")
print("-" * 58)
for alpha_target in [0.01, 0.05, 0.1, 0.5, 1.0]:
    idx = jnp.argmin(jnp.abs(alpha_values - alpha_target))
    print(
        f"{alpha_values[idx]:>8.3f} | "
        f"{mean_elpd_opt[idx]:>10.4f} ± {se_elpd_opt[idx]:<7.4f} | "
        f"{mean_elpd_unif[idx]:>10.4f} ± {se_elpd_unif[idx]:<7.4f}"
    )

print("\n" + "=" * 80)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

q05_opt = jnp.percentile(elpd_optimal, 5, axis=1)
q95_opt = jnp.percentile(elpd_optimal, 95, axis=1)
q05_unif = jnp.percentile(elpd_uniform, 5, axis=1)
q95_unif = jnp.percentile(elpd_uniform, 95, axis=1)

csv_path = "examples/linear_regression/data/linreg_elpd_studentt.csv"
with open(csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "alpha",
            "mean_opt",
            "se_opt",
            "q05_opt",
            "q95_opt",
            "mean_unif",
            "se_unif",
            "q05_unif",
            "q95_unif",
        ]
    )
    for idx in range(num_alphas):
        writer.writerow(
            [
                float(alpha_values[idx]),
                float(mean_elpd_opt[idx]),
                float(se_elpd_opt[idx]),
                float(q05_opt[idx]),
                float(q95_opt[idx]),
                float(mean_elpd_unif[idx]),
                float(se_elpd_unif[idx]),
                float(q05_unif[idx]),
                float(q95_unif[idx]),
            ]
        )

ax.plot(alpha_values, mean_elpd_opt, "-", color="0.1", lw=2.5, label="Sibson optimal")
ax.fill_between(alpha_values, q05_opt, q95_opt, alpha=0.2, color="0.1")

ax.plot(alpha_values, mean_elpd_unif, "-", color="0.5", lw=2.5, label="Uniform")
ax.fill_between(alpha_values, q05_unif, q95_unif, alpha=0.2, color="0.5")

ax.axvline(1.0, color="gray", linestyle=":", lw=1.5, label="Nominal (α=1)")

ax.set_xscale("log")
ax.set_xlabel("α", fontsize=12)
ax.set_ylabel("ELPD", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
