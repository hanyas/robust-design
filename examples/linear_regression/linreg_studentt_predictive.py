"""
Posterior Predictive Performance for Misspecified Linear Regression

TRUE PROCESS: Student-t
MODELS: LinearGaussian with α-tilted posterior and predictive

This ensures the caution exercised during learning is consistently applied to prediction.
"""
from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.scipy import special

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian, LinearRegression

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class LinearStudentT(NamedTuple):
    """
    Student-t linear likelihood for regression.

    Args:
        df: Degrees of freedom
        noise_stddev: Noise standard deviation
    """
    df: float
    noise_stddev: float

    def design_matrix(self, design: Array) -> Array:
        """Construct augmented design matrix H(ξ) = [ξ, 1]."""
        design = jnp.atleast_1d(design)
        size = design.shape[0]
        return jnp.column_stack([design, jnp.ones(size)])

    def mean(self, design: Array, theta: Array) -> Array:
        """Compute mean H(ξ)θ."""
        H = self.design_matrix(design)
        return H @ theta

    def sample(self, rng_key: Array, design: Array, theta: Array) -> Array:
        """Sample from multivariate Student-t likelihood."""
        size = design.shape[0]
        loc = self.mean(design, theta)
        scale_matrix = self.noise_stddev**2 * jnp.eye(size)

        # Sample via scale mixture of Gaussians
        rng_key, key1, key2 = jr.split(rng_key, 3)
        chi2_samples = jr.gamma(key1, self.df / 2, shape=(size,)) * 2

        L = jnp.linalg.cholesky(scale_matrix)
        z = jr.normal(key2, (size,))
        normal_samples = loc + L @ z

        scale_factors = jnp.sqrt(self.df / chi2_samples)
        return loc + (normal_samples - loc) * scale_factors


def generate_optimal_design(num_designs: int) -> Array:
    """Generate optimal design: alternating -1 and 1."""
    return jnp.array([(-1.0)**(i+1) for i in range(num_designs)])


def generate_uniform_design(rng_key: Array, num_designs: int) -> Array:
    """Generate uniform random design in [-1, 1]."""
    return jr.uniform(rng_key, shape=(num_designs,), minval=-1.0, maxval=1.0)


def custom_split(rng_key: Array, num_splits: int) -> Array:
    """Custom split function that splits the RNG key into num_splits parts."""
    keys = jr.split(rng_key, num_splits)
    return keys[0], keys[1:]

# ============================================================================
# Configuration
# ============================================================================

rng_key = jr.PRNGKey(42)

# True process: Student-t
df_true = 3.0
noise_stddev = jnp.sqrt(1e-1)

# Alpha values to evaluate
alpha_values = jnp.logspace(-2, 0, 25)
num_alphas = len(alpha_values)

# Experiment parameters
num_design_points = 2
num_test_points = 32
num_trials = 10_000

# Prior
prior_mean = jnp.zeros(2)
prior_cov = jnp.diag(jnp.array([3.0, 2.0]))  # jnp.eye(2)

# Create true process (Student-t)
process = LinearStudentT(df=df_true, noise_stddev=noise_stddev)

# Create model (Gaussian)
prior = Gaussian(mean=prior_mean, cov=prior_cov)
model = LinearGaussian(noise_stddev=noise_stddev)
linreg = LinearRegression(prior=prior, likelihood=model)

print("=" * 80)
print("PREDICTIVE PERFORMANCE EXPERIMENT: STUDENT-T TRUE PROCESS")
print("=" * 80)
print(f"TRUE PROCESS: Student-t with df={df_true}")
print(f"  Noise: σ² = {noise_stddev**2:.4f}")
print(f"\nMODEL: LinearGaussian")
print(f"  α-tilted posterior: q(θ|x) ∝ p(θ) p(x|θ)^α")
print(f"  α-inflated predictive: σ²/α")
print(f"\nDesign points:     {num_design_points}")
print(f"Test points:         {num_test_points}")
print(f"Number of trials:    {num_trials}")
print(f"Alpha values:        {num_alphas}")

# ============================================================================
# Run Experiments
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING EXPERIMENTS")
print("=" * 80)

# Storage for results
elpd_optimal = jnp.zeros((num_alphas, num_trials))
elpd_uniform = jnp.zeros((num_alphas, num_trials))

print(f"\nRunning {num_trials} trials...")

for trial in range(num_trials):
    if (trial + 1) % 200 == 0 or trial == 0:
        print(f"  Trial {trial + 1}/{num_trials}...")

    # Sample true parameter from prior
    rng_key, sub_key = jr.split(rng_key)
    theta_true = prior.sample(sub_key, 1)[0]

    # ========== OPTIMAL DESIGN ==========
    xi_optimal = generate_optimal_design(num_design_points)
    rng_key, sub_key = jr.split(rng_key)
    y_optimal = process.sample(sub_key, xi_optimal, theta_true)

    # ========== UNIFORM DESIGN ==========
    rng_key, sub_key = jr.split(rng_key)
    xi_uniform = generate_uniform_design(sub_key, num_design_points)
    rng_key, sub_key = jr.split(rng_key)
    y_uniform = process.sample(sub_key, xi_uniform, theta_true)

    # ========== TEST POINT ==========
    rng_key, sub_keys = custom_split(rng_key, num_test_points + 1)
    xi_test = jax.vmap(lambda k: generate_uniform_design(k, num_design_points))(sub_keys)
    
    # Sample test observations for each test design
    rng_key, sub_keys = custom_split(rng_key, num_test_points + 1)
    y_test = jax.vmap(lambda k, xi: process.sample(k, xi, theta_true))(sub_keys, xi_test)

    # ========== EVALUATE ALL ALPHAS ==========
    def evaluate_alpha(alpha):
        # Optimal design
        posterior_opt = linreg.get_tilted_posterior(y_optimal, xi_optimal, alpha)
        log_preds_opt = jax.vmap(
            lambda xi_i, y_i: linreg.get_posterior_predictive(posterior_opt, xi_i, alpha).logpdf(y_i)
        )(xi_test, y_test)
        
        # Uniform design
        posterior_unif = linreg.get_tilted_posterior(y_uniform, xi_uniform, alpha)
        log_preds_unif = jax.vmap(
            lambda xi_i, y_i: linreg.get_posterior_predictive(posterior_unif, xi_i, alpha).logpdf(y_i)
        )(xi_test, y_test)
        
        return jnp.mean(log_preds_opt), jnp.mean(log_preds_unif)
    
    elpds_opt, elpds_unif = jax.vmap(evaluate_alpha)(alpha_values)
    elpd_optimal = elpd_optimal.at[:, trial].set(elpds_opt)
    elpd_uniform = elpd_uniform.at[:, trial].set(elpds_unif)

print("✓ Complete")

# ============================================================================
# Analysis
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

# Statistics
mean_elpd_opt = jnp.mean(elpd_optimal, axis=1)
std_elpd_opt = jnp.std(elpd_optimal, axis=1)
mean_elpd_unif = jnp.mean(elpd_uniform, axis=1)
std_elpd_unif = jnp.std(elpd_uniform, axis=1)

print("\n" + "-" * 80)
print("ELPD AT SELECTED α VALUES")
print("-" * 80)
print(f"\n{'α':>8} | {'Optimal':>20} | {'Uniform':>20}")
print("-" * 55)
for alpha_target in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    idx = jnp.argmin(jnp.abs(alpha_values - alpha_target))
    print(f"{alpha_values[idx]:>8.3f} | {mean_elpd_opt[idx]:>10.4f} ± {std_elpd_opt[idx]:<7.4f} | {mean_elpd_unif[idx]:>10.4f} ± {std_elpd_unif[idx]:<7.4f}")

print("\n" + "=" * 80)

# ============================================================================
# Visualization
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Quantiles
q05_opt = jnp.percentile(elpd_optimal, 5, axis=1)
q95_opt = jnp.percentile(elpd_optimal, 95, axis=1)
q05_unif = jnp.percentile(elpd_uniform, 5, axis=1)
q95_unif = jnp.percentile(elpd_uniform, 95, axis=1)

# Optimal design (dark)
ax.plot(alpha_values, mean_elpd_opt, '-', color='0.1', lw=2.5, label='Optimal')
ax.fill_between(alpha_values, q05_opt, q95_opt, alpha=0.2, color='0.1')

# Uniform design (grey)
ax.plot(alpha_values, mean_elpd_unif, '-', color='0.5', lw=2.5, label='Uniform')
ax.fill_between(alpha_values, q05_unif, q95_unif, alpha=0.2, color='0.5')

# Nominal reference
ax.axvline(1.0, color='gray', linestyle=':', lw=1.5, label='Nominal (α=1)')

ax.set_xscale('log')
ax.set_xlabel('α', fontsize=12)
ax.set_ylabel('ELPD', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()