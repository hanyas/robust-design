"""
Posterior Predictive Performance for Misspecified Linear Regression
"""
from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian, LinearRegression

import matplotlib.pyplot as plt
import csv

jax.config.update("jax_enable_x64", True)


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

# Alpha values to evaluate
decades = 10.0 ** jnp.arange(-2, 0)
alpha_values = jnp.concatenate([jnp.arange(1, 11) * d for d in decades])
num_alphas = len(alpha_values)

# Experiment parameters
num_design_points = 2
num_test_points = 32
num_trials = 1024

# Noise scale
dim_theta = 2
base_stddev = jnp.sqrt(1e-2)

# Create true process
alpha_true = 5e-1

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
print("PREDICTIVE PERFORMANCE EXPERIMENT: GAUSSIAN TRUE PROCESS")
print("=" * 80)
print(f"TRUE PROCESS: Gaussian with σ² = {proc_stddev**2:.4f}")
print(f"\nMODEL: LinearGaussian")
print(f"  Robust learning:   q(θ|x) ∝ p(θ) p(x|θ)^α")
print(f"  Robust prediction: σ²/α")
print(f"\nDesign points:     {num_design_points}")
print(f"Test points:       {num_test_points}")
print(f"Number of trials:  {num_trials}")
print("=" * 80)

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
    theta_true = process.prior.sample(sub_key, 1)[0]

    # ========== OPTIMAL DESIGN ==========
    xi_optimal = generate_optimal_design(num_design_points)
    rng_key, sub_key = jr.split(rng_key)
    y_optimal = process.likelihood.sample(sub_key, xi_optimal, theta_true)

    # ========== UNIFORM DESIGN ==========
    rng_key, sub_key = jr.split(rng_key)
    xi_uniform = generate_uniform_design(sub_key, num_design_points)
    rng_key, sub_key = jr.split(rng_key)
    y_uniform = process.likelihood.sample(sub_key, xi_uniform, theta_true)

    # ========== TEST POINT ==========
    rng_key, sub_keys = custom_split(rng_key, num_test_points + 1)
    xi_test = jax.vmap(lambda k: generate_uniform_design(k, num_design_points))(sub_keys)

    # Sample test observations for each test design
    rng_key, sub_keys = custom_split(rng_key, num_test_points + 1)
    y_test = jax.vmap(lambda k, xi: process.likelihood.sample(k, xi, theta_true))(sub_keys, xi_test)

    # ========== EVALUATE ALL ALPHAS ==========
    def evaluate_alpha(alpha):
        # Optimal design
        posterior_opt = model.get_tilted_posterior(y_optimal, xi_optimal, alpha)
        log_preds_opt = jax.vmap(
            lambda xi_i, y_i: model.get_posterior_predictive(posterior_opt, xi_i, alpha).logpdf(y_i)
        )(xi_test, y_test)

        # Uniform design
        posterior_unif = model.get_tilted_posterior(y_uniform, xi_uniform, alpha)
        log_preds_unif = jax.vmap(
            lambda xi_i, y_i: model.get_posterior_predictive(posterior_unif, xi_i, alpha).logpdf(y_i)
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
print("RESULTS: ELPD across α")
print("=" * 80)

# Statistics
mean_elpd_opt = jnp.mean(elpd_optimal, axis=1)
std_elpd_opt = jnp.std(elpd_optimal, axis=1)
se_elpd_opt = std_elpd_opt / jnp.sqrt(num_trials)
mean_elpd_unif = jnp.mean(elpd_uniform, axis=1)
std_elpd_unif = jnp.std(elpd_uniform, axis=1)
se_elpd_unif = std_elpd_unif / jnp.sqrt(num_trials)

print(f"\n{'α':>8} | {'Optimal (mean ± SE)':>20} | {'Uniform (mean ± SE)':>20}")
print("-" * 55)
for alpha_target in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    idx = jnp.argmin(jnp.abs(alpha_values - alpha_target))
    print(f"{alpha_values[idx]:>8.3f} | {mean_elpd_opt[idx]:>10.4f} ± {se_elpd_opt[idx]:<7.4f} | {mean_elpd_unif[idx]:>10.4f} ± {se_elpd_unif[idx]:<7.4f}")

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

# Save summary statistics for LaTeX plotting
csv_path = "examples/linear_regression/data/linreg_elpd.csv"
with open(csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        "alpha",
        "mean_opt",
        "se_opt",
        "q05_opt",
        "q95_opt",
        "mean_unif",
        "se_unif",
        "q05_unif",
        "q95_unif",
    ])
    for idx in range(num_alphas):
        writer.writerow([
            float(alpha_values[idx]),
            float(mean_elpd_opt[idx]),
            float(se_elpd_opt[idx]),
            float(q05_opt[idx]),
            float(q95_opt[idx]),
            float(mean_elpd_unif[idx]),
            float(se_elpd_unif[idx]),
            float(q05_unif[idx]),
            float(q95_unif[idx]),
        ])

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
