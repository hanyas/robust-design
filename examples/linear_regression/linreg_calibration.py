"""
Compare nominal vs robust Bayesian inference for linear regression.

Evaluates the trade-off between information gain and calibration when the true
data-generating process differs from the nominal model. Shows that robust posteriors 
sacrifice information gain for better calibrated credible intervals.
"""

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as random
from scipy.stats import chi2

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
) -> float:
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


# Configuration
key = random.PRNGKey(42)

alpha_worst = 0.1       # Worst-case model
alpha_true = 0.4        # True process
alpha_nominal = 1.0     # Nominal model
noise_var = 0.1
num_samples = 5000

# Design
key, sub_key = random.split(key)
xi = jnp.array([-1.0, 1.0])

# Prior
prior_mean = jnp.zeros(2)
prior_cov = 0.1 * jnp.eye(2)
dim_theta = prior_mean.shape[0]

# Create models
prior = Gaussian(mean=prior_mean, cov=prior_cov)
likelihood = LinearGaussian(noise_var=noise_var)
model = LinearRegression(prior=prior, likelihood=likelihood)

print("=" * 80)
print("GENERATING DATA FROM TRUE MODEL")
print("=" * 80)

# Generate data using true model (alpha_true)
joint_true = model.get_tilted_joint(xi, alpha=alpha_true)

# Sample from joint distribution
key, sub_key = random.split(key)
joint_samples = joint_true.sample(sub_key, num_samples)
theta_true = joint_samples[:, :dim_theta]
x_true = joint_samples[:, dim_theta:]

print(f"Generated {num_samples} samples from joint distribution")
print(f"  α_true = {alpha_true}")
print(f"  Effective noise: σ²/α = {noise_var/alpha_true:.3f}")

print("\n" + "=" * 80)
print("COMPUTING POSTERIORS")
print("=" * 80)

# Compute posteriors for each sample under different models
def compute_nominal_posterior(x_sample: Array) -> Gaussian:
    return model.get_bayes_posterior(x_sample, xi)

def compute_robust_posterior(x_sample: Array) -> Gaussian:
    return model.get_tilted_posterior(x_sample, xi, alpha=alpha_worst)

def compute_true_posterior(x_sample: Array) -> Gaussian:
    return model.get_tilted_posterior(x_sample, xi, alpha=alpha_true)

# Vectorized computation
posteriors_nominal = jax.vmap(compute_nominal_posterior)(x_true)
posteriors_worst = jax.vmap(compute_robust_posterior)(x_true)
posteriors_true = jax.vmap(compute_true_posterior)(x_true)

# Extract means and covariances
mu_post_nominal = posteriors_nominal.mean
cov_post_nominal = posteriors_nominal.cov
mu_post_worst = posteriors_worst.mean
cov_post_worst = posteriors_worst.cov
mu_post_true = posteriors_true.mean
cov_post_true = posteriors_true.cov

print(f"Computed posteriors for {num_samples} samples")
print(f"  Nominal (α={alpha_nominal})")
print(f"  Worst-case (α={alpha_worst})")
print(f"  True (α={alpha_true})")

print("\n" + "=" * 80)
print("COMPUTING INFORMATION GAINS")
print("=" * 80)

# Compute information gain for each sample
def compute_ig_nominal(x_sample: Array) -> float:
    return model.shannon_ig(x_sample, xi)

def compute_ig_robust(x_sample: Array) -> float:
    return model.renyi_ig(x_sample, xi, alpha=alpha_worst)

def compute_ig_true(x_sample: Array) -> float:
    return model.renyi_ig(x_sample, xi, alpha=alpha_true)

info_gain_nominal = jax.vmap(compute_ig_nominal)(x_true)
info_gain_worst = jax.vmap(compute_ig_robust)(x_true)
info_gain_true = jax.vmap(compute_ig_true)(x_true)

print("Computed information gains")

print("\n" + "=" * 80)
print("COMPUTING CALIBRATION METRICS")
print("=" * 80)

# Mahalanobis distances
mahal_nominal = jax.vmap(compute_mahalanobis_distance)(
    theta_true, mu_post_nominal, cov_post_nominal
)
mahal_worst = jax.vmap(compute_mahalanobis_distance)(
    theta_true, mu_post_worst, cov_post_worst
)
mahal_true = jax.vmap(compute_mahalanobis_distance)(
    theta_true, mu_post_true, cov_post_true
)

# Coverage at 95%
coverage_95_nominal = compute_credible_coverage(mahal_nominal, 0.95, dim_theta)
coverage_95_worst = compute_credible_coverage(mahal_worst, 0.95, dim_theta)
coverage_95_true = compute_credible_coverage(mahal_true, 0.95, dim_theta)

# Calibration curves
coverage_levels = jnp.linspace(0.1, 0.99, 20)
cal_curve_nominal = compute_calibration_curve(mahal_nominal, coverage_levels, dim_theta)
cal_curve_worst = compute_calibration_curve(mahal_worst, coverage_levels, dim_theta)
cal_curve_true = compute_calibration_curve(mahal_true, coverage_levels, dim_theta)

print("Computed calibration metrics")

print("\n" + "=" * 80)
print("COMPUTING EXPECTED INFORMATION GAINS")
print("=" * 80)

# Expected information gains (Shannon's and Sibson's MI)
eig_nominal = model.shannon_mi(xi)
eig_worst = model.sibson_mi(xi, alpha=alpha_worst)
eig_true = model.sibson_mi(xi, alpha=alpha_true)

print(f"Expected IG (Shannon/Sibson MI):")
print(f"  Nominal (α={alpha_nominal}):  {eig_nominal:.4f}")
print(f"  Worst-case (α={alpha_worst}): {eig_worst:.4f}")
print(f"  True (α={alpha_true}):        {eig_true:.4f}")

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print("\n" + "-" * 80)
print("EXPERIMENTAL SETUP")
print("-" * 80)
print(f"Design points:  {xi.shape[0]}")
print(f"Samples:        {num_samples}")
print(f"\nModel specifications:")
print(f"  Nominal (α={alpha_nominal:.1f}):      σ²/α = {noise_var:.3f}")
print(f"  Worst-case (α={alpha_worst:.1f}):  σ²/α = {noise_var/alpha_worst:.3f}")
print(f"  True (α={alpha_true:.1f}):        σ²/α = {noise_var/alpha_true:.3f}")

print("\n" + "-" * 80)
print("INFORMATION GAIN ANALYSIS")
print("-" * 80)
print("Expected information gain (Shannon's/Sibson's MI):")
print(f"  Nominal model:     {eig_nominal:.4f}")
print(f"  Worst-case model:  {eig_worst:.4f}")
print(f"  True model:        {eig_true:.4f}")
print(f"\nRealized information gain (empirical mean ± std):")
print(f"  Nominal posterior:     {jnp.mean(info_gain_nominal):.4f} ± {jnp.std(info_gain_nominal):.4f}")
print(f"  Worst-case posterior:  {jnp.mean(info_gain_worst):.4f} ± {jnp.std(info_gain_worst):.4f}")
print(f"  True posterior:        {jnp.mean(info_gain_true):.4f} ± {jnp.std(info_gain_true):.4f}")

print("\n" + "-" * 80)
print("POSTERIOR CALIBRATION ANALYSIS")
print("-" * 80)
print(f"Mahalanobis distance M² (expected: χ²_{dim_theta} with mean {dim_theta:.0f}):")
print(f"  Nominal posterior:     {jnp.mean(mahal_nominal):.2f} ± {jnp.std(mahal_nominal):.2f}  ({jnp.mean(mahal_nominal)/dim_theta:.2f}× expected)")
print(f"  Worst-case posterior:  {jnp.mean(mahal_worst):.2f} ± {jnp.std(mahal_worst):.2f}  ({jnp.mean(mahal_worst)/dim_theta:.2f}× expected)")
print(f"  True posterior:        {jnp.mean(mahal_true):.2f} ± {jnp.std(mahal_true):.2f}  ({jnp.mean(mahal_true)/dim_theta:.2f}× expected)")
print(f"\n95% credible region coverage (expected: 95.0%):")
print(f"  Nominal posterior:     {100*coverage_95_nominal:.1f}%  {'✗ under-coverage' if coverage_95_nominal < 0.94 else '✓'}")
print(f"  Worst-case posterior:  {100*coverage_95_worst:.1f}%  {'✓ good coverage' if 0.94 <= coverage_95_worst <= 0.96 else '• over-coverage'}")
print(f"  True posterior:        {100*coverage_95_true:.1f}%  {'✓ good coverage' if 0.94 <= coverage_95_true <= 0.96 else ''}")

print("\n" + "-" * 80)
print("SUMMARY: TRADE-OFF BETWEEN INFORMATION AND CALIBRATION")
print("-" * 80)
print("Nominal (α=1.0) - assumes less noise than reality:")
print(f"  Information gain:  {jnp.mean(info_gain_nominal):.3f}  (highest)")
print(f"  Calibration:       M²={jnp.mean(mahal_nominal):.2f}, coverage={100*coverage_95_nominal:.1f}%  (overconfident)")
print()
print(f"Worst-case (α={alpha_worst:.1f}) - assumes more noise than reality:")
print(f"  Information gain:  {jnp.mean(info_gain_worst):.3f}  (lowest)")
print(f"  Calibration:       M²={jnp.mean(mahal_worst):.2f}, coverage={100*coverage_95_worst:.1f}%  (conservative)")
print()
print(f"True (α={alpha_true:.1f}) - knows the true noise level:")
print(f"  Information gain:  {jnp.mean(info_gain_true):.3f}  (optimal)")
print(f"  Calibration:       M²={jnp.mean(mahal_true):.2f}, coverage={100*coverage_95_true:.1f}%  (well-calibrated)")
print("\n" + "=" * 80)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# Panel 1: Information Gain
ax = axes[0]
all_kl = jnp.concatenate([info_gain_nominal, info_gain_worst, info_gain_true])
bins = jnp.linspace(all_kl.min() - 0.2, all_kl.max() + 0.2, 60)
ax.hist(info_gain_nominal, bins=bins, alpha=0.5, color='steelblue', edgecolor='black', linewidth=0.5, label='Nominal')
ax.hist(info_gain_worst, bins=bins, alpha=0.5, color='seagreen', edgecolor='black', linewidth=0.5, label='Worst-case')
ax.hist(info_gain_true, bins=bins, alpha=0.5, color='orange', edgecolor='black', linewidth=0.5, label='True')
ax.set_xlabel('Realized Info Gain', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Information Gain', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 2: Mahalanobis Distance
ax = axes[1]
all_mahal = jnp.concatenate([mahal_nominal, mahal_worst, mahal_true])
bins_mahal = jnp.linspace(0, jnp.percentile(all_mahal, 98), 60)

ax.hist(mahal_nominal, bins=bins_mahal, histtype='step', color='steelblue', linewidth=2.5, label='Nominal', density=True)
ax.hist(mahal_worst, bins=bins_mahal, histtype='step', color='seagreen', linewidth=2.5, label='Worst-case', density=True)
ax.hist(mahal_true, bins=bins_mahal, histtype='step', color='orange', linewidth=2.5, label='True', density=True)

x_chi2 = jnp.linspace(0, jnp.percentile(all_mahal, 98), 200)
ax.plot(x_chi2, chi2.pdf(x_chi2, df=dim_theta), 'k--', lw=2.5, label=f'χ²_{dim_theta}')

ax.set_xlabel('Mahalanobis Distance', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Calibration Test', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 3: Calibration Curve
ax = axes[2]
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect', alpha=0.7)
ax.plot(coverage_levels, cal_curve_nominal, 'o-', color='steelblue', lw=2, markersize=5, label='Nominal')
ax.plot(coverage_levels, cal_curve_worst, 's-', color='seagreen', lw=2, markersize=5, label='Worst-case')
ax.plot(coverage_levels, cal_curve_true, '^-', color='orange', lw=2, markersize=5, label='True')
ax.set_xlabel('Nominal Level', fontsize=12)
ax.set_ylabel('Actual Coverage', fontsize=12)
ax.set_title('Calibration Curve', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()