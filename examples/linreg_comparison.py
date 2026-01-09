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
import matplotlib.pyplot as plt
from scipy.stats import chi2


jax.config.update("jax_enable_x64", True)


def get_design_matrix(xi: Array) -> Array:
    """Construct design matrix with intercept column and design points."""
    return jnp.column_stack([jnp.ones_like(xi), xi])


def get_joint_distribution(
    H: Array,
    prior_mean: Array,
    prior_cov: Array,
    noise_var: float,
    alpha: float,
) -> tuple[Array, Array]:
    """Compute joint distribution of parameters and observations.
    
    For standard linear Gaussian model:
    theta ~ N(mu, Sigma), x | theta ~ N(H*theta, (sigma^2/alpha)*I)
    
    Args:
        H: Design matrix
        prior_mean: Prior mean
        prior_cov: Prior covariance
        noise_var: Observation noise variance
        alpha: Sibson order parameter
    
    Returns:
        Joint mean and covariance
    """
    N = H.shape[0]
    
    # Standard linear Gaussian joint (not Sibson-modified)
    cov_theta_x = prior_cov @ H.T
    cov_x = H @ prior_cov @ H.T + (noise_var / alpha) * jnp.eye(N)
    
    # Joint distribution
    joint_mean = jnp.concatenate([prior_mean, H @ prior_mean])
    joint_cov = jnp.block([
        [prior_cov, cov_theta_x],
        [cov_theta_x.T, cov_x]
    ])
    
    return joint_mean, joint_cov


def sample_from_joint(
    rng_key: Array,
    joint_mean: Array,
    joint_cov: Array,
    num_samples: int,
    dim_theta: int,
) -> tuple[Array, Array]:
    """Sample parameters and observations from joint distribution.
    
    Args:
        rng_key: JAX random key
        joint_mean: Joint mean vector
        joint_cov: Joint covariance matrix
        num_samples: Number of samples to draw
        dim_theta: Dimension of parameter vector
    
    Returns:
        Parameter samples and observation samples
    """
    d = joint_mean.shape[0]
    
    L = jnp.linalg.cholesky(joint_cov)
    z = random.normal(rng_key, (num_samples, d))
    samples = joint_mean + z @ L.T
    
    return samples[:, :dim_theta], samples[:, dim_theta:]


def get_marginal_covariance(
    H: Array,
    prior_cov: Array,
    noise_var: float,
    alpha: float,
) -> Array:
    """Compute marginal covariance of observations.
    
    Args:
        H: Design matrix
        prior_cov: Prior covariance
        noise_var: Observation noise variance
        alpha: Sibson order parameter
    
    Returns:
        Marginal covariance matrix
    """
    return H @ prior_cov @ H.T + (noise_var / alpha) * jnp.eye(H.shape[0])


def get_posterior_distribution(
    x: Array,
    H: Array,
    prior_mean: Array,
    prior_cov: Array,
    noise_var: float,
    alpha: float,
) -> tuple[Array, Array]:
    """Compute posterior distribution given observations.
    
    Args:
        x: Observation vector
        H: Design matrix
        prior_mean: Prior mean
        prior_cov: Prior covariance
        noise_var: Observation noise variance
        alpha: Sibson order parameter
    
    Returns:
        Posterior mean and covariance
    """
    prec_prior = jnp.linalg.inv(prior_cov)
    prec_lik = (alpha / noise_var) * H.T @ H
    cov_post = jnp.linalg.inv(prec_prior + prec_lik)
    mean_post = cov_post @ (prec_prior @ prior_mean + (alpha / noise_var) * H.T @ x)
    return mean_post, cov_post


def compute_kl_divergence(
    mu_post: Array,
    cov_post: Array,
    mu_prior: Array,
    cov_prior: Array,
) -> float:
    """Compute KL divergence between posterior and prior Gaussian distributions.
    
    Args:
        mu_post: Posterior mean
        cov_post: Posterior covariance
        mu_prior: Prior mean
        cov_prior: Prior covariance
    
    Returns:
        KL divergence value
    """
    d = mu_post.shape[0]
    cov_prior_inv = jnp.linalg.inv(cov_prior)
    diff = mu_prior - mu_post
    return 0.5 * (
        jnp.trace(cov_prior_inv @ cov_post)
        + diff @ cov_prior_inv @ diff
        - d
        + jnp.linalg.slogdet(cov_prior)[1] - jnp.linalg.slogdet(cov_post)[1]
    )


def compute_expected_information_gain(
    H: Array,
    prior_mean: Array,
    prior_cov: Array,
    noise_var: float,
    marginal_cov: Array,
    alpha_post: float,
) -> float:
    """Compute expected information gain (expected KL divergence).
    
    Args:
        H: Design matrix
        prior_mean: Prior mean
        prior_cov: Prior covariance
        noise_var: Observation noise variance
        marginal_cov: Marginal covariance under data-generating distribution
        alpha_post: Sibson order parameter for posterior computation
    
    Returns:
        Expected information gain
    """
    d = prior_cov.shape[0]
    prec_prior = jnp.linalg.inv(prior_cov)
    
    # Posterior covariance (deterministic, independent of x)
    prec_post = prec_prior + (alpha_post / noise_var) * H.T @ H
    cov_post = jnp.linalg.inv(prec_post)
    
    # Constant term
    const_term = 0.5 * (
        jnp.trace(prec_prior @ cov_post) 
        - d 
        + jnp.linalg.slogdet(prior_cov)[1] 
        - jnp.linalg.slogdet(cov_post)[1]
    )
    
    # Expected quadratic term
    K = (alpha_post / noise_var) * cov_post @ H.T  # Kalman gain
    quad_term = jnp.trace(K.T @ prec_prior @ K @ marginal_cov)
    
    return const_term + 0.5 * quad_term


def compute_mahalanobis_distance(
    theta: Array,
    mu_post: Array,
    cov_post: Array,
) -> float:
    """Compute Mahalanobis distance between true parameter and posterior mean.
    
    Args:
        theta: True parameter value
        mu_post: Posterior mean
        cov_post: Posterior covariance
    
    Returns:
        Squared Mahalanobis distance
    """
    diff = theta - mu_post
    prec_post = jnp.linalg.inv(cov_post)
    return diff @ prec_post @ diff


def compute_credible_coverage(
    mahalanobis_distances: Array,
    coverage_level: float,
    dim: int,
) -> float:
    """Compute fraction of samples within credible region.
    
    Args:
        mahalanobis_distances: Squared Mahalanobis distances
        coverage_level: Credible level (e.g., 0.95 for 95%)
        dim: Parameter dimension
    
    Returns:
        Fraction of samples within credible region
    """
    threshold = chi2.ppf(coverage_level, df=dim)
    return jnp.mean(mahalanobis_distances <= threshold)


def compute_calibration_curve(
    mahalanobis_distances: Array,
    coverage_levels: Array,
    dim: int,
) -> Array:
    """Compute actual coverage at different credible levels.
    
    Args:
        mahalanobis_distances: Squared Mahalanobis distances
        dim: Parameter dimension
        coverage_levels: Credible levels to evaluate
    
    Returns:
        Actual coverage fractions at each level
    """
    coverages = []
    for level in coverage_levels:
        coverage = compute_credible_coverage(mahalanobis_distances, level, dim)
        coverages.append(coverage)
    return jnp.array(coverages)


# Configuration
key = random.PRNGKey(42)

alpha_worst = 0.3      # Worst-case model
alpha_true = 0.5        # True process
alpha_nominal = 1.0     # Nominal model
noise_var = 0.1
num_samples = 5000

# Design
key, sub_key = random.split(key)
# xi = random.uniform(sub_key, shape=(10,), minval=-1.0, maxval=1.0)
xi = jnp.array([-1.0, 1.0])
H = get_design_matrix(xi)

# Prior
prior_mean = jnp.zeros(2)
prior_cov = 0.1 * jnp.eye(2)
dim_theta = prior_mean.shape[0]

print("Generating data from TRUE model...")
joint_mean_true, joint_cov_true = get_joint_distribution(
    H=H,
    prior_mean=prior_mean,
    prior_cov=prior_cov,
    noise_var=noise_var,
    alpha=alpha_true,
)

key, sub_key = random.split(key)
theta_true, x_true = sample_from_joint(
    rng_key=sub_key,
    joint_mean=joint_mean_true,
    joint_cov=joint_cov_true,
    num_samples=num_samples,
    dim_theta=dim_theta,
)

print("Computing posteriors...")

def compute_nominal_info_gain(x_sample: Array) -> tuple[float, Array, Array]:
    mu_post, cov_post = get_posterior_distribution(
        x=x_sample,
        H=H,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        noise_var=noise_var,
        alpha=alpha_nominal,
    )
    kl = compute_kl_divergence(mu_post, cov_post, prior_mean, prior_cov)
    return kl, mu_post, cov_post

def compute_robust_info_gain(x_sample: Array) -> tuple[float, Array, Array]:
    mu_post, cov_post = get_posterior_distribution(
        x=x_sample, H=H,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        noise_var=noise_var,
        alpha=alpha_worst,
    )
    kl = compute_kl_divergence(mu_post, cov_post, prior_mean, prior_cov)
    return kl, mu_post, cov_post

def compute_true_info_gain(x_sample: Array) -> tuple[float, Array, Array]:
    mu_post, cov_post = get_posterior_distribution(
        x=x_sample, H=H,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        noise_var=noise_var,
        alpha=alpha_true,
    )
    kl = compute_kl_divergence(mu_post, cov_post, prior_mean, prior_cov)
    return kl, mu_post, cov_post

# Vectorized computation
info_gain_nominal, mu_post_nominal, cov_post_nominal = jax.vmap(compute_nominal_info_gain)(x_true)
info_gain_worst, mu_post_worst, cov_post_worst = jax.vmap(compute_robust_info_gain)(x_true)
info_gain_true, mu_post_true, cov_post_true = jax.vmap(compute_true_info_gain)(x_true)


print("Computing calibration metrics...")

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

# Calibration curve
coverage_levels = jnp.linspace(0.1, 0.99, 20)
cal_curve_nominal = compute_calibration_curve(mahal_nominal, coverage_levels, dim_theta)
cal_curve_worst = compute_calibration_curve(mahal_worst, coverage_levels, dim_theta)
cal_curve_true = compute_calibration_curve(mahal_true, coverage_levels, dim_theta)

print("Computing expected information gains...")

# Get marginal covariances
cov_x_nominal = get_marginal_covariance(H, prior_cov, noise_var, alpha=alpha_nominal)
cov_x_worst = get_marginal_covariance(H, prior_cov, noise_var, alpha=alpha_worst)
cov_x_true = get_marginal_covariance(H, prior_cov, noise_var, alpha=alpha_true)

# Expected IGs under each model
eig_nominal_nominal = compute_expected_information_gain(
    H, prior_mean, prior_cov, noise_var,
    marginal_cov=cov_x_nominal,
    alpha_post=alpha_nominal,
)

eig_worst_worst = compute_expected_information_gain(
    H, prior_mean, prior_cov, noise_var,
    marginal_cov=cov_x_worst,
    alpha_post=alpha_worst,
)

eig_true_true = compute_expected_information_gain(
    H, prior_mean, prior_cov, noise_var,
    marginal_cov=cov_x_true,
    alpha_post=alpha_true,
)

eig_true_nominal = compute_expected_information_gain(
    H, prior_mean, prior_cov, noise_var,
    marginal_cov=cov_x_true,
    alpha_post=alpha_nominal,
)

eig_true_worst = compute_expected_information_gain(
    H, prior_mean, prior_cov, noise_var,
    marginal_cov=cov_x_true,
    alpha_post=alpha_worst,
)


print("\n" + "=" * 80)
print("ROBUST BAYESIAN EXPERIMENTAL DESIGN")
print("=" * 80)

print("\n" + "-" * 80)
print("EXPERIMENTAL SETUP")
print("-" * 80)
print(f"Design points:  {H.shape[0]}")
print(f"Samples:        {num_samples}")
print(f"\nModel specifications:")
print(f"  Nominal (α=1.0):      σ²/α = {noise_var:.3f}")
print(f"  Worst-case (α={alpha_worst:.1f}):  σ²/α = {noise_var/alpha_worst:.3f}")
print(f"  True (α={alpha_true:.1f}):        σ²/α = {noise_var/alpha_true:.3f}")

print("\n" + "-" * 80)
print("INFORMATION GAIN ANALYSIS")
print("-" * 80)
print("Expected information gain:")
print(f"  Nominal model:     {eig_nominal_nominal:.4f} (self-consistent)")
print(f"  Worst-case model:  {eig_worst_worst:.4f} (self-consistent)")
print(f"  True model:        {eig_true_true:.4f} (self-consistent)")
print(f"\nExpected IG when data comes from true model (α={alpha_true:.1f}):")
print(f"  Using nominal posterior:     {eig_true_nominal:.4f}")
print(f"  Using worst-case posterior:  {eig_true_worst:.4f}")
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

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# Panel 1: Information Gain
ax = axes[0]
all_kl = jnp.concatenate([info_gain_nominal, info_gain_worst, info_gain_true])
bins = jnp.linspace(all_kl.min() - 0.2, all_kl.max() + 0.2, 60)
ax.hist(info_gain_nominal, bins=bins, alpha=0.5, color='steelblue',
        edgecolor='black', linewidth=0.5, label='Nominal')
ax.hist(info_gain_worst, bins=bins, alpha=0.5, color='seagreen',
        edgecolor='black', linewidth=0.5, label='Worst-case')
ax.hist(info_gain_true, bins=bins, alpha=0.5, color='orange',
        edgecolor='black', linewidth=0.5, label='True')
ax.set_xlabel('Realized Info Gain', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Information Gain', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 2: Mahalanobis Distance
ax = axes[1]
all_mahal = jnp.concatenate([mahal_nominal, mahal_worst, mahal_true])
bins_mahal = jnp.linspace(0, jnp.percentile(all_mahal, 98), 60)

ax.hist(mahal_nominal, bins=bins_mahal, histtype='step', color='steelblue',
        linewidth=2.5, label='Nominal', density=True)
ax.hist(mahal_worst, bins=bins_mahal, histtype='step', color='seagreen',
        linewidth=2.5, label='Worst-case', density=True)
ax.hist(mahal_true, bins=bins_mahal, histtype='step', color='orange',
        linewidth=2.5, label='True', density=True)

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
ax.plot(coverage_levels, cal_curve_nominal, 'o-', color='steelblue',
        lw=2, markersize=5, label='Nominal')
ax.plot(coverage_levels, cal_curve_worst, 's-', color='seagreen',
        lw=2, markersize=5, label='Worst-case')
ax.plot(coverage_levels, cal_curve_true, '^-', color='orange',
        lw=2, markersize=5, label='True')
ax.set_xlabel('Nominal Level', fontsize=12)
ax.set_ylabel('Actual Coverage', fontsize=12)
ax.set_title('Calibration Curve', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()
