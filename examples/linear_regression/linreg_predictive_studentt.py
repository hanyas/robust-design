"""
Bayesian updating experiment for posterior predictive quality with Student-t true process.

TRUE PROCESS: Student-t (heavy-tailed noise)
MODELS: LinearGaussian (Gaussian noise - MISSPECIFIED)

For each seed:
- Samples true parameter θ from prior
- Generates training data from Student-t process
- Fits LinearGaussian models (misspecified) with different alpha values
- Evaluates posterior predictive at test point
- Computes ELPD and TV distance

This reveals which Gaussian model best approximates the heavy-tailed Student-t reality.
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


class StudentTLinearLikelihood(NamedTuple):
    """
    Student-t linear likelihood for regression (heavy-tailed noise).
    
    Model: x_i | θ, ξ_i ~ Student-t(θ₀ + θ₁ ξ_i, σ², df)
    
    Args:
        noise_var: Noise variance (scale parameter)
        df: Degrees of freedom (controls tail heaviness)
    """
    noise_var: float
    df: float
    
    def design_matrix(self, design: Array) -> Array:
        design = jnp.atleast_1d(design)
        return jnp.column_stack([design, jnp.ones(len(design))])
    
    def mean(self, design: Array, theta: Array) -> Array:
        """Location parameter (mean for df > 1)."""
        H = self.design_matrix(design)
        return H @ theta
    
    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        """Sample from Student-t likelihood."""
        N = len(jnp.atleast_1d(design))
        loc = self.mean(design, theta)
        scale_matrix = self.noise_var * jnp.eye(N)
        
        # Sample from chi-squared
        key, key1, key2 = jr.split(key, 3)
        chi2_samples = jr.gamma(key1, self.df/2, shape=(N,)) * 2
        
        # Sample from multivariate normal
        L = jnp.linalg.cholesky(scale_matrix)
        z = jr.normal(key2, (N,))
        normal_samples = loc + L @ z
        
        # Scale to get Student-t
        scale_factors = jnp.sqrt(self.df / chi2_samples)
        t_samples = loc + (normal_samples - loc) * scale_factors
        
        return t_samples
    
    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """Compute log p(x | θ, ξ) for Student-t."""
        N = len(jnp.atleast_1d(design))
        loc = self.mean(design, theta)
        scale_matrix = self.noise_var * jnp.eye(N)
        
        # Mahalanobis distance
        diff = x - loc
        scale_inv = jnp.linalg.inv(scale_matrix)
        mahal = diff @ scale_inv @ diff
        
        # Log normalizing constant
        log_norm = (
            special.gammaln((self.df + N) / 2)
            - special.gammaln(self.df / 2)
            - (N / 2) * jnp.log(self.df * jnp.pi)
            - 0.5 * jnp.linalg.slogdet(scale_matrix)[1]
        )
        
        # Log density
        return log_norm - ((self.df + N) / 2) * jnp.log(1 + mahal / self.df)


# ============================================================================
# Utility Functions
# ============================================================================

def compute_tv_distance_deterministic(
    mu_true: float,
    var_true: float,
    df_true: float,
    mu_model: float,
    var_model: float,
    num_points: int = 1000,
) -> float:
    """Compute Total Variation distance between Student-t and Gaussian."""
    # Create a grid covering both distributions
    std_true = jnp.sqrt(var_true)
    std_model = jnp.sqrt(var_model)
    
    # Student-t has heavier tails, so extend range
    x_min = mu_true - 10 * std_true
    x_max = mu_true + 10 * std_true
    
    x = jnp.linspace(x_min, x_max, num_points)
    dx = (x_max - x_min) / num_points
    
    # Compute Student-t PDF
    diff_t = x - mu_true
    mahal_t = diff_t**2 / var_true
    log_p_true = (
        special.gammaln((df_true + 1) / 2)
        - special.gammaln(df_true / 2)
        - 0.5 * jnp.log(df_true * jnp.pi * var_true)
        - ((df_true + 1) / 2) * jnp.log(1 + mahal_t / df_true)
    )
    p_true = jnp.exp(log_p_true)
    
    # Compute Gaussian PDF
    log_p_model = -0.5 * jnp.log(2 * jnp.pi * var_model) - 0.5 * (x - mu_model)**2 / var_model
    p_model = jnp.exp(log_p_model)
    
    # TV = (1/2) ∫ |p - q| dx
    tv = 0.5 * jnp.sum(jnp.abs(p_true - p_model) * dx)
    
    return tv


def run_experiment(
    rng_key: Array,
    num_train: int,
    num_test: int,
    model: LinearRegression,
    true_lklhd: StudentTLinearLikelihood,
    alpha_values: Array,
    random_design: bool = False,
) -> tuple[Array, Array]:
    """
    Run single experiment with Student-t true process.
    
    Steps:
    1. Sample true θ from prior
    2. Generate training data from Student-t process
    3. Fit LinearGaussian models (misspecified) for each alpha
    4. Evaluate posterior predictive at test point
    5. Compute ELPD and TV distance
    
    Args:
        rng_key: Random key
        num_train: Number of training points
        num_test: Number of test samples at single test point
        model: LinearRegression model (with Gaussian likelihood)
        true_lklhd: Student-t likelihood (true process)
        alpha_values: Alpha values to evaluate for models
    
    Returns:
        ELPD scores and TV distances for each alpha
    """
    # 1. Sample true parameter
    rng_key, key = jr.split(rng_key)
    theta_true = model.prior.sample(key, 1)[0]
    
    # 2. Generate training designs
    rng_key, key = jr.split(rng_key)
    if random_design:
        xi_train = jr.uniform(key, shape=(num_train,), minval=-1.0, maxval=1.0)
    else:
        xi_train = jnp.concatenate([-jnp.ones(num_train // 2), jnp.ones(num_train - num_train // 2)])
    
    # 3. Sample training data from TRUE Student-t process
    rng_key, key = jr.split(rng_key)
    x_train = true_lklhd.sample(key, xi_train, theta_true)
    
    # 4. Generate test design
    rng_key, key = jr.split(rng_key)
    xi_test = jr.uniform(key, minval=-1.0, maxval=1.0)
    xi_test_array = jnp.array([xi_test])
    
    # 5. True distribution at test point (Student-t)
    true_test_mean = true_lklhd.mean(xi_test_array, theta_true)[0]
    true_test_var = true_lklhd.noise_var
    true_test_df = true_lklhd.df
    
    # 6. Generate test samples from TRUE Student-t process
    rng_key, key = jr.split(rng_key)
    # Sample from 1D Student-t
    chi2_sample = jr.gamma(key, true_test_df/2, shape=(num_test,)) * 2
    rng_key, key = jr.split(rng_key)
    normal_samples = jr.normal(key, (num_test,))
    scale_factors = jnp.sqrt(true_test_df / chi2_sample)
    test_samples = true_test_mean + jnp.sqrt(true_test_var) * normal_samples * scale_factors
    
    # 7. Evaluate each model (different alphas) - VECTORIZED
    def evaluate_alpha(alpha):
        # Fit misspecified Gaussian model
        posterior = model.get_tilted_posterior(x_train, xi_train, alpha)
        predictive = model.get_posterior_predictive(posterior, xi_test_array)
        
        # Extract scalar mean and variance from predictive
        pred_mean = predictive.mean[0]
        pred_var = predictive.cov[0, 0]
        
        # ELPD: log p(test_samples | Gaussian model)
        # Reshape test_samples to (num_test, 1) for multivariate logpdf
        score = jnp.mean(predictive.logpdf(test_samples[:, None]))
        
        # TV distance (deterministic computation)
        tv_distance = compute_tv_distance_deterministic(
            true_test_mean, true_test_var, true_test_df,
            pred_mean, pred_var
        )
        
        return score, tv_distance
    
    # Vectorize over all alpha values at once
    scores, tv_distances = jax.vmap(evaluate_alpha)(alpha_values)
    
    return scores, tv_distances


if __name__ == "__main__":
    print("=" * 80)
    print("POSTERIOR PREDICTIVE EXPERIMENT: STUDENT-T TRUE PROCESS")
    print("=" * 80)
    
    # Configuration
    df_true = 2.5
    noise_var = 0.1
    
    # Prior
    prior = Gaussian(
        mean=jnp.zeros(2),
        cov=jnp.eye(2)
    )
    
    # TRUE process: Student-t (heavy-tailed)
    true_lklhd = StudentTLinearLikelihood(
        noise_var=noise_var,
        df=df_true
    )
    
    # MODEL: Gaussian (MISSPECIFIED)
    model_lklhd = LinearGaussian(noise_var=noise_var)
    model = LinearRegression(prior=prior, likelihood=model_lklhd)
    
    # Experiment parameters
    num_train = 16
    num_test = 1024
    num_seeds = 1024
    num_alphas = 64
    
    alpha_values = jnp.logspace(-2, 0, num_alphas)
    
    print(f"\nTRUE PROCESS: Student-t (df={df_true}, heavy-tailed)")
    print(f"  Noise: σ² = {noise_var}")
    print(f"\nMODELS: LinearGaussian (MISSPECIFIED)")
    print(f"  Noise: constant σ² = {noise_var}")
    print(f"\nSetup:")
    print(f"  Training points: {num_train}")
    print(f"  Test samples:    {num_test}")
    print(f"  Random seeds:    {num_seeds}")
    print(f"  Alpha values:    {num_alphas}")
    
    # Storage
    results_elpd = jnp.zeros((num_alphas, num_seeds))
    results_tv = jnp.zeros((num_alphas, num_seeds))
    
    rng_key = jr.PRNGKey(42)
    
    # JIT compile the experiment function for speed
    run_experiment_jit = jax.jit(run_experiment, static_argnums=(1, 2))
    
    print(f"\nRunning experiments...")
    print(f"  Compiling JIT (first iteration may be slow)...")
    
    for seed in range(num_seeds):
        if (seed + 1) % 100 == 0 or seed == 0:
            print(f"  Seed {seed+1}/{num_seeds}...")
        
        rng_key, key = jr.split(rng_key)
        scores, tv_dists = run_experiment_jit(
            key, num_train, num_test, model, true_lklhd,
            alpha_values
        )
        
        results_elpd = results_elpd.at[:, seed].set(scores)
        results_tv = results_tv.at[:, seed].set(tv_dists)
    
    print("✓ Complete\n")
    
    # ========================================================================
    # Analysis
    # ========================================================================
    
    # Scale TV by sqrt(n)
    sqrt_n = jnp.sqrt(num_train)
    results_tv_scaled = sqrt_n * results_tv
    
    # Statistics
    mean_scores = jnp.mean(results_elpd, axis=1)
    std_scores = jnp.std(results_elpd, axis=1)
    quantile_05 = jnp.percentile(results_elpd, 5, axis=1)
    quantile_95 = jnp.percentile(results_elpd, 95, axis=1)
    
    mean_tv_scaled = jnp.mean(results_tv_scaled, axis=1)
    quantile_05_tv = jnp.percentile(results_tv_scaled, 5, axis=1)
    quantile_95_tv = jnp.percentile(results_tv_scaled, 95, axis=1)
    
    # Find best
    best_alpha_idx = jnp.argmax(mean_scores)
    best_alpha = alpha_values[best_alpha_idx]
    best_score = mean_scores[best_alpha_idx]
    
    nominal_idx = -1
    nominal_alpha = alpha_values[nominal_idx]
    nominal_score = mean_scores[nominal_idx]
    
    best_tv_idx = jnp.argmin(mean_tv_scaled)
    best_tv_alpha = alpha_values[best_tv_idx]
    best_tv_value = mean_tv_scaled[best_tv_idx]
    nominal_tv_scaled = mean_tv_scaled[nominal_idx]
    
    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\nTrue process: Student-t with df={df_true} (heavy-tailed)")
    print(f"Setup: {num_train} training points, {num_test} test samples")
    
    print(f"\nELPD Performance:")
    print(f"  Best α:    {best_alpha:.3f} → ELPD = {best_score:.4f} ± {std_scores[best_alpha_idx]:.4f}")
    print(f"  Nominal α: {nominal_alpha:.3f} → ELPD = {nominal_score:.4f} ± {std_scores[nominal_idx]:.4f}")
    print(f"  Improvement: {best_score - nominal_score:.4f}")
    
    print(f"\nTV Distance Performance:")
    print(f"  Best α:    {best_tv_alpha:.3f} → √n×TV = {best_tv_value:.4f}")
    print(f"  Nominal α: {nominal_alpha:.3f} → √n×TV = {nominal_tv_scaled:.4f}")
    print(f"  Reduction: {nominal_tv_scaled - best_tv_value:.4f}")
    
    print(f"\nCorrelation check:")
    print(f"  ELPD at α={best_alpha:.3f}: {mean_scores[best_alpha_idx]:.4f}")
    print(f"  TV at α={best_alpha:.3f}:   {mean_tv_scaled[best_alpha_idx]:.4f}")
    print(f"  ELPD at α={best_tv_alpha:.3f}: {mean_scores[best_tv_idx]:.4f}")
    print(f"  TV at α={best_tv_alpha:.3f}:   {mean_tv_scaled[best_tv_idx]:.4f}")
    
    print(f"\nSelected α values:")
    for alpha_val in [0.1, 0.2, 0.5, 1.0]:
        idx = jnp.argmin(jnp.abs(alpha_values - alpha_val))
        print(f"  α={alpha_values[idx]:.3f}: ELPD={mean_scores[idx]:.4f}, √n×TV={mean_tv_scaled[idx]:.4f}")
    
    print("=" * 80)
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ELPD Plot
    for seed in range(num_seeds):
        ax1.plot(alpha_values, results_elpd[:, seed], color='gray', alpha=0.15, linewidth=0.5)
    
    ax1.plot(alpha_values, quantile_05, 'k--', linewidth=1.5, alpha=0.7, label='5%/95% quantiles')
    ax1.plot(alpha_values, quantile_95, 'k--', linewidth=1.5, alpha=0.7)
    ax1.plot(alpha_values, mean_scores, 'k-', linewidth=3, label='Mean ELPD')
    ax1.axhline(nominal_score, color='gray', linestyle='-', linewidth=2, label=f'Nominal (α={nominal_alpha:.2f})')
    ax1.scatter(
        [best_alpha], [best_score], s=150, zorder=5,
        facecolors='none', edgecolors='black', linewidths=2,
        label=f'Best α={best_alpha:.3f}'
    )
    
    ax1.set_xlabel('Alpha (α)', fontsize=12)
    ax1.set_ylabel('ELPD', fontsize=12)
    ax1.set_title(f'ELPD vs Alpha (Student-t True Process, df={df_true})', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_ylim([-5, 0])
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # TV Distance Plot
    for seed in range(num_seeds):
        ax2.plot(alpha_values, results_tv_scaled[:, seed], color='gray', alpha=0.15, linewidth=0.5)
    
    ax2.plot(alpha_values, quantile_05_tv, 'k--', linewidth=1.5, alpha=0.7, label='5%/95% quantiles')
    ax2.plot(alpha_values, quantile_95_tv, 'k--', linewidth=1.5, alpha=0.7)
    ax2.plot(alpha_values, mean_tv_scaled, 'k-', linewidth=3, label='Mean √n×TV')
    ax2.axhline(nominal_tv_scaled, color='gray', linestyle='-', linewidth=2, label=f'Nominal (α={nominal_alpha:.2f})')
    ax2.scatter(
        [best_tv_alpha], [best_tv_value], s=150, zorder=5,
        facecolors='none', edgecolors='black', linewidths=2,
        label=f'Best α={best_tv_alpha:.3f}'
    )
    
    ax2.set_xlabel('Alpha (α)', fontsize=12)
    ax2.set_ylabel(f'√n × TV Distance (n={num_train})', fontsize=12)
    ax2.set_title(f'TV Distance vs Alpha (Student-t True Process, df={df_true})', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_ylim([-0.5, 3.0])
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()    
    plt.show()