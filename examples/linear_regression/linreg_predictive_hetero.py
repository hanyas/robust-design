"""
Bayesian updating experiment for posterior predictive quality with misspecified models.

TRUE PROCESS: HeteroLinearGaussian (heteroscedastic noise)
MODELS: LinearGaussian (homoscedastic noise - MISSPECIFIED)

For each seed:
- Samples true parameter θ from prior
- Generates training data from HeteroLinearGaussian with alpha_true
- Fits LinearGaussian models (misspecified) with different alpha values
- Evaluates posterior predictive at test point
- Computes ELPD and TV distance

This reveals which homoscedastic model best approximates the heteroscedastic reality.
"""
from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr

from rdesign.distributions import Gaussian 
from rdesign.linear_regression import LinearGaussian, LinearRegression

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class HeteroLinearGaussian(NamedTuple):
    """
    Heteroscedastic Gaussian linear likelihood for regression.
    
    Args:
        base_noise_std: Base noise standard deviation
        hetero_scale: Scale factor for heteroscedasticity
    """
    base_noise_std: float
    hetero_scale: float = 0.01
    
    def design_matrix(self, design: Array) -> Array:
        design = jnp.atleast_1d(design)
        return jnp.column_stack([design, jnp.ones(len(design))])
    
    def mean(self, design: Array, theta: Array) -> Array:
        H = self.design_matrix(design)
        return H @ theta
    
    def stddev(self, design: Array, theta: Array) -> Array:
        mean_pred = self.mean(design, theta)
        return self.base_noise_std + self.hetero_scale * jnp.abs(mean_pred)
    
    def cov(self, design: Array, theta: Array, alpha: float = 1.0) -> Array:
        std = self.stddev(design, theta)
        return jnp.diag((std / jnp.sqrt(alpha)) ** 2)
    
    def sample(self, key: Array, design: Array, theta: Array, alpha: float = 1.0) -> Array:
        """Sample from heteroscedastic likelihood with alpha scaling."""
        mean = self.mean(design, theta)
        cov = self.cov(design, theta, alpha)
        return jr.multivariate_normal(key, mean, cov)


def compute_tv_distance_mc(
    rng_key: Array,
    mu_true: float,
    var_true: float,
    mu_model: float,
    var_model: float,
    num_samples: int = 1024,
) -> float:
    """Compute Total Variation distance TV(Gaussian || Gaussian) via Monte Carlo."""
    # Sample from both distributions
    rng_key, key1, key2 = jr.split(rng_key, 3)
    samples_true = mu_true + jnp.sqrt(var_true) * jr.normal(key1, (num_samples // 2,))
    samples_model = mu_model + jnp.sqrt(var_model) * jr.normal(key2, (num_samples // 2,))
    
    samples = jnp.concatenate([samples_true, samples_model])
    
    # Compute densities
    log_p_true = -0.5 * jnp.log(2 * jnp.pi * var_true) - 0.5 * (samples - mu_true)**2 / var_true    
    log_p_model = -0.5 * jnp.log(2 * jnp.pi * var_model) - 0.5 * (samples - mu_model)**2 / var_model

    p_true = jnp.exp(log_p_true)
    p_model = jnp.exp(log_p_model)

    # TV = (1/2) E[|p - q|]
    return 0.5 * jnp.mean(jnp.abs(p_true - p_model))


def run_experiment(
    rng_key: Array,
    num_train: int,
    num_test: int,
    model: LinearRegression,
    lklhd: HeteroLinearGaussian,
    alpha_true: float,
    alpha_values: Array,
    random_design: bool = False,
) -> tuple[Array, Array]:
    """
    Run single experiment with misspecified models.
    
    Steps:
    1. Sample true θ from prior
    2. Generate training data from HeteroLinearGaussian with alpha_true
    3. Fit LinearGaussian models (misspecified) for each alpha
    4. Evaluate posterior predictive at test point
    5. Compute ELPD and TV distance
    
    Args:
        rng_key: Random key
        num_train: Number of training points
        num_test: Number of test samples at single test point
        model: LinearRegression model
        lklhd: True process likelihood
        alpha_true: Alpha for true process
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

    # 3. Sample training data from TRUE heteroscedastic process
    rng_key, key = jr.split(rng_key)
    x_train = lklhd.sample(key, xi_train, theta_true, alpha=alpha_true)
    
    # 4. Generate test design
    rng_key, key = jr.split(rng_key)
    xi_test = jr.uniform(key, minval=-1.0, maxval=1.0)
    xi_test_array = jnp.array([xi_test])
    
    # 5. True distribution at test point (heteroscedastic)
    true_test_mean = lklhd.mean(xi_test_array, theta_true)[0]
    true_test_std = lklhd.stddev(xi_test_array, theta_true)[0]
    true_test_var = (true_test_std / jnp.sqrt(alpha_true)) ** 2
    
    # 6. Generate test samples from TRUE process (multiple samples at test point)
    rng_key, key = jr.split(rng_key)
    test_samples = true_test_mean + jnp.sqrt(true_test_var) * jr.normal(key, (num_test,))
    
    # 7. Evaluate each model (different alphas) - VECTORIZED
    def evaluate_alpha(alpha):
        # Fit misspecified homoscedastic model
        posterior = model.get_tilted_posterior(x_train, xi_train, alpha)
        predictive = model.get_posterior_predictive(posterior, xi_test_array)

        # Extract scalar mean and variance from predictive
        pred_mean = predictive.mean[0]
        pred_var = predictive.cov[0, 0]
        
        # ELPD: log p(test_samples | model)
        # Reshape test_samples to (num_test, 1) for multivariate logpdf
        score = jnp.mean(predictive.logpdf(test_samples[:, None]))
        
        # TV distance (compute deterministically, no random key needed)
        # For 1D Gaussians: TV ≈ |CDF_diff| which we approximate via sampling
        # Use a fixed computation instead
        tv_distance = compute_tv_distance_deterministic(true_test_mean, true_test_var, pred_mean, pred_var)

        return score, tv_distance
    
    # Vectorize over all alpha values at once
    scores, tv_distances = jax.vmap(evaluate_alpha)(alpha_values)
    
    return scores, tv_distances


def compute_tv_distance_deterministic(
    mu_true: float,
    var_true: float,
    mu_model: float,
    var_model: float,
    num_points: int = 1000,
) -> float:
    """Compute Total Variation distance via deterministic grid evaluation."""
    # Create a grid covering both distributions
    std_true = jnp.sqrt(var_true)
    std_model = jnp.sqrt(var_model)
    
    # Cover ±5 standard deviations of both distributions
    x_min = jnp.min(jnp.array([mu_true - 5*std_true, mu_model - 5*std_model]))
    x_max = jnp.max(jnp.array([mu_true + 5*std_true, mu_model + 5*std_model]))
    
    x = jnp.linspace(x_min, x_max, num_points)
    dx = (x_max - x_min) / num_points
    
    # Compute PDFs
    log_p_true = -0.5 * jnp.log(2 * jnp.pi * var_true) - 0.5 * (x - mu_true)**2 / var_true
    p_true = jnp.exp(log_p_true)
    
    log_p_model = -0.5 * jnp.log(2 * jnp.pi * var_model) - 0.5 * (x - mu_model)**2 / var_model
    p_model = jnp.exp(log_p_model)
    
    # TV = (1/2) ∫ |p - q| dx
    tv = 0.5 * jnp.sum(jnp.abs(p_true - p_model) * dx)
    
    return tv


if __name__ == "__main__":
    print("=" * 80)
    print("POSTERIOR PREDICTIVE EXPERIMENT: MISSPECIFIED MODELS")
    print("=" * 80)
    
    # Configuration
    alpha_true = 0.3
    base_noise_std = 0.1
    
    prior_mean = jnp.array([0., 1.])
    prior_cov = jnp.array([[2.0, 0.75], [0.75, 1.0]])

    prior = Gaussian(mean=prior_mean, cov=prior_cov)
    true_lklhd = HeteroLinearGaussian(base_noise_std=base_noise_std, hetero_scale=0.01)
    model_lklhd = LinearGaussian(noise_var=base_noise_std ** 2)
    model = LinearRegression(prior=prior, likelihood=model_lklhd)

    num_train = 2
    num_test = 1024
    num_seeds = 1024
    num_alphas = 64
    
    alpha_values = jnp.logspace(-2, 0, num_alphas)
    
    print(f"\nTRUE PROCESS: HeteroLinearGaussian (α={alpha_true})")
    print(f"  Noise: σ_base + 0.01|θ₀ξ + θ₁|")
    print(f"\nMODELS: LinearGaussian (MISSPECIFIED)")
    print(f"  Noise: constant σ = {base_noise_std}")
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
            alpha_true, alpha_values
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
    
    print(f"\nELPD Performance:")
    print(f"  Best α:    {best_alpha:.3f} → ELPD = {best_score:.4f} ± {std_scores[best_alpha_idx]:.4f}")
    print(f"  Nominal α: {nominal_alpha:.3f} → ELPD = {nominal_score:.4f} ± {std_scores[nominal_idx]:.4f}")
    print(f"  Improvement: {best_score - nominal_score:.4f}")
    
    print(f"\nTV Distance Performance:")
    print(f"  Best α:    {best_tv_alpha:.3f} → √n×TV = {best_tv_value:.4f}")
    print(f"  Nominal α: {nominal_alpha:.3f} → √n×TV = {nominal_tv_scaled:.4f}")
    print(f"  Reduction: {nominal_tv_scaled - best_tv_value:.4f}")
    
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
    ax1.set_title(f'ELPD vs Alpha (Misspecified Model, α_true={alpha_true})', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_ylim([-5.0, 2.0])
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
    ax2.set_title(f'TV Distance vs Alpha (Misspecified Model, α_true={alpha_true})', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_ylim([-0.5, 2.0])
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()    
    plt.show()