"""
Sequential Bayesian updating experiment for posterior predictive quality.

Samples a true parameter θ from the prior, then sequentially adds design points.
After each observation, computes posterior and evaluates predictive quality via
average KL divergence per test point from the TRUE data-generating process.

Measures: (1/N_test) * KL(p(x_test | θ_true, α_true) || p(x_test | D, α_model))

This reveals how quickly each model's posterior predictive converges to the true
process as training data accumulates. The average per test point makes results
interpretable regardless of the number of test locations.
"""

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

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
    """Sample parameters and observations from joint distribution."""
    d = joint_mean.shape[0]
    
    L = jnp.linalg.cholesky(joint_cov)
    z = random.normal(rng_key, (num_samples, d))
    samples = joint_mean + z @ L.T
    
    return samples[:, :dim_theta], samples[:, dim_theta:]


def get_posterior_distribution(
    x: Array,
    H: Array,
    prior_mean: Array,
    prior_cov: Array,
    noise_var: float,
    alpha: float,
) -> tuple[Array, Array]:
    """Compute posterior distribution given observations."""
    prec_prior = jnp.linalg.inv(prior_cov)
    prec_lik = (alpha / noise_var) * H.T @ H
    cov_post = jnp.linalg.inv(prec_prior + prec_lik)
    mean_post = cov_post @ (prec_prior @ prior_mean + (alpha / noise_var) * H.T @ x)
    return mean_post, cov_post


def get_posterior_predictive(
    H_test: Array,
    mu_post: Array,
    cov_post: Array,
    noise_var: float,
    alpha_post: float,
) -> tuple[Array, Array]:
    """Compute posterior predictive distribution p(x_test | D).
    
    Args:
        H_test: Test design matrix
        mu_post: Posterior mean over theta
        cov_post: Posterior covariance over theta
        noise_var: Observation noise variance
        alpha_post: Alpha used for predictive (should match posterior)
    
    Returns:
        Predictive mean and covariance
    """
    pred_mean = H_test @ mu_post
    pred_cov = H_test @ cov_post @ H_test.T + (noise_var / alpha_post) * jnp.eye(H_test.shape[0])
    return pred_mean, pred_cov


def compute_kl_gaussian(
    mu1: Array,
    cov1: Array,
    mu2: Array,
    cov2: Array,
) -> float:
    """Compute KL divergence KL(N(mu1, cov1) || N(mu2, cov2))."""
    d = mu1.shape[0]
    cov2_inv = jnp.linalg.inv(cov2)
    diff = mu2 - mu1
    
    kl = 0.5 * (
        jnp.trace(cov2_inv @ cov1)
        + diff @ cov2_inv @ diff
        - d
        + jnp.linalg.slogdet(cov2)[1] - jnp.linalg.slogdet(cov1)[1]
    )
    return kl


def run_sequential_experiment(
    rng_key: Array,
    max_train: int,
    num_test: int,
    prior_mean: Array,
    prior_cov: Array,
    noise_var: float,
    alpha_nominal: float,
    alpha_worst: float,
    alpha_true: float,
) -> dict[str, list[float]]:
    """Run sequential Bayesian updating experiment.
    
    Samples a true parameter θ from the prior, then sequentially:
    - Generates design points
    - Observes data from that fixed θ
    - Updates posteriors for all models
    - Computes KL between oracle and model predictives
    
    Args:
        rng_key: Random key
        max_train: Maximum number of training points (will run from 1 to max_train)
        num_test: Number of test points
        prior_mean: Initial prior mean
        prior_cov: Initial prior covariance
        noise_var: Observation noise variance
        alpha_nominal: Alpha for nominal model
        alpha_worst: Alpha for worst-case model
        alpha_true: Alpha for true model
    
    Returns:
        Dictionary with KL divergences at each step for each model
    """
    # Sample true parameter from prior (fixed for entire experiment)
    rng_key, sub_key = random.split(rng_key)
    theta_true = random.multivariate_normal(sub_key, prior_mean, prior_cov)
    
    # Generate test design (fixed for all iterations)
    rng_key, sub_key = random.split(rng_key)
    xi_test = random.uniform(sub_key, shape=(num_test,), minval=-1.0, maxval=1.0)
    H_test = get_design_matrix(xi_test)
    
    # True predictive distribution at test points (this is FIXED - the ground truth)
    true_pred_mean = H_test @ theta_true
    true_pred_cov = (noise_var / alpha_true) * jnp.eye(num_test)
    
    # Storage for results at each step
    kl_nominal_history = []
    kl_worst_history = []
    kl_true_history = []
    
    # Initialize priors for each model
    mu_prior_nominal = prior_mean.copy()
    cov_prior_nominal = prior_cov.copy()
    
    mu_prior_worst = prior_mean.copy()
    cov_prior_worst = prior_cov.copy()
    
    mu_prior_true = prior_mean.copy()
    cov_prior_true = prior_cov.copy()
    
    # Sequential updating
    for i in range(max_train):
        # Generate new design point
        rng_key, sub_key = random.split(rng_key)
        xi_i = random.uniform(sub_key, shape=(1,), minval=-1.0, maxval=1.0)
        H_i = get_design_matrix(xi_i)
        
        # Sample observation from TRUE parameter with TRUE noise level
        mean_x = H_i @ theta_true
        rng_key, sub_key = random.split(rng_key)
        x_i = mean_x + jnp.sqrt(noise_var / alpha_true) * random.normal(sub_key, mean_x.shape)
        
        # Update posteriors for each model
        mu_post_nominal, cov_post_nominal = get_posterior_distribution(
            x_i, H_i, mu_prior_nominal, cov_prior_nominal, noise_var, alpha_nominal
        )
        mu_post_worst, cov_post_worst = get_posterior_distribution(
            x_i, H_i, mu_prior_worst, cov_prior_worst, noise_var, alpha_worst
        )
        mu_post_true, cov_post_true = get_posterior_distribution(
            x_i, H_i, mu_prior_true, cov_prior_true, noise_var, alpha_true
        )
        
        # Compute posterior predictives at test points
        pred_mean_nominal, pred_cov_nominal = get_posterior_predictive(
            H_test, mu_post_nominal, cov_post_nominal, noise_var, alpha_nominal
        )
        pred_mean_worst, pred_cov_worst = get_posterior_predictive(
            H_test, mu_post_worst, cov_post_worst, noise_var, alpha_worst
        )
        pred_mean_true_model, pred_cov_true_model = get_posterior_predictive(
            H_test, mu_post_true, cov_post_true, noise_var, alpha_true
        )
        
        # Compute KL divergences from TRUE PROCESS (not posterior predictive!)
        # Average over test points for numerical stability and interpretability
        kl_nominal = compute_kl_gaussian(true_pred_mean, true_pred_cov, pred_mean_nominal, pred_cov_nominal) / num_test
        kl_worst = compute_kl_gaussian(true_pred_mean, true_pred_cov, pred_mean_worst, pred_cov_worst) / num_test
        kl_true = compute_kl_gaussian(true_pred_mean, true_pred_cov, pred_mean_true_model, pred_cov_true_model) / num_test
        
        kl_nominal_history.append(float(kl_nominal))
        kl_worst_history.append(float(kl_worst))
        kl_true_history.append(float(kl_true))
        
        # Update priors for next iteration (posterior becomes prior)
        mu_prior_nominal = mu_post_nominal
        cov_prior_nominal = cov_post_nominal
        
        mu_prior_worst = mu_post_worst
        cov_prior_worst = cov_post_worst
        
        mu_prior_true = mu_post_true
        cov_prior_true = cov_post_true
    
    return {
        "nominal": kl_nominal_history,
        "worst": kl_worst_history,
        "true": kl_true_history,
    }


if __name__ == "__main__":
    # Configuration
    alpha_worst = 0.3
    alpha_true = 0.5
    alpha_nominal = 1.0
    noise_var = 0.1
    
    # Prior
    prior_mean = jnp.zeros(2)
    prior_cov = jnp.eye(2)
    
    # Experiment parameters
    max_train = 10  # Maximum number of training points
    num_test = 10  # Fixed number of test points
    num_seeds = 50  # Number of random seeds per experiment
    
    # Storage for results
    results = {
        "nominal": jnp.zeros((max_train, num_seeds)),
        "worst": jnp.zeros((max_train, num_seeds)),
        "true": jnp.zeros((max_train, num_seeds)),
    }
    
    print("Running sequential experiments...")
    print(f"Max training points: {max_train}")
    print(f"Number of seeds:     {num_seeds}")
    print(f"Test points:         {num_test}")
    print(f"Mode:                Sequential Bayesian updating")
    
    rng_key = random.PRNGKey(0)
    
    for seed in range(num_seeds):
        print(f"\rSeed {seed+1}/{num_seeds}...", end=" ", flush=True)
        
        rng_key, sub_key = random.split(rng_key)
        result = run_sequential_experiment(
            sub_key,
            max_train,
            num_test,
            prior_mean,
            prior_cov,
            noise_var,
            alpha_nominal,
            alpha_worst,
            alpha_true,
        )
        
        # Store results for all training steps
        for i in range(max_train):
            results["nominal"] = results["nominal"].at[i, seed].set(result["nominal"][i])
            results["worst"] = results["worst"].at[i, seed].set(result["worst"][i])
            results["true"] = results["true"].at[i, seed].set(result["true"][i])
    
    print("✓\n\nGenerating plots...")
    
    num_train_range = jnp.arange(1, max_train + 1)
    
    # Compute statistics
    mean_kl_nominal = jnp.mean(results["nominal"], axis=1)
    std_kl_nominal = jnp.std(results["nominal"], axis=1)
    
    mean_kl_worst = jnp.mean(results["worst"], axis=1)
    std_kl_worst = jnp.std(results["worst"], axis=1)
    
    mean_kl_true = jnp.mean(results["true"], axis=1)
    std_kl_true = jnp.std(results["true"], axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Mean KL with error bars
    ax.errorbar(num_train_range, mean_kl_nominal, yerr=std_kl_nominal, 
                marker='o', capsize=5, capthick=2, linewidth=2.5, markersize=8,
                color='steelblue', label='Nominal')
    ax.errorbar(num_train_range, mean_kl_worst, yerr=std_kl_worst,
                marker='s', capsize=5, capthick=2, linewidth=2.5, markersize=8,
                color='seagreen', label='Worst-case')
    ax.errorbar(num_train_range, mean_kl_true, yerr=std_kl_true,
                marker='^', capsize=5, capthick=2, linewidth=2.5, markersize=8,
                color='orange', label='True')
    
    ax.set_xlabel('Number of Training Design Points', fontsize=12)
    ax.set_ylabel('Average KL per Test Point', fontsize=12)
    ax.set_title('Predictive Quality vs Training Data', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(num_train_range)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: SEQUENTIAL PREDICTIVE QUALITY")
    print("=" * 80)
    print(f"\nAfter 1 observation:")
    print(f"  Nominal:     {mean_kl_nominal[0]:.4f} ± {std_kl_nominal[0]:.4f}")
    print(f"  Worst-case:  {mean_kl_worst[0]:.4f} ± {std_kl_worst[0]:.4f}")
    print(f"  True:        {mean_kl_true[0]:.4f} ± {std_kl_true[0]:.4f}")
    print(f"\nAfter {max_train} observations (sequential updating):")
    print(f"  Nominal:     {mean_kl_nominal[-1]:.4f} ± {std_kl_nominal[-1]:.4f}")
    print(f"  Worst-case:  {mean_kl_worst[-1]:.4f} ± {std_kl_worst[-1]:.4f}")
    print(f"  True:        {mean_kl_true[-1]:.4f} ± {std_kl_true[-1]:.4f}")
    print(f"\nReduction from start to end:")
    print(f"  Nominal:     {mean_kl_nominal[0] - mean_kl_nominal[-1]:.4f} ({100*(mean_kl_nominal[0] - mean_kl_nominal[-1])/mean_kl_nominal[0]:.1f}%)")
    print(f"  Worst-case:  {mean_kl_worst[0] - mean_kl_worst[-1]:.4f} ({100*(mean_kl_worst[0] - mean_kl_worst[-1])/mean_kl_worst[0]:.1f}%)")
    print("\n" + "=" * 80)
