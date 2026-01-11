"""
Bayesian updating experiment for posterior predictive quality with Student-t true process.

TRUE PROCESS: Student-t (heavy-tailed noise)
MODELS: ABTesting with Gaussian noise (MISSPECIFIED)

For each seed:
- Samples true parameter θ = [θ₀, θ₁] from prior
- Generates training data from Student-t process
- Fits ABTesting models (misspecified) with different alpha values
- Evaluates posterior predictive at test design
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
from rdesign.ab_testing import ABTesting, MaskedLinearGaussian

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class MaskedStudentT(NamedTuple):
    """
    Student-t likelihood for A/B testing with independent observations (heavy-tailed noise).
    
    Each participant in group g has observations from Student-t(θ_g, σ_g², df).
    
    Args:
        num_participants: Total number of participants n
        observation_std_devs: Standard deviations [σ₀, σ₁] for each group
        df: Degrees of freedom (controls tail heaviness)
    """
    num_participants: int
    observation_std_devs: Array
    df: float
    
    def _group_mask(self, design: Array) -> Array:
        """
        Create boolean mask indicating group assignment.
        
        Args:
            design: Number of participants in group 0 (scalar or array)
            
        Returns:
            Boolean array of shape (num_participants,) where True = group 0
        """
        design = jnp.asarray(design, dtype=jnp.int32)
        design = jnp.clip(design, 0, self.num_participants)
        idx = jnp.arange(self.num_participants)
        return idx < design
    
    def mean(self, design: Array, theta: Array) -> Array:
        """Location parameter (mean for df > 1)."""
        mask = self._group_mask(design)
        return jnp.where(mask, theta[0], theta[1])
    
    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        """
        Sample from Student-t likelihood.
        
        Args:
            key: Random key
            design: Number of participants in group 0
            theta: Parameters [θ₀, θ₁]
            
        Returns:
            Observations x of shape (num_participants,)
        """
        loc = self.mean(design, theta)
        
        # Sample from chi-squared for each participant
        key, key1, key2 = jr.split(key, 3)
        chi2_samples = jr.gamma(key1, self.df/2, shape=(self.num_participants,)) * 2
        
        # Sample from normal for each participant
        mask = self._group_mask(design)
        std = jnp.where(mask, self.observation_std_devs[0], self.observation_std_devs[1])
        normal_samples = jr.normal(key2, (self.num_participants,)) * std
        
        # Scale to get Student-t
        scale_factors = jnp.sqrt(self.df / chi2_samples)
        t_samples = loc + normal_samples * scale_factors
        
        return t_samples
    
    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """
        Compute log p(x | θ, ξ) for Student-t.
        
        Since observations are independent, this is the sum of individual log pdfs.
        
        Args:
            x: Observations, shape (num_participants,)
            design: Number of participants in group 0
            theta: Parameters [θ₀, θ₁]
            
        Returns:
            Log likelihood scalar
        """
        loc = self.mean(design, theta)
        mask = self._group_mask(design)
        scale = jnp.where(
            mask, 
            self.observation_std_devs[0], 
            self.observation_std_devs[1]
        )
        
        # Log density for each observation (sum over independent components)
        diff = x - loc
        mahal = (diff / scale) ** 2
        
        log_norm = (
            special.gammaln((self.df + 1) / 2)
            - special.gammaln(self.df / 2)
            - 0.5 * jnp.log(self.df * jnp.pi)
            - jnp.log(scale)
        )
        
        log_density = log_norm - ((self.df + 1) / 2) * jnp.log(1 + mahal / self.df)
        
        return jnp.sum(log_density)


# ============================================================================
# Utility Functions
# ============================================================================

def compute_tv_distance_deterministic(
    mu_true: float,
    var_true: float,
    df_true: float,
    mu_model: float,
    var_model: float,
    num_points: int = 1024,
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


def _compute_tilted_posterior_params(
    x_train: Array,
    design_train: int,
    alpha: float,
    prior_mean: Array,
    prior_cov: Array,
    obs_stds: Array,
    num_participants: int,
) -> tuple[Array, Array]:
    """JIT-compiled computation of tilted posterior parameters."""
    design = jnp.clip(design_train, 0, num_participants)
    idx = jnp.arange(num_participants)
    mask = idx < design
    n_group0 = design
    n_group1 = num_participants - design
    
    prior_vars = jnp.diag(prior_cov)
    obs_vars = obs_stds ** 2
    
    # Tilted posterior variances
    posterior_var0 = 1.0 / (1.0 / prior_vars[0] + alpha * n_group0 / obs_vars[0])
    posterior_var1 = 1.0 / (1.0 / prior_vars[1] + alpha * n_group1 / obs_vars[1])
    
    # Sum of observations in each group
    sum_group0 = jnp.sum(x_train * mask)
    sum_group1 = jnp.sum(x_train * (~mask))
    
    # Tilted posterior means
    posterior_mean0 = posterior_var0 * (prior_mean[0] / prior_vars[0] + alpha * sum_group0 / obs_vars[0])
    posterior_mean1 = posterior_var1 * (prior_mean[1] / prior_vars[1] + alpha * sum_group1 / obs_vars[1])
    
    posterior_mean = jnp.array([posterior_mean0, posterior_mean1])
    posterior_vars_vec = jnp.array([posterior_var0, posterior_var1])
    
    return posterior_mean, posterior_vars_vec


def _compute_posterior_predictive_params(
    posterior_mean: Array,
    posterior_vars: Array,
    design_test: int,
    obs_vars: Array,
    num_participants: int,
) -> tuple[Array, Array]:
    """Computation of posterior predictive parameters."""
    test_mask = jnp.arange(num_participants) < design_test
    pred_means = jnp.where(test_mask, posterior_mean[0], posterior_mean[1])
    pred_vars = jnp.where(
        test_mask,
        obs_vars[0] + posterior_vars[0],
        obs_vars[1] + posterior_vars[1]
    )
    return pred_means, pred_vars


@jax.jit
def _compute_elpd(test_samples: Array, pred_means: Array, pred_vars: Array) -> float:
    """ELPD computation."""
    num_participants = len(pred_means)
    
    def compute_logpdf(test_sample):
        diff = test_sample - pred_means
        log_norm = -0.5 * num_participants * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(jnp.log(pred_vars))
        mahal = jnp.sum(diff**2 / pred_vars)
        return log_norm - 0.5 * mahal
    
    return jnp.mean(jax.vmap(compute_logpdf)(test_samples))


@jax.jit
def _compute_mean_tv(
    test_means: Array,
    test_vars: Array,
    pred_means: Array,
    pred_vars: Array,
    df: float,
) -> float:
    """TV distance computation."""
    def compute_tv_single(true_mean, true_var, pred_mean, pred_var):
        return compute_tv_distance_deterministic(
            true_mean, true_var, df,
            pred_mean, pred_var
        )
    
    tv_distances = jax.vmap(compute_tv_single)(
        test_means, test_vars, pred_means, pred_vars
    )
    return jnp.mean(tv_distances)


def run_experiment(
    rng_key: Array,
    num_participants: int,
    num_test: int,
    model: ABTesting,
    lklhd: MaskedStudentT,
    alpha_values: Array,
    random_design: bool = False,
) -> tuple[Array, Array]:
    """
    Run single experiment with Student-t true process.
    
    Steps:
    1. Sample true θ = [θ₀, θ₁] from prior
    2. Generate training data from Student-t process
    3. Fit ABTesting models (misspecified) for each alpha
    4. Evaluate posterior predictive at test design
    5. Compute ELPD and TV distance
    
    Args:
        rng_key: Random key
        num_participants: Number of participants in experiment
        num_test: Number of test samples at test design
        model: ABTesting model (with Gaussian likelihood)
        lklhd: Student-t likelihood (true process)
        alpha_values: Alpha values to evaluate for models
        random_design: If True, use random training design; else use balanced (n/2, n/2)
    
    Returns:
        ELPD scores and TV distances for each alpha
    """
    # 1. Sample true parameter
    rng_key, key = jr.split(rng_key)
    theta_true = model.prior.sample(key, 1)[0]
    
    # 2. Generate training design (number of participants in group 0)
    rng_key, key = jr.split(rng_key)
    if random_design:
        design_train = int(jr.randint(key, shape=(), minval=0, maxval=num_participants + 1))
    else:
        design_train = num_participants // 2
    
    # 3. Sample training data from TRUE Student-t process
    rng_key, key = jr.split(rng_key)
    x_train = lklhd.sample(key, design_train, theta_true)
    
    # 4. Generate test design
    rng_key, key = jr.split(rng_key)
    design_test = int(jr.randint(key, shape=(), minval=0, maxval=num_participants + 1))
    
    # 5. True distribution at test design (Student-t per participant)
    # For ELPD, we need to evaluate on individual test observations
    # Generate test samples from TRUE Student-t process
    rng_key, key = jr.split(rng_key)
    test_samples = jax.vmap(lambda k: lklhd.sample(k, design_test, theta_true))(
        jr.split(key, num_test)
    )  # Shape: (num_test, num_participants)
    
    # 6. Get test location parameters for TV distance computation
    test_means = lklhd.mean(design_test, theta_true)
    test_mask = lklhd._group_mask(design_test)
    test_vars = jnp.where(
        test_mask,
        lklhd.observation_std_devs[0] ** 2,
        lklhd.observation_std_devs[1] ** 2
    )
    
    # 7. Evaluate each model (different alphas) - VECTORIZED with JIT helpers
    obs_vars = model.likelihood.observation_std_devs ** 2
    
    # Create JIT-compiled functions with num_participants fixed
    compute_tilted_posterior = jax.jit(
        lambda x, d, a, pm, pc, obs: _compute_tilted_posterior_params(x, d, a, pm, pc, obs, num_participants)
    )
    compute_predictive = jax.jit(
        lambda pm, pv, d, ov: _compute_posterior_predictive_params(pm, pv, d, ov, num_participants)
    )
    
    def evaluate_alpha(alpha):
        # Compute tilted posterior (JIT-compiled)
        posterior_mean, posterior_vars = compute_tilted_posterior(
            x_train, design_train, alpha,
            model.prior.mean, model.prior.cov,
            model.likelihood.observation_std_devs
        )
        
        # Compute posterior predictive (JIT-compiled)
        pred_means, pred_vars = compute_predictive(
            posterior_mean, posterior_vars, design_test, obs_vars
        )
        
        # ELPD (JIT-compiled)
        score = _compute_elpd(test_samples, pred_means, pred_vars)
        
        # TV distance (JIT-compiled)
        tv_distance = _compute_mean_tv(
            test_means, test_vars, pred_means, pred_vars, lklhd.df
        )
        
        return score, tv_distance
    
    # Vectorize over all alpha values at once
    scores, tv_distances = jax.vmap(evaluate_alpha)(alpha_values)
    
    return scores, tv_distances


if __name__ == "__main__":
    print("=" * 80)
    print("POSTERIOR PREDICTIVE EXPERIMENT: STUDENT-T TRUE PROCESS (AB TESTING)")
    print("=" * 80)
    
    # Configuration
    df_true = 2.5
    num_participants = 20
    observation_std_devs = jnp.array([0.3, 0.3])
    
    prior_means = jnp.array([0.0, 0.0])
    prior_cov = jnp.array([[10.0, 0.0], [0.0, 1.0]])

    # Prior
    prior = Gaussian(mean=prior_means, cov=prior_cov)
    
    # TRUE process: Student-t (heavy-tailed)
    true_lklhd = MaskedStudentT(
        num_participants=num_participants,
        observation_std_devs=observation_std_devs,
        df=df_true
    )
    
    # MODEL: Gaussian (MISSPECIFIED)
    model_lklhd = MaskedLinearGaussian(
        num_participants=num_participants,
        observation_std_devs=observation_std_devs
    )
    model = ABTesting(prior=prior, likelihood=model_lklhd)
    
    # Experiment parameters
    num_test = 1024
    num_seeds = 1024
    num_alphas = 64
    
    alpha_values = jnp.logspace(-2, 0, num_alphas)
    
    print(f"\nTRUE PROCESS: Student-t (df={df_true}, heavy-tailed)")
    print(f"  Participants: {num_participants}")
    print(f"  Noise: σ₀ = {observation_std_devs[0]}, σ₁ = {observation_std_devs[1]}")
    print(f"\nMODELS: ABTesting with Gaussian noise (MISSPECIFIED)")
    print(f"  Noise: σ₀ = {observation_std_devs[0]}, σ₁ = {observation_std_devs[1]}")
    print(f"\nSetup:")
    print(f"  Participants:    {num_participants}")
    print(f"  Test samples:    {num_test}")
    print(f"  Random seeds:    {num_seeds}")
    print(f"  Alpha values:    {num_alphas}")
    
    # Storage
    results_elpd = jnp.zeros((num_alphas, num_seeds))
    results_tv = jnp.zeros((num_alphas, num_seeds))
    
    rng_key = jr.PRNGKey(42)
    
    print(f"\nRunning experiments...")
    
    for seed in range(num_seeds):
        if (seed + 1) % 100 == 0 or seed == 0:
            print(f"  Seed {seed+1}/{num_seeds}...")
        
        rng_key, key = jr.split(rng_key)
        scores, tv_dists = run_experiment(
            key, num_participants, num_test, model, true_lklhd, alpha_values
        )
        
        results_elpd = results_elpd.at[:, seed].set(scores)
        results_tv = results_tv.at[:, seed].set(tv_dists)
    
    print("✓ Complete\n")
    
    # ========================================================================
    # Analysis
    # ========================================================================
    
    # Scale TV by sqrt(n)
    sqrt_n = jnp.sqrt(num_participants)
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
    print(f"Setup: {num_participants} participants, {num_test} test samples")
    
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
    for seed in range(min(100, num_seeds)):  # Limit to 100 lines for visibility
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
    ax2.set_ylim([-10.0, -70.0])
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # TV Distance Plot
    for seed in range(min(100, num_seeds)):  # Limit to 100 lines for visibility
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
    ax2.set_ylabel(f'√n × TV Distance (n={num_participants})', fontsize=12)
    ax2.set_title(f'TV Distance vs Alpha (Student-t True Process, df={df_true})', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_ylim([0.0, 2.5])
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()    
    plt.show()
