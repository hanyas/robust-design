"""
Linear Gaussian Regression Experiment - Variance Analysis

Modified version to evaluate the variance of robust_info_gain over multiple seeds
for a fixed design point, instead of evaluating over a 2D grid.
"""
import jax
from jax import Array, random, numpy as jnp
from jax.scipy import stats

from rdesign.base import Prior, Model
from rdesign.utils import robust_info_gain


class GaussianPrior(Prior):
    """Gaussian prior over 2D parameter vector theta."""
    def __init__(self, mean: Array, covar: Array):
        """
        Args:
            mean: Mean vector, shape (2,)
            covar: Covariance matrix, shape (2, 2)
        """
        self.mean = mean
        self.covar = covar

    def sample(self, key: Array, num_samples: int) -> Array:
        """Sample theta from multivariate Gaussian."""
        return random.multivariate_normal(key, self.mean, self.covar, shape=(num_samples,))


class LinearGaussianModel(Model):
    """
    Linear regression model: x | theta, xi ~ N(theta_0 + theta_1 * xi, sigma^2)
    """
    def __init__(self, noise_std: Array):
        self.noise_std = noise_std

    def _compute_mean(self, design: Array, theta: Array):
        return theta[0] + theta[1] * design

    def _compute_stddev(self, design: Array, theta: Array):
        return self.noise_std

    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        """
        Sample from the likelihood.
        Args:
            key: JAX PRNGKey
            design: shape (N, 2)
            theta: shape (2,)
        Returns:
            x: shape (N,)
        """
        mean = jax.vmap(self._compute_mean, in_axes=(0, None))(design, theta)
        stddev = jax.vmap(self._compute_stddev, in_axes=(0, None))(design, theta)
        return random.multivariate_normal(key, mean=mean, cov=jnp.diag(stddev**2))

    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """
        Log likelihood of x given design and theta.
        """
        mean = jax.vmap(self._compute_mean, in_axes=(0, None))(design, theta)
        stddev = jax.vmap(self._compute_stddev, in_axes=(0, None))(design, theta)
        return stats.multivariate_normal.logpdf(x, mean=mean, cov=jnp.diag(stddev**2))


def create_regression_experiment(
    prior_mean: Array = jnp.zeros(2),
    prior_cov: Array = jnp.eye(2),
    noise_std: Array = jnp.ones(1),
) -> tuple[Prior, Model]:
    prior = GaussianPrior(prior_mean, prior_cov)
    model = LinearGaussianModel(noise_std=noise_std)
    return prior, model


# Example usage
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_disable_jit", True)

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg")

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "axes.unicode_minus": False,
        # Font sizes
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        # Line and marker sizes
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        # Figure size for better visibility
        "figure.figsize": (12, 8)
    })

    prior_mean = jnp.array([0., 1.])
    prior_cov = jnp.array([
        [4.0, 0.75],
        [0.75, 2.0]
    ])

    key = random.PRNGKey(0)
    prior, model = create_regression_experiment(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        noise_std=jnp.sqrt(1.0)
    )

    # Define alphas and fixed design
    alphas = jnp.array([0.01, 0.1, 0.5, 0.9, 0.99])

    # Fixed design point
    fixed_design = jnp.array([-1.0, 1.0])

    num_seeds = 25
    num_outer = int(2 * 64)
    num_inner = int(10 * 64)

    # Store info gain values for each alpha and seed
    info_gain_vals = jnp.zeros((len(alphas), num_seeds))

    for i, alpha in enumerate(alphas):
        print(f"Evaluating alpha = {alpha:.2f}...")
        for seed_idx in range(num_seeds):
            key, sub_key = random.split(key)
            info_gain = robust_info_gain(
                key=sub_key,
                design=fixed_design,
                prior=prior,
                model=model,
                num_outer=num_outer,
                num_inner=num_inner,
                alpha=float(alpha)
            )
            info_gain_vals = info_gain_vals.at[i, seed_idx].set(info_gain)

        # Compute statistics
        mean_ig = jnp.mean(info_gain_vals[i])
        var_ig = jnp.var(info_gain_vals[i])
        std_ig = jnp.std(info_gain_vals[i])

        print(f"  Mean: {mean_ig:.6f}, Variance: {var_ig:.6f}, Std: {std_ig:.6f}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Variance vs Alpha
    variances = jnp.var(info_gain_vals, axis=1)
    axes[0].plot(alphas, variances, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel(r'$\alpha$')
    axes[0].set_ylabel('Variance of Robust Info Gain')
    axes[0].set_title('Variance vs Alpha')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Box plot for each alpha
    positions = jnp.arange(len(alphas))
    box_data = [info_gain_vals[i] for i in range(len(alphas))]
    bp = axes[1].boxplot(box_data, positions=positions, widths=0.6)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f'{alpha:.2f}' for alpha in alphas])
    axes[1].set_xlabel(r'$\alpha$')
    axes[1].set_ylabel('Robust Info Gain')
    axes[1].set_title(f'Distribution over {num_seeds} seeds (design = {fixed_design})')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('robust_info_gain_variance.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary table
    print("\n" + "="*60)
    print(f"Summary for fixed design = {fixed_design}")
    print("="*60)
    print(f"{'Alpha':<10} {'Mean':<15} {'Variance':<15} {'Std Dev':<15}")
    print("-"*60)
    for i, alpha in enumerate(alphas):
        mean_ig = jnp.mean(info_gain_vals[i])
        var_ig = jnp.var(info_gain_vals[i])
        std_ig = jnp.std(info_gain_vals[i])
        print(f"{alpha:<10.2f} {mean_ig:<15.6f} {var_ig:<15.6f} {std_ig:<15.6f}")
    print("="*60)