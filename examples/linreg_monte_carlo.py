"""
Linear Gaussian Regression Experiment

A generic implementation for Bayesian linear regression with a 2D parameter vector.
The experiment uses a Gaussian prior over the regression parameters and a Gaussian likelihood
with linear structure in the parameters. The model is:
    theta ~ N(mu, Sigma)
    x | theta, xi ~ N(theta_0 + theta_1 * xi, sigma^2)
where xi is the design, and x is the observed output.
"""
import jax
from jax import Array, random, numpy as jnp
from jax.scipy import stats

from rdesign.base import Prior, Likelihood
from rdesign.utils import robust_info_gain


class Gaussian(Prior):
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


class LinearRegression(Likelihood):
    """
    Linear regression model: x | theta, xi ~ N(theta_0 + theta_1 * xi, sigma^2)
    """
    def __init__(self, noise_std: Array):
        self.noise_std = noise_std

    def _compute_mean(self, design: Array, theta: Array):
        return theta[0] + theta[1] * design

    def _compute_stddev(self, design: Array, theta: Array):
        return self.noise_std # + 1e-2 * (theta[0] + theta[1] * design)
        # return self.noise_std

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
) -> tuple[Prior, Likelihood]:
    prior = Gaussian(prior_mean, prior_cov)
    lklhd = LinearRegression(noise_std=noise_std)
    return prior, lklhd


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
        [2.0, 0.75],
        [0.75, 1.0]
    ])

    key = random.PRNGKey(0)
    prior, lklhd = create_regression_experiment(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        noise_std=jnp.sqrt(1e-1)
    )

    # Define alphas and 2D design grid
    alphas = jnp.array([0.01, 0.99])

    n_grid = 40
    x = jnp.linspace(-1.0, 1.0, n_grid)
    y = jnp.linspace(-1.0, 1.0, n_grid)
    X, Y = jnp.meshgrid(x, y)

    info_gain_vals = jnp.zeros((len(alphas), n_grid, n_grid))

    num_outer = int(1 * 256)
    num_inner = int(1 * 256)

    for i, alpha in enumerate(alphas):
        print(f"Evaluating alpha = {alpha:.2f}...")
        for j in range(n_grid):
            for k in range(n_grid):
                key, sub_key = random.split(key)
                design = jnp.array([X[j, k], Y[j, k]])
                info_gain_vals = info_gain_vals.at[i, j, k].set(
                    robust_info_gain(
                        rng_key=sub_key,
                        design=design,
                        prior=prior,
                        lklhd=lklhd,
                        num_outer=num_outer,
                        num_inner=num_inner,
                        alpha=float(alpha)
                    )
                )

    # Plotting
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4))
    axes = axes.ravel()

    for i, (alpha, ax) in enumerate(zip(alphas, axes)):
        im = ax.imshow(
            info_gain_vals[i],
            origin='lower',
            extent=[-1, 1, -1, 1],
            aspect='auto',
            cmap='viridis'
        )
        ax.set_title(f'alpha = {alpha:.2f}')
        ax.set_xlabel('Design 1')
        ax.set_ylabel('Design 2')

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(15, 12))
    axes = []
    for i in range(len(alphas)):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        axes.append(ax)
    
    for i, (alpha, ax) in enumerate(zip(alphas, axes)):
        surf = ax.plot_surface(
            X, Y, info_gain_vals[i],
            cmap='viridis',
            edgecolor='none',
            alpha=0.8
        )
        ax.set_xlabel('Design 1')
        ax.set_ylabel('Design 2')
        ax.set_title(f'alpha = {alpha:.2f}')
    
    plt.tight_layout()
    plt.show()
