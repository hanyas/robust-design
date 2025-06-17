"""
Location Finding Experiment

A generic implementation for locating K sources using sensor measurements.
The experiment estimates the positions of K sources based on noisy observations
from sensors placed at design locations.

Mathematical model:
    theta ~ N(mu, Sigma)
    x | theta, xi ~ N(\log f(theta, xi), sigma^2)
where:
    - theta: source locations (parameters)
    - xi: sensor (design) locations
    - f(theta, xi): signal model (sum of inverse power law signals from all sources plus baseline)
    - sigma^2: noise variance (may depend on design and theta)
"""
import jax
from jax import Array, random, numpy as jnp, scipy as jsc

from rdesign.base import Prior, Model
from rdesign.utils import info_gain

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Qt5Agg")


class UniformPrior(Prior):
    """Uniform prior over source locations in a bounded region."""

    def __init__(self, bounds: tuple, dim: int, num_sources: int = 1):
        """
        Initialize uniform prior.

        Args:
            bounds: Tuple of (min, max) values for each dimension
            dim: Spatial dimension (1, 2, or 3)
            num_sources: Number of sources (K)
        """
        self.dim = dim  # spatial dimension
        self.bounds = bounds  # (min, max)
        self.num_sources = num_sources

    def sample(self, key: Array, num_samples: int) -> Array:
        """
        Sample source locations.

        Returns:
            Array of shape (num_samples, K, D) for K sources in D dimensions
        """
        min_val, max_val = self.bounds

        return random.uniform(
            key,
            shape=(num_samples, self.num_sources, self.dim),
            minval=min_val,
            maxval=max_val
        )

    def logpdf(self, x: Array) -> Array:
        """
        Compute log probability density.

        Args:
            x: Source locations of shape (..., K, D)

        Returns:
            Log probability of shape (...,)
        """
        min_val, max_val = self.bounds

        # Calculate log probability using uniform distribution
        range_val = max_val - min_val
        log_probs = jsc.stats.uniform.logpdf(x, loc=min_val, scale=range_val)
        return jnp.sum(log_probs, axis=(-2, -1))  # (..., K, D) -> (...)


class GaussianPrior(Prior):
    """Gaussian prior over source locations."""

    def __init__(self, mean: Array, covar: Array, dim: int, num_sources: int = 1):
        """
        Initialize Gaussian prior.

        Args:
            mean: Mean location, shape (D,) for D-dimensional space
            covar: Covariance matrix, shape (D, D)
            num_sources: Number of sources (K)
        """
        self.dim = dim  # spatial dimension
        self.mean = mean
        self.covar = covar
        self.num_sources = num_sources

    def sample(self, key: Array, num_samples: int) -> Array:
        """Sample source locations from multivariate Gaussian."""

        shape = (num_samples, self.num_sources, self.dim)
        return random.multivariate_normal(
            key, self.mean, self.covar, shape=shape
        )

    def logpdf(self, x: Array) -> Array:
        """Compute log probability density for all sources."""

        # x has shape (num_samples, K, D)
        log_probs = jsc.stats.multivariate_normal.logpdf(x, self.mean, self.covar)
        return jnp.sum(log_probs, axis=(-2, -1))  # (..., K, D) -> (...)


class LocationModel(Model):
    """
    Location finding model with K sources in D-dimensional space.

    Sensor measures signal strength that decays with distance to sources.
    """

    def __init__(
        self,
        spatial_dim: int = 1,
        num_sources: int = 2,
        noise_std: float = 0.1,  # Reduced base noise
        decay_rate: float = 3.0,  # Increased decay rate
        baseline: float = 0.1    # Reduced baseline
    ):
        """
        Initialize location model.

        Args:
            spatial_dim: Spatial dimensionality (1, 2, or 3)
            num_sources: Number of sources (K)
            noise_std: Base standard deviation of measurement noise
            decay_rate: How fast signal decays with distance
            baseline: Minimum signal level
        """
        self.spatial_dim = spatial_dim
        self.num_sources = num_sources
        self.noise_std = noise_std
        self.decay_rate = decay_rate
        self.baseline = baseline

    def _compute_signal(self, design: Array, theta: Array) -> Array:
        """
        Compute expected signal at sensor locations.

        Args:
            design: Sensor locations, shape (N, D) for N designs in D dimensions
            theta: Source locations, shape (K, D) for K sources in D dimensions

        Returns:
            Expected signal, shape (N,) - one signal per design
        """
        # For each design point, compute distance to each source
        # Using broadcasting: (N, 1, D) - (1, K, D) -> (N, K, D)
        distances = jnp.linalg.norm(design[:, None, :] - theta[None, :, :], axis=-1)  # (N, K)

        # Signal from each source (inverse power law)
        signals = 1. / (distances ** self.decay_rate + 1e-3)  # (N, K)

        # Total signal is sum from all sources plus baseline
        total_signal = jnp.sum(signals, axis=-1) + self.baseline  # (N,)

        return total_signal

    def _compute_noise_std(self, design: Array, theta: Array) -> Array:
        """
        Compute noise standard deviation based on distance to sources.

        Args:
            design: Sensor locations, shape (N, D) for N designs in D dimensions
            theta: Source locations, shape (K, D) for K sources in D dimensions

        Returns:
            Noise standard deviation, shape (N,) - one std per design
        """
        # For each design point, compute distance to each source
        distances = jnp.linalg.norm(design[:, None, :] - theta[None, :, :], axis=-1)  # (N, K)

        # Noise increases with distance (linear scaling)
        distance_factor = jnp.min(distances ** self.decay_rate, axis=-1)  # (N,) min distance to any source
        return self.noise_std + 0.25 * distance_factor

    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        """
        Sample noisy measurements.

        Args:
            key: JAX PRNGKey
            design: Sensor locations, shape (N, D) for N designs in D dimensions
            theta: Source locations, shape (K, D) for K sources in D dimensions

        Returns:
            Noisy measurements, shape (N,) - one measurement per design
        """
        signal = self._compute_signal(design, theta)
        noise_std = self._compute_noise_std(design, theta)
        noise = random.normal(key, shape=signal.shape) * noise_std

        return jnp.log(signal) + noise

    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """
        Compute log likelihood of measurements.

        Args:
            x: Measurements, shape (N,) - one measurement per design
            design: Sensor locations, shape (N, D) for N designs in D dimensions
            theta: Source locations, shape (K, D) for K sources in D dimensions

        Returns:
            Log likelihood, shape (N,) - one likelihood per design
        """
        signal = self._compute_signal(design, theta)
        noise_std = self._compute_noise_std(design, theta)

        log_probs = jsc.stats.norm.logpdf(x, loc=jnp.log(signal), scale=noise_std)
        return jnp.sum(log_probs)


def create_location_experiment(
    spatial_dim: int = 1,
    num_sources: int = 2,
    bounds: tuple = (-3.0, 3.0),
    prior_type: str = "uniform",
    **kwargs
) -> tuple[Prior, Model]:
    """
    Create a location finding experiment.

    Args:
        spatial_dim: Spatial dimensionality (1, 2, or 3)
        num_sources: Number of sources to locate (K)
        bounds: Tuple of (min, max) values for each dimension
        prior_type: "uniform" or "gaussian"
        **kwargs: Additional arguments for prior/model

    Returns:
        Tuple of (prior, model) objects
    """
    # Create prior
    if prior_type == "uniform":
        prior = UniformPrior(bounds, spatial_dim, num_sources)
    elif prior_type == "gaussian":
        mean = kwargs.get("prior_mean", jnp.zeros(spatial_dim))
        cov = kwargs.get("prior_cov", jnp.eye(spatial_dim))
        prior = GaussianPrior(mean, cov, num_sources)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    # Create model
    model_kwargs = {k: v for k, v in kwargs.items() if k.startswith("model_")}
    model_kwargs = {k[6:]: v for k, v in model_kwargs.items()}  # Remove "model_" prefix

    model = LocationModel(spatial_dim=spatial_dim, num_sources=num_sources, **model_kwargs)

    return prior, model


# Example usage:
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    # # Create 1D model with 1 source
    # key = random.PRNGKey(0)
    # prior, model = create_location_experiment(
    #     spatial_dim=1,
    #     num_sources=1,
    #     bounds=(-3.0, 3.0)
    # )
    #
    # # Evaluate for both alpha values
    # alphas = jnp.array([0.05, 0.95])
    #
    # # Create design points to evaluate
    # n_grid = 1024
    # designs = jnp.linspace(-3.0, 3.0, n_grid)
    # info_gain_vals = jnp.zeros((len(alphas), n_grid))
    #
    # key = random.PRNGKey(0)
    # for i, alpha in enumerate(alphas):
    #     print(f"Evaluating alpha = {alpha:.2f}...")
    #     for j, design in enumerate(designs):
    #         key, sub_key = random.split(key)
    #         design = jnp.array([design]).reshape(1, 1)
    #         info_gain_vals = info_gain_vals.at[i, j].set(
    #             info_gain(
    #                 key=sub_key,
    #                 design=design,
    #                 prior=prior,
    #                 model=model,
    #                 num_outer=1024,
    #                 num_inner=1024,
    #                 alpha=float(alpha)
    #             )
    #         )
    #
    # # Plot results using subplots
    # fig, axes = plt.subplots(len(alphas) + 1, 1, figsize=(8, 10))
    # fig.suptitle('Information gain vs design location for different α values')
    #
    # for i, (ax, alpha) in enumerate(zip(axes, alphas)):
    #     ax.plot(designs, info_gain_vals[i], 'b-', linewidth=2)
    #     ax.set_ylabel(f'Information gain (alpha = {alpha:.2f})')
    #     ax.set_xlabel('Design location')
    #     ax.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.show()

    # Example with 2D grid of design pairs
    print("\nCreating surface plots for paired designs...")

    # Create 1D model with single source
    key = random.PRNGKey(0)
    prior, model = create_location_experiment(
        spatial_dim=1,
        num_sources=1,
        bounds=(-3.0, 3.0)
    )

    alphas = jnp.array([0.05, 0.95])

    # Create grid of design pairs
    nx = ny = 60  # Grid size
    x = jnp.linspace(-3.0, 3.0, nx)
    y = jnp.linspace(-3.0, 3.0, ny)
    X, Y = jnp.meshgrid(x, y)

    # Pre-allocate array for all surfaces
    info_gain_vals = jnp.zeros((len(alphas), nx, ny))
    key = random.PRNGKey(1)

    num_outer = int(1 * 1024)
    num_inner = int(1 * 1024)

    # Evaluate info gain for each alpha and each design pair
    for i, alpha in enumerate(alphas):
        print(f"Evaluating alpha = {alpha:.2f}...")
        for j in range(nx):
            for k in range(ny):
                key, sub_key = random.split(key)
                design = jnp.array([X[j, k], Y[j, k]]).reshape(2, 1)
                info_gain_vals = info_gain_vals.at[i, j, k].set(
                    info_gain(
                        key=sub_key,
                        design=design,
                        prior=prior,
                        model=model,
                        num_outer=num_outer,
                        num_inner=num_inner,
                        alpha=float(alpha)
                    )
                )

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.ravel()

    for i, (alpha, ax) in enumerate(zip(alphas, axes)):
        im = ax.imshow(
            info_gain_vals[i],
            origin='lower',
            extent=[-3.0, 3.0, -3.0, 3.0],
            aspect='equal',
            cmap='viridis',
        )
        ax.set_xlabel('Design 1')
        ax.set_ylabel('Design 2')
        ax.set_title(f'alpha = {alpha:.2f}')

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(15, 12))
    axes = []
    for i in range(len(alphas)):
        ax = fig.add_subplot(3, 2, i + 1, projection='3d')
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
