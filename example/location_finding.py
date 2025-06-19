"""
Location Finding Experiment

A generic implementation for locating K sources using sensor measurements.
The experiment estimates the positions of K sources based on noisy observations
from sensors placed at design locations.

Mathematical model:
    theta ~ N(mu, Sigma)
    x | theta, xi ~ N(log f(theta, xi), sigma^2)
where:
    - theta: source locations (parameters)
    - xi: sensor (design) locations
    - f(theta, xi): signal model (sum of inverse power law signals from all sources plus baseline)
    - sigma^2: noise variance (may depend on design and theta)
"""
import jax
from jax import Array, random, numpy as jnp
from jax.scipy import stats, special

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
            Array of shape (N, K, D) for K sources in D dimensions
        """
        min_val, max_val = self.bounds

        return random.uniform(
            key,
            shape=(num_samples, self.num_sources, self.dim),
            minval=min_val,
            maxval=max_val
        )


class GaussianPrior(Prior):
    """Gaussian prior over source locations."""

    def __init__(self, mean: Array, covar: Array, dim: int, num_sources: int = 1):
        """
        Initialize Gaussian prior.

        Args:
            mean: Mean location, shape (K, D) for K sources in D-dimensional space
            covar: Covariance matrix, shape (K, D, D)
            num_sources: Number of sources (K)
        """
        self.dim = dim  # spatial dimension
        self.mean = mean
        self.covar = covar
        self.num_sources = num_sources

    def sample(self, key: Array, num_samples: int) -> Array:
        """
        Sample source locations.

        Returns:
            Array of shape (N, K, D) for K sources in D dimensions
        """
        samples = jnp.zeros((num_samples, self.num_sources, self.dim))
        for k in range(self.num_sources):
            key, sub_key = random.split(key, 2)
            samples = samples.at[:, k, :].set(
                random.multivariate_normal(
                    sub_key,
                    self.mean[k],
                    self.covar[k],
                    shape=(num_samples,)
                )
            )
        return samples


class MixtureGaussianPrior(Prior):
    """Mixture of Gaussian priors over source locations."""

    def __init__(self, means: Array, covars: Array, weights: Array, dim: int, num_sources: int = 1):
        """
        Initialize mixture of Gaussian priors.

        Args:
            means: Mean locations, shape (K, M, D) for M mixture components in D-dimensional space
            covars: Covariance matrices, shape (K, M, D, D)
            weights: Mixture weights, shape (K, M) that sum to 1
            dim: Spatial dimension
            num_sources: Number of sources (K)
        """
        self.dim = dim
        self.means = means
        self.covars = covars
        self.weights = weights
        self.num_sources = num_sources

    def sample(self, key: Array, num_samples: int) -> Array:
        """
        Sample source locations from mixture of Gaussians.

        Returns:
            Array of shape (N, K, D) for K sources in D dimensions
        """
        # Split keys for component selection and sampling
        categorical_key, gaussian_key = random.split(key, 2)

        # Sample mixture components for all sources and samples at once
        # We need to handle each source separately due to broadcasting constraints
        component_idx = jnp.zeros((self.num_sources, num_samples), dtype=jnp.int32)

        # Sample components for each source
        for k in range(self.num_sources):
            key_k = random.fold_in(categorical_key, k)
            component_idx = component_idx.at[k, :].set(
                random.categorical(
                    key_k,
                    logits=jnp.log(self.weights[k]),  # Shape: (M,)
                    shape=(num_samples,)
                )
            )

        # Use advanced indexing to get selected means and covariances
        # Create index arrays for advanced indexing
        source_idx = jnp.arange(self.num_sources)[:, None]  # Shape: (K, 1)

        # Get selected means and covariances using advanced indexing
        # Shape: (K, N, D) and (K, N, D, D)
        selected_means = self.means[source_idx, component_idx, :]
        selected_covars = self.covars[source_idx, component_idx, :, :]

        # Reshape for vectorized sampling
        # Combine K and N dimensions: (K, N, D) -> (K*N, D)
        flat_means = selected_means.reshape(-1, self.dim)
        flat_covars = selected_covars.reshape(-1, self.dim, self.dim)

        # Sample from all distributions at once
        flat_samples = random.multivariate_normal(
            gaussian_key,
            mean=flat_means,
            cov=flat_covars
        )  # Shape: (K*N, D)

        # Reshape back to desired output shape
        samples = flat_samples.reshape(self.num_sources, num_samples, self.dim)

        # Transpose to get (N, K, D)
        return samples.transpose(1, 0, 2)


# def sample(self, key: Array, num_samples: int) -> Array:
#     # Split key for each source
#     keys = random.split(key, self.num_sources + 1)
#
#     # Sample mixture components for all sources at once
#     component_keys = random.split(keys[0], self.num_sources)
#     component_idx = jnp.array([
#         random.categorical(
#             component_keys[k],
#             logits=jnp.log(self.weights[k]),
#             shape=(num_samples,)
#         ) for k in range(self.num_sources)
#     ])  # Shape: (K, N)
#
#     # Sample from selected components
#     samples = jnp.zeros((num_samples, self.num_sources, self.dim))
#     for k in range(self.num_sources):
#         # Get the selected components for this source
#         selected_means = self.means[k][component_idx[k]]  # Shape: (N, D)
#         selected_covs = self.covars[k][component_idx[k]]  # Shape: (N, D, D)
#
#         # Sample from multivariate normal
#         samples = samples.at[:, k, :].set(
#             random.multivariate_normal(
#                 keys[k + 1],
#                 mean=selected_means,
#                 cov=selected_covs
#             )
#         )
#     return samples


class LocationModel(Model):
    """
    Location finding model with K sources in D-dimensional space.

    Sensor measures signal strength that decays with distance to sources.
    """

    def __init__(
        self,
        spatial_dim: int = 1,
        num_sources: int = 2,
        noise_std: Array = 0.25,
        decay_rate: Array = 2.0,
        baseline: Array = 0.1
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

    def _compute_mean(self, design: Array, theta: Array) -> Array:
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
        signals = 1. / (distances ** self.decay_rate + 1e-3)  # (N, K)
        total_signal = jnp.sum(signals, axis=-1) + self.baseline  # (N,)
        return total_signal

    def _compute_stddev(self, design: Array, theta: Array) -> Array:
        """
        Compute noise standard deviation based on distance to sources.

        Args:
            design: Sensor locations, shape (N, D) for N designs in D dimensions
            theta: Source locations, shape (K, D) for K sources in D dimensions

        Returns:
            Noise standard deviation, shape (N,) - one std per design
        """
        # For each design point, compute distance to each source
        return self.noise_std

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
        mean = self._compute_mean(design, theta)
        stddev = self._compute_stddev(design, theta)
        return jnp.log(mean) + random.normal(key, shape=mean.shape) * stddev

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
        mean = self._compute_mean(design, theta)
        stddev = self._compute_stddev(design, theta)
        log_probs = stats.norm.logpdf(x, loc=jnp.log(mean), scale=stddev)
        return jnp.sum(log_probs)


def create_location_experiment(
    spatial_dim: int = 1,
    num_sources: int = 2,
    prior_type: str = "uniform",
    **kwargs
) -> tuple[Prior, Model]:
    """
    Create a location finding experiment.

    Args:
        spatial_dim: Spatial dimensionality (1, 2, or 3)
        num_sources: Number of sources to locate (K)
        prior_type: "uniform", "gaussian", or "mixture"
        **kwargs: Additional arguments for prior/model:
            For uniform prior:
                - bounds: Tuple of (min, max) values
            For gaussian prior:
                - prior_mean: Mean location, shape (D,)
                - prior_cov: Covariance matrix, shape (D, D)
            For mixture prior:
                - prior_means: Mean locations, shape (M, D)
                - prior_covars: Covariance matrices, shape (M, D, D)
                - prior_weights: Mixture weights, shape (M,)

    Returns:
        Tuple of (prior, model) objects
    """
    # Create prior
    if prior_type == "uniform":
        bounds = kwargs.get("bounds", (-3.0, 3.0))
        prior = UniformPrior(bounds, spatial_dim, num_sources)
    elif prior_type == "gaussian":
        mean = kwargs.get("prior_mean", jnp.zeros(spatial_dim))
        cov = kwargs.get("prior_cov", jnp.eye(spatial_dim))
        prior = GaussianPrior(mean, cov, spatial_dim, num_sources)
    elif prior_type == "mixture":
        means = kwargs.get("prior_means", jnp.zeros((2, spatial_dim)))
        covars = kwargs.get("prior_covars", jnp.stack([jnp.eye(spatial_dim)] * 2))
        weights = kwargs.get("prior_weights", jnp.array([0.5, 0.5]))
        prior = MixtureGaussianPrior(means, covars, weights, spatial_dim, num_sources)
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
    # jax.config.update("jax_disable_jit", True)

    #  Example with 1D grid, 2 sources, 2 sensors, Gaussian mixture prior
    key = random.PRNGKey(0)

    prior_means = jnp.array([
        [[-4.], [-2.]],
        [[2.], [4.]],
    ])
    prior_covars = jnp.array([
        [[[1.0]], [[1.0]]],
        [[[1.0]], [[1.0]]]
    ])
    prior_weights = jnp.array([
        [0.75, 0.25],
        [0.75, 0.25]
    ])

    prior, model = create_location_experiment(
        spatial_dim=1,
        num_sources=2,
        prior_type="mixture",
        prior_means=prior_means,
        prior_covars=prior_covars,
        prior_weights=prior_weights,
    )

    alphas = jnp.array([0.05, 0.95])

    # Create grid of design pairs
    nx = ny = 30  # Grid size
    x = jnp.linspace(-4.0, 4.0, nx)
    y = jnp.linspace(-4.0, 4.0, ny)
    X, Y = jnp.meshgrid(x, y)

    # Pre-allocate array for all surfaces
    info_gain_vals = jnp.zeros((len(alphas), nx, ny))

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 15))
    axes = axes.ravel()

    for i, (alpha, ax) in enumerate(zip(alphas, axes)):
        im = ax.imshow(
            info_gain_vals[i],
            origin='lower',
            extent=[-4.0, 4.0, -4.0, 4.0],
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
