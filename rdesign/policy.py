from typing import Callable

from jax import Array, numpy as jnp
from flax import linen as nn

from distrax import (
    Bijector,
    Transformed,
    MultivariateNormalDiag,
    Block
)

from rdesign.base import Policy


class GaussianParams(nn.Module):
    """Learnable mean and log_std parameters for a Gaussian distribution."""
    dim: int
    init_mean: Callable = nn.initializers.ones
    init_log_std: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self) -> tuple[Array, Array]:
        mean = self.param("mean", self.init_mean, self.dim)
        log_std = self.param("log_std", self.init_log_std, self.dim)
        return mean, log_std


class ParametricGaussian(Policy):
    """Gaussian policy with optional bijector for constrained action spaces."""

    def __init__(
        self,
        params_module: GaussianParams,
        bijector: Bijector
    ):
        self.params_module = params_module
        self.dim = params_module.dim
        self._bijector = Block(bijector, ndims=1)

    def _get_distribution(self, params: dict) -> Transformed:
        """Build distribution from parameters."""
        mean, log_std = self.params_module.apply({"params": params})
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        return Transformed(distribution=base, bijector=self._bijector)

    def sample(self, rng_key: Array, params: dict) -> Array:
        """Sample from the policy."""
        return self._get_distribution(params).sample(seed=rng_key)

    def log_prob(self, sample: Array, params: dict) -> Array:
        """Compute log probability of sample under policy."""
        return self._get_distribution(params).log_prob(sample)

    def sample_and_log_prob(self, rng_key: Array, params: dict) -> tuple[Array, Array]:
        """Sample and compute log probability."""
        return self._get_distribution(params).sample_and_log_prob(seed=rng_key)

    def entropy(self, params: dict) -> Array:
        """Compute entropy of the base Gaussian."""
        _, log_std = self.params_module.apply({"params": params})
        return 0.5 * jnp.sum(2 * log_std + jnp.log(2 * jnp.pi * jnp.e))

    def init_params(self, rng_key: Array) -> dict:
        """Initialize policy parameters."""
        return self.params_module.init(rng_key)["params"]
