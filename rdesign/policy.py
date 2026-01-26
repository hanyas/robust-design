from typing import Callable

import jax
from jax import Array, random
from jax import numpy as jnp

from flax import linen as nn
from distrax import Bijector, Block
from distrax import MultivariateNormalDiag, Transformed

from rdesign.base import Policy


class BoxGaussian(Policy):
    """Gaussian policy with optional bijector for constrained action spaces."""

    def __init__(
        self,
        dim: int,
        bijector: Bijector,
        init_mean: Callable = nn.initializers.ones,
        init_log_std: Callable = nn.initializers.zeros,
    ):
        self.dim = dim
        self._bijector = Block(bijector, ndims=1)
        self.init_mean = init_mean
        self.init_log_std = init_log_std

    def init_params(self, rng_key: Array) -> dict:
        """Initialize policy parameters."""
        mean_key, log_std_key = random.split(rng_key)
        mean = self.init_mean(mean_key, (self.dim,))
        log_std = self.init_log_std(log_std_key, (self.dim,))
        return {"mean": mean, "log_std": log_std}

    def _get_distribution(self, params: dict) -> Transformed:
        """Build distribution from parameters."""
        mean, log_std = params["mean"], params["log_std"]
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
        log_std = params["log_std"]
        return 0.5 * jnp.sum(2 * log_std + jnp.log(2 * jnp.pi * jnp.e))


class Categorical(Policy):
    """Categorical policy over a range of discrete choices."""

    def __init__(self, num_options: int):
        self.num_options = num_options

    def init_params(self, rng_key: Array) -> dict:
        """Initialize logits uniformly."""
        return {"logits": jnp.zeros(self.num_options)}

    def sample(self, rng_key: Array, params: dict) -> Array:
        """Sample an option."""
        return random.categorical(rng_key, params["logits"]).astype(jnp.float64)

    def log_prob(self, sample: Array, params: dict) -> Array:
        """Compute log probability of a sample."""
        idx = sample.astype(jnp.int32)
        log_probs = jax.nn.log_softmax(params["logits"])
        return log_probs[idx]

    def sample_and_log_prob(self, rng_key: Array, params: dict) -> tuple[Array, Array]:
        """Sample and compute log probability."""
        sample = self.sample(rng_key, params)
        return sample, self.log_prob(sample, params)

    def entropy(self, params: dict) -> Array:
        """Compute categorical entropy."""
        probs = jax.nn.softmax(params["logits"])
        return -jnp.sum(probs * jnp.log(probs + 1e-10))
