import jax
from jax import Array, random
import jax.numpy as jnp
from typing import NamedTuple


class MM_Prior:
    def sample(self, key: Array, num_samples: int) -> Array:
        theta = random.uniform(key, shape=(num_samples, 2)) * jnp.array([20.0, 200.0])
        return theta


class MM_Model(NamedTuple):
    """The Michaelis-Menten kinetics model from Forster, Ivanova and Rainforth (2025).

    The assumed model has `s = 1.0`, and the degree of misspecification is controlled
    by varying `s`.
    """

    s: float = 1.0
    obs_noise_std: float = 5.0

    def mean_response(self, design: Array, theta: Array) -> Array:
        v_max, v_half = theta
        return v_max * design**self.s / (design**self.s + v_half**self.s)

    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        rand = random.normal(key)
        return self.mean_response(design, theta) + rand * self.obs_noise_std

    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        mean = self.mean_response(design, theta)
        return jax.scipy.stats.norm.logpdf(x, loc=mean, scale=self.obs_noise_std)
