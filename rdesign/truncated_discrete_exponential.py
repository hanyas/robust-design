# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TruncatedDiscreteExponential distribution."""

from typing import Tuple

import chex
from distrax._src.distributions import distribution
from distrax._src.utils import conversion
import jax.numpy as jnp
from jax.random import uniform
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
Numeric = chex.Numeric
PRNGKey = chex.PRNGKey


class TruncatedDiscreteExponential(distribution.Distribution):
    """Truncated discrete exponential distribution with parameter `rate`.

    The rate parameter must be positive (rate > 0) to ensure valid probabilities.
    The distribution is defined on the non-negative integers, optionally truncated
    between lower and upper bounds.
    """

    equiv_tfp_cls = None  # No equivalent TFP distribution

    def __init__(
        self,
        rate: Numeric,
        lower: Numeric = 0,
        upper: Numeric = None,
    ):
        """Initialize a truncated discrete exponential distribution.

        Args:
            rate: The rate parameter of the exponential distribution. Must be positive.
            lower: The lower truncation bound (inclusive). Must be non-negative.
            upper: The upper truncation bound (inclusive). If None, no upper bound is applied.
                If specified, must be greater than or equal to lower.

        Raises:
            ValueError: If rate is not positive, or if bounds are invalid.
        """
        super().__init__()
        self._rate = conversion.as_float_array(rate)
        
        # Validate rate parameter
        if jnp.any(self._rate <= 0):
            raise ValueError("Rate parameter must be positive")
            
        self._lower = conversion.as_float_array(lower)
        if jnp.any(self._lower < 0):
            raise ValueError("Lower bound must be non-negative")
            
        self._upper = conversion.as_float_array(upper) if upper is not None else None
        if self._upper is not None and jnp.any(self._upper < self._lower):
            raise ValueError("Upper bound must be greater than or equal to lower bound")
            
        self._batch_shape = jnp.broadcast_shapes(
            self._rate.shape,
            self._lower.shape,
            self._upper.shape if self._upper is not None else (),
        )

    @property
    def rate(self) -> Array:
        """Rate parameter of the distribution."""
        return jnp.broadcast_to(self._rate, self.batch_shape)

    @property
    def lower(self) -> Array:
        """Lower truncation bound."""
        return jnp.broadcast_to(self._lower, self.batch_shape)

    @property
    def upper(self) -> Array:
        """Upper truncation bound."""
        if self._upper is None:
            return None
        return jnp.broadcast_to(self._upper, self.batch_shape)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Shape of batch of distributions."""
        return self._batch_shape

    @property
    def event_shape(self) -> Tuple[int, ...]:
        """Shape of event of distribution samples."""
        return ()

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        """See `Distribution._sample_n`."""
        # Generate uniform random numbers
        u = uniform(key, shape=(n,) + self.batch_shape)
        
        # Compute the CDF values at truncation bounds
        lower_cdf = 1 - jnp.exp(-self.rate * self.lower)
        if self.upper is not None:
            upper_cdf = 1 - jnp.exp(-self.rate * (self.upper + 1))
        else:
            upper_cdf = 1.0
            
        # Scale uniform samples to the truncated range
        scaled_u = lower_cdf + u * (upper_cdf - lower_cdf)
        
        # Transform back to the domain using the inverse CDF
        x = -jnp.log(1 - scaled_u) / self.rate
        
        # Round to nearest integer
        return jnp.floor(x).astype(jnp.int32)

    def log_prob(self, value: Array) -> Array:
        """See `Distribution.log_prob`."""
        # Ensure value is within bounds
        value = jnp.asarray(value, dtype=jnp.int32)
        valid = value >= self.lower
        if self.upper is not None:
            valid = valid & (value <= self.upper)
            
        # Compute log probability
        log_prob = -self.rate * value - jnp.log(1 - jnp.exp(-self.rate))
        
        # Normalize by truncation bounds
        lower_cdf = 1 - jnp.exp(-self.rate * self.lower)
        if self.upper is not None:
            upper_cdf = 1 - jnp.exp(-self.rate * (self.upper + 1))
            log_norm = jnp.log(upper_cdf - lower_cdf)
        else:
            log_norm = -jnp.log(1 - lower_cdf)
            
        log_prob = log_prob - log_norm
        
        # Return -inf for values outside bounds
        return jnp.where(valid, log_prob, -jnp.inf)

    def mean(self) -> Array:
        """Mean of the distribution."""
        # For truncated discrete exponential, we need to compute the mean numerically
        # This is a simplified version that works for the truncated case
        if self.upper is not None:
            x = jnp.arange(self.lower, self.upper + 1)
            probs = jnp.exp(self.log_prob(x))
            return jnp.sum(x * probs)
        else:
            # For non-truncated case, mean is 1/rate
            return 1.0 / self.rate

    def variance(self) -> Array:
        """Variance of the distribution."""
        # For truncated discrete exponential, we need to compute the variance numerically
        if self.upper is not None:
            x = jnp.arange(self.lower, self.upper + 1)
            probs = jnp.exp(self.log_prob(x))
            mean = self.mean()
            return jnp.sum((x - mean) ** 2 * probs)
        else:
            # For non-truncated case, variance is 1/rate^2
            return 1.0 / (self.rate ** 2) 