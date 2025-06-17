from typing import Protocol
from jax import Array


class Prior(Protocol):
    """Protocol for prior distributions."""
    
    def sample(self, key: Array, num_samples: int) -> Array:
        """Sample from the prior distribution."""
        ...
    
    def logpdf(self, x: Array) -> Array:
        """Compute log probability density function."""
        ...


class Model(Protocol):
    """Protocol for likelihood models."""
    
    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        """Sample from the likelihood p(x | design, theta)."""
        ...
    
    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """Compute log likelihood p(x | design, theta)."""
        ...
