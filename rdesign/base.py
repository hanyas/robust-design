from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, NamedTuple

from jax import Array
from flax.training.train_state import TrainState
from optax import GradientTransformation


class Prior(Protocol):
    """Protocol for prior distributions."""

    def sample(self, key: Array, num_samples: int) -> Array:
        """Sample from the prior distribution."""
        ...


class Model(Protocol):
    """Protocol for likelihood models."""

    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        """Sample from the likelihood p(x | design, theta)."""
        ...

    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """Compute log likelihood p(x | design, theta)."""
        ...


class Policy(ABC):
    """Abstract base class for policies."""

    @abstractmethod
    def sample(self, rng_key: Array, params: dict) -> Array:
        """Sample from the policy."""
        ...

    @abstractmethod
    def log_prob(self, sample: Array, params: dict) -> Array:
        """Compute log probability of sample under policy."""
        ...

    @abstractmethod
    def sample_and_log_prob(self, rng_key: Array, params: dict) -> tuple[Array, Array]:
        """Sample and compute log probability."""
        ...

    @abstractmethod
    def entropy(self, params: dict) -> Array:
        """Compute entropy of the policy."""
        ...

    @abstractmethod
    def init_params(self, rng_key: Array) -> dict:
        """Initialize policy parameters."""
        ...

    def init_state(
        self,
        rng_key: Array,
        optimizer: GradientTransformation,
    ) -> TrainState:
        """Initialize training state with given optimizer."""
        return TrainState.create(
            apply_fn=None,
            params=self.init_params(rng_key),
            tx=optimizer,
        )


@dataclass
class REPSConfig:
    """Configuration for REPS (mirror descent) optimization.

    Args:
        num_samples: Number of samples per iteration for Monte Carlo estimation.
        batch_size: Mini-batch size for stochastic gradient updates.
        max_iterations: Total number of mirror descent iterations.
        log_interval: Iterations between progress logs.
        verbose: Whether to print progress.
    """
    num_samples: int = 100
    batch_size: int = 16
    max_iterations: int = 1000
    log_interval: int = 100
    verbose: bool = True


class StepInfo(NamedTuple):
    """Information from a single optimization step."""
    mean_value: Array
    entropy: Array
    temperature: float


class OptimizationResult(NamedTuple):
    """Result of policy optimization."""
    policy_state: TrainState
    history: dict[str, list]