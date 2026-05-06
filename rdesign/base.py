from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol, NamedTuple

from jax import Array
from flax.training.train_state import TrainState
from optax import GradientTransformation


class Distribution(Protocol):
    """Base protocol for all distributions.

    Uses structural typing - any class with these methods satisfies the protocol.
    """

    def sample(self, key: Array, num_samples: int) -> Array:
        """Sample from the distribution.

        Args:
            key: Random key
            num_samples: Number of samples to draw

        Returns:
            Samples of shape (num_samples, dim)
        """
        ...

    def logpdf(self, value: Array) -> Array:
        """Compute log probability density.

        Args:
            value: Value(s) to evaluate, shape (dim,) or (n, dim)

        Returns:
            Log probability, scalar or shape (n,)
        """
        ...

    @property
    def dim(self) -> int:
        """Dimension of the distribution."""
        ...


class Prior(Distribution, Protocol):
    """Prior distribution p(θ) over parameters.

    Extends Distribution with no additional requirements.
    Any Distribution can serve as a Prior.
    """

    pass


class Likelihood(Protocol):
    """Likelihood model p(x | θ, ξ).

    Maps parameters θ and designs ξ to observation distributions.
    """

    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        """Sample observations given parameters and design.

        Args:
            key: Random key
            design: Design points ξ
            theta: Parameters θ

        Returns:
            Observations x
        """
        ...


class ProposalKernel(NamedTuple):
    """Generic Metropolis-Hastings proposal kernel."""

    sample: Callable[[Array, Array], Array]
    log_prob: Callable[[Array, Array], Array]


class IntegerLadderMCMCConfig(NamedTuple):
    """Configuration for the generic integer-ladder MCMC sampler."""

    num_warmup_steps: int = 250
    num_retained_samples: int = 8
    thinning: int = 1
    compute_ess: bool = True


class IntegerLadderMCMCResult(NamedTuple):
    """Output of the generic integer-ladder MCMC sampler."""

    x: Array
    x_by_chain: Array
    theta_copies: Array
    theta_copies_by_chain: Array
    x_acceptance_rate: Array
    theta_acceptance_rate: Array
    ess_x: Array
    ess_theta: Array


class RussianRouletteMCMCConfig(NamedTuple):
    """Configuration for the generic Russian-roulette pseudo-marginal sampler."""

    num_warmup_steps: int = 500
    num_retained_samples: int = 16
    thinning: int = 1
    rr_chunk_size: int = 16
    compute_ess: bool = True


class RussianRouletteMCMCResult(NamedTuple):
    """Output of the generic Russian-roulette pseudo-marginal sampler."""

    x: Array
    x_by_chain: Array
    t: Array
    t_by_chain: Array
    theta_copies: Array
    theta_copies_by_chain: Array
    rr_counts: Array
    rr_counts_by_chain: Array
    x_acceptance_rate: Array
    t_acceptance_rate: Array
    theta_acceptance_rate: Array
    ess_x: Array
    ess_t: Array
    ess_theta: Array

    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """Compute log likelihood p(x | θ, ξ).

        Args:
            x: Observations
            design: Design points ξ
            theta: Parameters θ

        Returns:
            Log likelihood
        """
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
    """Configuration for REPS optimization.

    Args:
        num_samples: Number of samples per iteration.
        batch_size: Mini-batch size for updates.
        max_iterations: Total number of iterations.
        log_interval: Iterations between progress logs.
        verbose: Whether to print progress.
    """

    num_samples: int = 100
    batch_size: int = 16
    max_iterations: int = 1000
    log_interval: int = 100
    verbose: bool = True
    lmbda: float = 1.0


class StepInfo(NamedTuple):
    """Information from a single optimization step."""

    mean_value: Array
    entropy: Array
    temperature: float


class OptimizationResult(NamedTuple):
    """Result of policy optimization."""

    state: TrainState
    history: dict[str, list]
