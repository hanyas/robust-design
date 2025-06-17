from typing import Callable, Tuple
from functools import partial

import math
import jax
import optax

from jax import Array, random, numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from distrax import (
    Bijector,
    Transformed,
    MultivariateNormalDiag,
    Block
)


def batch_samples(
    rng_key: Array,
    sample_size: int,
    batch_size: int,
    skip_last: bool = True,
) -> list[Array]:
    batch_idx = random.permutation(rng_key, sample_size)

    if skip_last:
        # Skip incomplete batch
        num_batches = sample_size // batch_size
        batch_idx = batch_idx[: num_batches * batch_size]
    else:
        # Include incomplete batch
        num_batches = math.ceil(sample_size / batch_size)

    batch_idx = jnp.array_split(batch_idx, num_batches)
    return batch_idx


class Gaussian(nn.Module):
    """Simple Gaussian distribution with learnable mean and log_std parameters."""
    dim: int
    init_mean: Callable = nn.initializers.ones
    init_log_std: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        mean = self.param("mean", self.init_mean, self.dim)
        log_std = self.param("log_std", self.init_log_std, self.dim)
        return mean, log_std


class Policy:
    """Simple policy with distribution and bijector."""

    def __init__(
        self,
        dist: Gaussian,
        bijector: Bijector,
    ):
        self.dist = dist
        self.dim = dist.dim
        self.bijector = Block(bijector, ndims=1)

    def sample(
        self,
        key: Array,
        params: dict,
    ) -> Array:
        """Sample from the policy."""
        mean, log_std = self.dist.apply({"params": params})
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        trans = Transformed(distribution=base, bijector=self.bijector)
        return trans.sample(seed=key)

    def log_prob(
        self,
        sample: Array,
        params: dict,
    ) -> float:
        """Compute log probability of action under policy."""
        mean, log_std = self.dist.apply({"params": params})
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        trans = Transformed(distribution=base, bijector=self.bijector)
        return trans.log_prob(sample)

    def sample_and_log_prob(
        self,
        key: Array,
        params: dict,
    ) -> Tuple[Array, float]:
        """Sample and compute log probability."""
        mean, log_std = self.dist.apply({"params": params})
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        trans = Transformed(distribution=base, bijector=self.bijector)
        sample, log_prob = trans.sample_and_log_prob(seed=key)
        return sample, log_prob

    def entropy(self, params: dict) -> float:
        """Compute entropy of the policy."""
        _, log_std = self.dist.apply({"params": params})
        return 0.5 * jnp.sum(2 * log_std + jnp.log(2 * jnp.pi * jnp.exp(1.0)))

    def initialize(
        self,
        key: Array,
        learning_rate: float
    ) -> TrainState:
        """Initialize training state with network parameters."""
        init_params = self.dist.init(key)["params"]
        return TrainState.create(
            apply_fn=None,
            params=init_params,
            tx=optax.adam(learning_rate)
        )


def get_weights(
    values: Array,
    tempering: float
) -> Array:
    # Compute weights
    log_weights = tempering * values
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jax.nn.softmax(log_weights)
    return weights


def get_kl_div(
    weights: Array,
) -> Array:
    # Compute KL divergence
    w = jnp.clip(weights, 1e-4, jnp.inf)
    w = w / jnp.mean(w, axis=0)
    return jnp.mean(w * jnp.log(w), axis=0)


@partial(jax.jit, static_argnames="policy")
def update_policy(
    learner: TrainState,
    policy: Policy,
    samples: Array,
    weights: Array,
) -> tuple[TrainState, Array]:
    """Update policy parameters using weighted maximum likelihood."""
    def loss_fn(params):
        log_probs = jax.vmap(lambda s: policy.log_prob(s, params))(samples)
        return - 1.0 * jnp.sum(weights * log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, loss


def optimize(
    key: Array,
    func: Callable,
    policy: Policy,
    learner: TrainState,
    tempering: float,
    num_samples: int,
    batch_size: int,
    max_iterations: int,
) -> Tuple[TrainState, dict]:
    """
    Optimize the policy using REPS.

    Args:
        key: JAX PRNG key
        func: Function to optimize
        policy: Policy to optimize
        learner: Initial training state
        tempering: Tempering factor
        num_samples: Number of samples to use for each iteration
        batch_size: Number of samples to use for each batch
        max_iterations: Maximum number of iterations

    Returns:
        Tuple of (optimized train_state, optimization info)
    """

    def step(key, learner):
        # Generate samples from current policy
        sample_key, batch_key = random.split(key, 2)

        sample_keys = random.split(sample_key, num_samples)
        samples = jax.vmap(lambda k: policy.sample(k, learner.params))(sample_keys)

        # Compute values
        values = jax.vmap(func)(samples)

        # Compute weights
        weights = get_weights(values, tempering)

        # Update policy
        key, sub_key = random.split(batch_key)
        batch_indices = batch_samples(sub_key, num_samples, batch_size)

        for batch_idx in batch_indices:
            samples_batch = jax.tree.map(lambda x: x[batch_idx, ...], samples)
            weights_batch = jax.tree.map(lambda x: x[batch_idx, ...], weights)

            learner, _ = update_policy(
                policy=policy,
                learner=learner,
                samples=samples_batch,
                weights=weights_batch,
            )

        return learner, {
            'value': jnp.mean(values),
            'entropy': policy.entropy(learner.params)
        }

    # Run optimization
    info = {}
    for i in range(max_iterations):
        key, step_key = random.split(key)
        learner, info = step(step_key, learner)

        value, entropy = info['value'], info['entropy']
        mean_str = ", ".join([f"{x:.3f}" for x in learner.params['mean']])
        print(f"Iteration {i}, Value: {value:.3f}, Entropy: {entropy:.3f}, Mean: [{mean_str}]")

    return learner


def main():
    dim = 2

    # Configuration for REPS optimizer
    num_samples = 100               # Number of samples to use for each iteration
    tempering: float = 100          # Tempering value
    batch_size: int = 16            # Number of samples to use for each batch
    learning_rate: float = 0.01     # Learning rate for the optimizer
    max_iterations: int = 1000       # Maximum number of iterations

    # Initialize
    key = jax.random.PRNGKey(0)

    # Create policy
    dist = Gaussian(dim=dim)

    # Using custom initializers
    init_mean = lambda key, shape: 2.5 * jnp.ones((dim,))
    init_log_std = lambda key, shape: jnp.zeros((dim,))
    dist = Gaussian(
        dim=dim,
        init_mean=init_mean,
        init_log_std=init_log_std
    )

    bijector = lambda x: x
    policy = Policy(dist, bijector)

    # Initialize training state
    init_key, opt_key = jax.random.split(key)
    learner = policy.initialize(init_key, learning_rate)

    # Example func
    quadratic_fn = lambda x: -0.5 * jnp.dot(x, x)

    # Run optimization
    learner = optimize(
        key=opt_key,
        func=quadratic_fn,
        policy=policy,
        learner=learner,
        tempering=tempering,
        num_samples=num_samples,
        batch_size=batch_size,
        max_iterations=max_iterations,
    )


if __name__ == "__main__":
    main()
