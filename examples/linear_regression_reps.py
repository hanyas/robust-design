from typing import Callable, Tuple
from functools import partial

import jax
import optax

from jax import Array, random, numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from distrax._src.utils import math
from distrax import (
    Bijector,
    Block,
    Transformed,
    MultivariateNormalDiag,
)

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


class Tanh(Bijector):
    def __init__(self):
        super().__init__(event_ndims_in=0)

    def forward_log_det_jacobian(self, x: Array) -> Array:
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        return jnp.tanh(x), self.forward_log_det_jacobian(x)

    def inverse(self, y):
        x = jnp.clip(y, -0.995, 0.995)
        return jnp.arctanh(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        x = self.inverse(y)
        return x, -self.forward_log_det_jacobian(x)


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


class GaussBase(nn.Module):
    """Simple Gaussian distribution with learnable mean and log_std parameters."""
    dim: int
    init_mean: Callable = nn.initializers.ones
    init_log_std: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        mean = self.param("mean", self.init_mean, self.dim)
        log_std = self.param("log_std", self.init_log_std, self.dim)
        return mean, log_std


class GaussPolicy:
    """Simple policy with distribution and bijector."""

    def __init__(
        self,
        dist: GaussBase,
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

    def mean(
        self,
        params,
    ):
        mean, _ = self.dist.apply({"params": params})
        return self.bijector.forward(mean)

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

    def entropy(self, params: dict) -> Array:
        """Compute entropy of the policy."""
        mean, log_std = self.dist.apply({"params": params})
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        return base.entropy()

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
    log_probs: Array,
    tempering: float,
    order: float,
) -> Array:
    # Compute weights
    log_weights = order / (1 - order + order * tempering) * values
    log_weights -= (1 - order) / (1 - order + order * tempering) * log_probs
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
    samples: Array,
    weights: Array,
    policy: GaussPolicy,
    policy_state: TrainState,
) -> tuple[TrainState, Array]:
    """Update policy parameters using weighted maximum likelihood."""
    def loss_fn(params):
        log_probs = jax.vmap(lambda s: policy.log_prob(s, params))(samples)
        return - 1.0 * jnp.sum(weights * log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(policy_state.params)
    policy_state = policy_state.apply_gradients(grads=grads)
    return policy_state, loss


def optimize(
    key: Array,
    func: Callable,
    policy: GaussPolicy,
    policy_state: TrainState,
    tempering: float,
    order: float,
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
        policy_state: Initial training state
        tempering: Tempering factor
        order: Order of the approximation
        num_samples: Number of samples to use for each iteration
        batch_size: Number of samples to use for each batch
        max_iterations: Maximum number of iterations

    Returns:
        Tuple of (optimized train_state, optimization info)
    """

    def step(key, policy_state):
        # Generate samples from current policy
        sample_key, batch_key = random.split(key, 2)

        sample_keys = random.split(sample_key, num_samples)
        samples, log_probs = jax.vmap(
            lambda k: policy.sample_and_log_prob(k, policy_state.params)
        )(sample_keys)

        # Compute values
        values = jax.vmap(lambda x: func(design=x, order=order))(samples)

        # Compute weights
        weights = get_weights(values, log_probs, tempering, order)

        # Update policy
        key, sub_key = random.split(batch_key)
        batch_indices = batch_samples(sub_key, num_samples, batch_size)

        for batch_idx in batch_indices:
            samples_batch = jax.tree_util.tree_map(lambda x: x[batch_idx, ...], samples)
            weights_batch = jax.tree_util.tree_map(lambda x: x[batch_idx, ...], weights)

            policy_state, _ = update_policy(
                samples=samples_batch,
                weights=weights_batch,
                policy=policy,
                policy_state=policy_state,
            )

        return policy_state, {
            'fval': jnp.mean(values),
            'mean': jnp.mean(samples, axis=0),
            'entropy': policy.entropy(policy_state.params)
        }

    # Run optimization
    info = {}
    for i in range(max_iterations):
        key, step_key = random.split(key)
        policy_state, info = step(step_key, policy_state)

        value, mean, entropy = info['fval'], info['mean'], info['entropy']
        mean_str = ", ".join([f"{x:+.3f}" for x in mean])
        print(f"iter {i:3d}, fval: {value:+.3f}, x: [{mean_str}], entropy: {entropy:+.3f}")

    return policy_state


###

# Configuration for REPS optimizer
num_samples = 1024                # Number of samples to use for each iteration
tempering: float = 1e-2           # Tempering value
order: float = 0.9999              # Renyi order
batch_size: int = 64              # Number of samples to use for each batch
learning_rate: float = 1e-2       # Learning rate for the optimizer
max_iterations: int = 100        # Maximum number of iterations

# Initialize
key = jax.random.PRNGKey(0)

# Create policy
dim = 1
init_mean = lambda key, shape: 0.0 * jnp.ones((dim,))
init_log_std = lambda key, shape: jnp.log(1.0) * jnp.ones((dim,))
dist = GaussBase(
    dim=dim,
    init_mean=init_mean,
    init_log_std=init_log_std
)

bijector = Tanh()
policy = GaussPolicy(dist, bijector)

# Initialize training state
init_key, opt_key = jax.random.split(key)
policy_state = policy.initialize(init_key, learning_rate)


# sibson mutual info
def sibsons_estimator(
    design: Array,  # shape (N,)
    prior_cov: Array = 1. * jnp.eye(2),  # shape (d, d)
    lklhd_cov: float = 1e-1 * jnp.eye(1),  # shape (1, 1)
    order: float = 0.95,
):
    N = design.shape[0]
    A = jnp.concatenate([
        jnp.ones((N, 1)),
        jnp.expand_dims(design, axis=1)
    ], axis=1)

    Lambda = A @ prior_cov @ A.T + (lklhd_cov / order) * jnp.eye(N)
    _, logdet = jnp.linalg.slogdet(Lambda)
    return 0.5 * logdet


# Run optimization
policy_state = optimize(
    key=opt_key,
    func=sibsons_estimator,
    policy=policy,
    policy_state=policy_state,
    tempering=tempering,
    order=order,
    num_samples=num_samples,
    batch_size=batch_size,
    max_iterations=max_iterations,
)


# plot x = f(xi) = 1 + xi + 4*xi^2
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use("Qt5Agg")

range_xi = jnp.linspace(-1.0, 1.0, 100)
f = lambda xi: 1 + xi + 4 * xi ** 2
h = lambda xi: xi + 5.0

# Create figure with 3 subplots
# Collect all samples for histograms
all_opt_xi = []
all_xs = []

for _ in range(num_samples):
    key, sub_key = random.split(key)
    sample_key, plt_key = random.split(sub_key)
    opt_xi = policy.sample(sample_key, policy_state.params)
    xs = jax.vmap(f)(opt_xi) + jnp.sqrt(1e-1) * random.normal(plt_key, opt_xi.shape)

    all_opt_xi.append(opt_xi)
    all_xs.append(xs)

# Convert to arrays
all_opt_xi = jnp.array(all_opt_xi)  # shape: (num_samples, 2)
all_xs = jnp.array(all_xs)  # shape: (num_samples, 2)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [2, 1, 1]})

ax1.plot(range_xi, f(range_xi), color='k', linewidth=3, ls='--')
ax1.plot(range_xi, h(range_xi), color='k', linewidth=3)
# ax1.plot(range_xi, f(range_xi), label=r"$x = 1 + \xi + 4\xi^2$", color='C0', linewidth=3)
ax1.scatter(all_opt_xi[:, 0], all_xs[:, 0], color='blue', label=r"$\xi_1$",
            zorder=5, s=10, edgecolor='black', linewidth=1.5, alpha=0.85, marker='o')
ax1.scatter(all_opt_xi[:, 1], all_xs[:, 1], color='crimson', label=r"$\xi_2$",
            zorder=5, s=10, edgecolor='black', linewidth=1.5, alpha=0.85, marker='s')
ax1.set_xlabel(r"$\xi$", fontsize=24)
ax1.set_ylabel(r"$x$", fontsize=24)
# ax1.set_title(rf"Linear Regression - $\alpha = {order}$", fontsize=26, pad=20)
ax1.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(f(range_xi).min() - 1, f(range_xi).max() + 1)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.legend(fontsize=18, framealpha=0.9)

# Histogram for first dimension
counts1, bins1, _ = ax2.hist(all_opt_xi[:, 0], bins=200, alpha=0.7, color='skyblue',
                              edgecolor='black', linewidth=1.5)
ax2.set_xlabel(r"$\xi_1$", fontsize=24)
ax2.set_ylabel("Frequency", fontsize=24)
ax2.set_title(r"Distribution over $\xi_1$", fontsize=26, pad=20)
ax2.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
ax2.set_xlim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=18)

# Histogram for second dimension
counts2, bins2, _ = ax3.hist(all_opt_xi[:, 1], bins=200, alpha=0.7, color='lightgreen',
                              edgecolor='black', linewidth=1.5)
ax3.set_xlabel(r"$\xi_2$", fontsize=24)
ax3.set_ylabel("Frequency", fontsize=24)
ax3.set_title(r"Distribution over $\xi_2$", fontsize=26, pad=20)
ax3.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
ax3.set_xlim(-1, 1)
ax3.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.show()
