import jax
import optax

from jax import Array, random, numpy as jnp

from rdesign.base import REPSConfig
from rdesign.policy import GaussianParams, GaussianPolicy
from rdesign.reps import optimize_policy
from rdesign.bijectors import Tanh

jax.config.update("jax_enable_x64", True)


# Sibson mutual information estimator for linear regression
def sibsons_estimator(
    design: Array,
    prior_cov: Array = 1.0 * jnp.eye(2),
    lklhd_cov: Array = 0.1 * jnp.eye(1),
    order: float = 0.95,
) -> Array:
    """Compute Sibson mutual information for linear regression.
    
    Args:
        design: Design points, shape (N,)
        prior_cov: Prior covariance, shape (2, 2)
        lklhd_cov: Likelihood covariance, shape (1, 1)
        order: Sibson order parameter alpha in (0, 1)
    
    Returns:
        Mutual information value (scalar)
    """
    N = design.shape[0]
    A = jnp.concatenate(
        [
            jnp.ones((N, 1)),
            jnp.expand_dims(design, axis=1)
        ]
    , axis=1)

    Lambda = A @ prior_cov @ A.T + (lklhd_cov / order) * jnp.eye(N)
    _, logdet = jnp.linalg.slogdet(Lambda)
    return 0.5 * logdet


# Configuration
dim = 2
order = 0.995
learning_rate = 1e-2

# REPS configuration
config = REPSConfig(
    num_samples=1024,
    batch_size=64,
    max_iterations=100,
    log_interval=1,
    verbose=True,
)

# Initialize RNG
rng_key = jax.random.PRNGKey(0)
init_key, opt_key, plot_key = jax.random.split(rng_key, 3)

# Create policy with Tanh bijector to constrain to (-1, 1)
params_module = GaussianParams(
    dim=dim,
    init_mean=lambda key, shape: 0.0 * jnp.ones((dim,)),
    init_log_std=lambda key, shape: jnp.log(1.0) * jnp.ones((dim,)),
)
policy = GaussianPolicy(params_module, bijector=Tanh())

# Initialize training state
optimizer = optax.adam(learning_rate)
policy_state = policy.init_state(init_key, optimizer)

# Define objective (Sibson mutual information)
objective = lambda x: sibsons_estimator(design=x, order=order)

# Run optimization
result = optimize_policy(
    rng_key=opt_key,
    func=objective,
    policy=policy,
    policy_state=policy_state,
    config=config,
)

policy_state = result.policy_state

# Visualization
import matplotlib.pyplot as plt

num_samples = 1024
range_xi = jnp.linspace(-1.0, 1.0, 100)
f = lambda xi: 1 + xi + 4 * xi ** 2
h = lambda xi: xi + 5.0

# Collect samples for histograms
all_opt_xi = []
all_xs = []

for _ in range(num_samples):
    plot_key, sample_key, noise_key = random.split(plot_key, 3)
    opt_xi = policy.sample(sample_key, policy_state.params)
    xs = jax.vmap(f)(opt_xi) + jnp.sqrt(1e-1) * random.normal(noise_key, opt_xi.shape)

    all_opt_xi.append(opt_xi)
    all_xs.append(xs)

all_opt_xi = jnp.array(all_opt_xi)
all_xs = jnp.array(all_xs)

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [2, 1, 1]})

ax1.plot(range_xi, f(range_xi), color='k', linewidth=3, ls='--')
ax1.plot(range_xi, h(range_xi), color='k', linewidth=3)
ax1.scatter(all_opt_xi[:, 0], all_xs[:, 0], color='blue', label=r"$\xi_1$",
            zorder=5, s=10, edgecolor='black', linewidth=1.5, alpha=0.85, marker='o')
ax1.scatter(all_opt_xi[:, 1], all_xs[:, 1], color='crimson', label=r"$\xi_2$",
            zorder=5, s=10, edgecolor='black', linewidth=1.5, alpha=0.85, marker='s')
ax1.set_xlabel(r"$\xi$", fontsize=24)
ax1.set_ylabel(r"$x$", fontsize=24)
ax1.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(f(range_xi).min() - 1, f(range_xi).max() + 1)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.legend(fontsize=18, framealpha=0.9)

# Histogram for first dimension
ax2.hist(all_opt_xi[:, 0], bins=200, alpha=0.7, color='skyblue',
            edgecolor='black', linewidth=1.5)
ax2.set_xlabel(r"$\xi_1$", fontsize=24)
ax2.set_ylabel("Frequency", fontsize=24)
ax2.set_title(r"Distribution over $\xi_1$", fontsize=26, pad=20)
ax2.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
ax2.set_xlim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=18)

# Histogram for second dimension
ax3.hist(all_opt_xi[:, 1], bins=200, alpha=0.7, color='lightgreen',
            edgecolor='black', linewidth=1.5)
ax3.set_xlabel(r"$\xi_2$", fontsize=24)
ax3.set_ylabel("Frequency", fontsize=24)
ax3.set_title(r"Distribution over $\xi_2$", fontsize=26, pad=20)
ax3.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
ax3.set_xlim(-1, 1)
ax3.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.show()
