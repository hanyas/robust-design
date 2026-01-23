"""
Robust Experimental Design Optimization using REPS

This script optimizes experimental designs using:
- LinearRegression class with Sibson's α-mutual information
- REPS (Relative Entropy Policy Search) for policy optimization
- Gaussian policy over design space
"""

import jax
import optax

from jax import Array, random, numpy as jnp

from rdesign.base import REPSConfig
from rdesign.policy import GaussianParams, GaussianPolicy
from rdesign.reps import optimize_policy
from rdesign.bijectors import Tanh

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian, LinearRegression

jax.config.update("jax_enable_x64", True)


# Model parameters
prior_mean = jnp.zeros(2)  # Prior mean: [slope, intercept]
prior_cov = 1.0 * jnp.eye(2)  # Prior covariance
noise_var = 0.1  # Observation noise variance

# Design optimization parameters
dim = 2  # Number of design points to optimize
order = 0.995  # Sibson order parameter (alpha) - closer to 1 = less robust
learning_rate = 1e-2

# Create prior and likelihood
prior = Gaussian(mean=prior_mean, cov=prior_cov)
likelihood = LinearGaussian(noise_var=noise_var)

# Create LinearRegression model
model = LinearRegression(prior=prior, likelihood=likelihood)

# Define objective using LinearRegression class
objective = lambda design: model.sibson_alpha_mutual_information(design, alpha=order)

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

print("=" * 80)
print("RUNNING REPS OPTIMIZATION")
print("=" * 80)
print()

result = optimize_policy(
    rng_key=opt_key,
    func=objective,
    policy=policy,
    policy_state=policy_state,
    config=config,
)

policy_state = result.policy_state

print()
print("=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)
print()

# Sample from optimized policy
final_key, sample_key = random.split(plot_key)
optimal_design = policy.sample(sample_key, policy_state.params)

# Evaluate objective at optimal design
optimal_value = model.sibson_alpha_mutual_information(optimal_design, alpha=order)
shannon_value = model.shannon_mutual_information(optimal_design)

print(f"Optimal design found:       {optimal_design}")
print(f"Sibson's α-MI (α={order}):  {optimal_value:.6f}")
print(f"Shannon's MI (α=1):         {shannon_value:.6f}")
print(f"Robustness gap:             {shannon_value - optimal_value:.6f}")
print()

import matplotlib.pyplot as plt

print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)
print()

num_samples = 1024
range_xi = jnp.linspace(-1.0, 1.0, 100)

# Example functions for visualization
h = lambda xi: xi

# Collect samples for histograms
all_opt_xi = []
all_xs = []

for _ in range(num_samples):
    plot_key, sample_key, theta_key, noise_key = random.split(plot_key, 4)
    
    # Sample design from optimized policy
    opt_xi = policy.sample(sample_key, policy_state.params)
    
    # Sample true parameters from prior
    theta = prior.sample(theta_key, 1)[0]
    
    # Generate observations using the model's likelihood
    xs = likelihood.sample(noise_key, opt_xi, theta)
    
    all_opt_xi.append(opt_xi)
    all_xs.append(xs)

all_opt_xi = jnp.array(all_opt_xi)
all_xs = jnp.array(all_xs)

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [2, 1, 1]})

# Main scatter plot
ax1.plot(range_xi, h(range_xi), color='k', linewidth=3, label='Example function h')
ax1.scatter(
    all_opt_xi[:, 0], all_xs[:, 0], color='blue', label=r"$\xi_1$",
    zorder=5, s=10, edgecolor='black', linewidth=1.5, alpha=0.85, marker='o'
)
ax1.scatter(
    all_opt_xi[:, 1], all_xs[:, 1], color='crimson', label=r"$\xi_2$",
    zorder=5, s=10, edgecolor='black', linewidth=1.5, alpha=0.85, marker='s'
)
ax1.set_xlabel(r"$\xi$", fontsize=24)
ax1.set_ylabel(r"$x$", fontsize=24)
ax1.set_title(f"Optimal Designs (α={order})", fontsize=28, pad=20)
ax1.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
ax1.set_xlim(-1.1, 1.1)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.legend(fontsize=18, framealpha=0.9)

# Histogram for first dimension
ax2.hist(all_opt_xi[:, 0], bins=200, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.5)
ax2.set_xlabel(r"$\xi_1$", fontsize=24)
ax2.set_ylabel("Frequency", fontsize=24)
ax2.set_title(r"Distribution over $\xi_1$", fontsize=26, pad=20)
ax2.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
ax2.set_xlim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.axvline(optimal_design[0], color='red', linestyle='--', linewidth=2, label='Mode')
ax2.legend(fontsize=16)

# Histogram for second dimension
ax3.hist(all_opt_xi[:, 1], bins=200, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=1.5)
ax3.set_xlabel(r"$\xi_2$", fontsize=24)
ax3.set_ylabel("Frequency", fontsize=24)
ax3.set_title(r"Distribution over $\xi_2$", fontsize=26, pad=20)
ax3.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
ax3.set_xlim(-1, 1)
ax3.tick_params(axis='both', which='major', labelsize=18)
ax3.axvline(optimal_design[1], color='red', linestyle='--', linewidth=2, label='Mode')
ax3.legend(fontsize=16)

plt.tight_layout()
plt.show()

fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(16, 6))

# Create grid for 2D designs
xi_range = jnp.linspace(-1, 1, 50)
XI1, XI2 = jnp.meshgrid(xi_range, xi_range)

# Compute MI for each design
MI_sibson = jnp.zeros_like(XI1)
MI_shannon = jnp.zeros_like(XI1)

for i in range(len(xi_range)):
    for j in range(len(xi_range)):
        design = jnp.array([XI1[i, j], XI2[i, j]])
        MI_sibson = MI_sibson.at[i, j].set(
            model.sibson_alpha_mutual_information(design, alpha=order)
        )
        MI_shannon = MI_shannon.at[i, j].set(
            model.shannon_mutual_information(design)
        )

# Plot Sibson's α-MI
contour1 = ax4.contourf(XI1, XI2, MI_sibson, levels=20, cmap='viridis')
ax4.set_xlabel(r"$\xi_1$", fontsize=20)
ax4.set_ylabel(r"$\xi_2$", fontsize=20)
ax4.set_title(f"Sibson's α-MI (α={order})", fontsize=22)
ax4.grid(True, linestyle='--', alpha=0.3)
plt.colorbar(contour1, ax=ax4, label='MI')

# Plot Shannon's MI
contour2 = ax5.contourf(XI1, XI2, MI_shannon, levels=20, cmap='viridis')
ax5.set_xlabel(r"$\xi_1$", fontsize=20)
ax5.set_ylabel(r"$\xi_2$", fontsize=20)
ax5.set_title("Shannon's MI (α=1)", fontsize=22)
ax5.grid(True, linestyle='--', alpha=0.3)
plt.colorbar(contour2, ax=ax5, label='MI')

plt.tight_layout()
plt.show()

print()
print("=" * 80)
print("DONE!")
print("=" * 80)