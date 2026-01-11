import jax.numpy as jnp

import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from rdesign.ab_testing import ABTesting, MaskedLinearGaussian
from rdesign.distributions import Gaussian

# Configuration
num_participants = 25
prior_stds = jnp.array([10.0, 1.0])  # Moderate asymmetry: 2x difference  
obs_stds = jnp.array([1.0, 1.0])  # Equal noise

# Create the ABTesting model manually
prior = Gaussian(
    mean=jnp.zeros(2),
    cov=jnp.diag(prior_stds ** 2)
)

likelihood = MaskedLinearGaussian(
    num_participants=num_participants,
    observation_std_devs=obs_stds
)

model = ABTesting(prior=prior, likelihood=likelihood)

# Alpha values: logarithmic space from 1e-2 to 1.0, 10 values
alpha_values = jnp.logspace(-2, 0, 10)

# Compute MI for all designs and all alpha values
designs = jnp.arange(num_participants + 1)

# Create a matrix: rows = alpha values, cols = designs
mi_matrix = jnp.zeros((len(alpha_values), len(designs)))
optimal_designs = []

for i, alpha in enumerate(alpha_values):
    for j, design in enumerate(designs):
        sibson_mi = model.sibson_mi(design, alpha)
        mi_matrix = mi_matrix.at[i, j].set(sibson_mi)
    optimal_designs.append(int(jnp.argmax(mi_matrix[i, :])))

# Create figure with 10 subplots (2 rows x 5 columns)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

# Plot each alpha value in its own subplot
for i, alpha in enumerate(alpha_values):
    ax = axes[i]
    
    # All bars in light gray
    ax.bar(designs, mi_matrix[i, :], width=0.8, color='#CCCCCC', edgecolor='none')
    
    # Highlight optimal design in dark gray
    opt_design = optimal_designs[i]
    ax.bar(opt_design, mi_matrix[i, opt_design], width=0.8, color='#404040', edgecolor='none')
    
    # Minimal styling - clean academic look
    ax.set_xlabel('$n_A$', fontsize=11)
    ax.set_ylabel("$I_{\\alpha}$", fontsize=11)
    ax.set_title(f'$\\alpha = {alpha:.2f}$', fontsize=11, pad=8)
    
    # Clean axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Minimal ticks
    ax.tick_params(axis='both', which='major', labelsize=9, length=3, width=0.8)
    
    # Set x-axis limits with some padding
    ax.set_xlim(-1, num_participants + 1)
    
    # Remove grid
    ax.grid(False)

plt.tight_layout()
plt.show()

# Print summary
print("=" * 80)
print("Sibson's Mutual Information Analysis Across Alpha Values")
print("=" * 80)
print(f"\nAlpha range: {alpha_values[0]:.4f} to {alpha_values[-1]:.4f} (log-spaced, {len(alpha_values)} values)")
print(f"\nOptimal designs for each alpha:")
print("-" * 80)
print(f"{'Alpha':<12} {'Optimal n_A':<15} {'Max MI':<15}")
print("-" * 80)
for i, alpha in enumerate(alpha_values):
    opt_design = optimal_designs[i]
    max_mi = float(mi_matrix[i, opt_design])
    print(f"{alpha:<12.4f} {opt_design:<15} {max_mi:<15.6f}")
print("=" * 80)
