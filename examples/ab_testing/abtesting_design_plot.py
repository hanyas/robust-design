"""
Sibson's Mutual Information Analysis for A/B Testing

This script visualizes how the optimal experimental design (allocation of
participants between groups A and B) changes as the robustness parameter α varies.
"""

import jax

import jax.numpy as jnp
import matplotlib.pyplot as plt
import csv

from rdesign.ab_testing import ABTesting, Beta, Binomial

jax.config.update("jax_enable_x64", True)


# Configuration
num_participants = 25

# Beta prior parameters
prior_delta = jnp.array([1.0, 1.0])
prior_gamma = jnp.array([3.0, 5.0])

# Create the ABTesting model
prior = Beta(delta=prior_delta, gamma=prior_gamma)
model = Binomial(num_participants=num_participants)
abtest = ABTesting(prior=prior, likelihood=model)

print("A/B Testing Model Configuration")
print("=" * 60)
print(f"Group A prior: Beta({prior_delta[0]:.0f}, {prior_gamma[0]:.0f})")
print(f"  Mean: {prior.mean[0]:.2f}, Concentration: {prior.concentration[0]:.0f}")
print(f"Group B prior: Beta({prior_delta[1]:.0f}, {prior_gamma[1]:.0f})")
print(f"  Mean: {prior.mean[1]:.2f}, Concentration: {prior.concentration[1]:.0f}")
print(f"Total participants: {num_participants}")
print()

# Alpha values: logarithmic space from 0.01 to 1.0
alpha_values = jnp.logspace(-2, 0, 3)

# Compute MI for all designs and all alpha values
designs = jnp.arange(num_participants + 1)

# Compute MI for all designs and all alpha values
print("Computing Sibson MI for each design and alpha value...")
vmap_sibson = jax.vmap(jax.vmap(abtest.sibson_mi, in_axes=(0, None)), in_axes=(None, 0))
mi_matrix = vmap_sibson(designs, alpha_values)
optimal_designs = jnp.argmax(mi_matrix, axis=1)

for i, alpha in enumerate(alpha_values):
    print(f"  α = {alpha:.4f}... optimal n_A = {optimal_designs[i]}")

# Write plotted data to CSV (one row per design)
csv_path = "examples/ab_testing/abtesting_design_plot.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = ["n_A"]
    header += [f"mi_alpha_{alpha:.4f}" for alpha in alpha_values.tolist()]
    header += [f"max_alpha_{alpha:.4f}" for alpha in alpha_values.tolist()]
    writer.writerow(header)
    for idx, n_a in enumerate(designs.tolist()):
        max_values = []
        for alpha_idx in range(len(alpha_values)):
            if idx == int(optimal_designs[alpha_idx]):
                max_values.append(float(mi_matrix[alpha_idx, idx]))
            else:
                max_values.append(0.0)
        row = [n_a] + mi_matrix[:, idx].tolist() + max_values
        writer.writerow(row)

# Create figure with 10 subplots (2 rows x 5 columns)
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
axes = axes.flatten()

# Plot each alpha value in its own subplot
for i, alpha in enumerate(alpha_values):
    ax = axes[i]

    # All bars in light gray
    ax.bar(designs, mi_matrix[i, :], width=0.8, color="#CCCCCC", edgecolor="none")

    # Highlight optimal design in dark gray
    opt_design = optimal_designs[i]
    ax.bar(
        opt_design,
        mi_matrix[i, opt_design],
        width=0.8,
        color="#404040",
        edgecolor="none",
    )

    # Minimal styling - clean academic look
    ax.set_xlabel("$n_A$", fontsize=11)
    ax.set_ylabel("$I^S_{\\alpha}$", fontsize=11)
    ax.set_title(f"$\\alpha = {alpha:.2f}$", fontsize=11, pad=8)

    # Clean axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Minimal ticks
    ax.tick_params(axis="both", which="major", labelsize=9, length=3, width=0.8)

    # Set x-axis limits with some padding
    ax.set_xlim(-1, num_participants + 1)

    # Set y-axis to start at 0 so bars sit on x-axis
    ax.set_ylim(bottom=0)

    # Remove grid
    ax.grid(False)

plt.tight_layout()
plt.show()
