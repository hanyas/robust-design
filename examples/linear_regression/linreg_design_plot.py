"""
Linear Gaussian Regression Experiment
"""

import jax
from jax import Array
from jax import numpy as jnp
from jax import random as jr

import numpy as np
import matplotlib.pyplot as plt
import csv

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian
from rdesign.linear_regression import LinearRegression

jax.config.update("jax_enable_x64", True)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
        "axes.unicode_minus": False,
        # Font sizes
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        # Line and marker sizes
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        # Figure size for better visibility
        "figure.figsize": (12, 5),
    }
)

prior_mean = jnp.array([1.0, 1.0])
prior_cov = jnp.array([[3.0, 2.0], [2.0, 2.0]])

key = jr.PRNGKey(0)
prior = Gaussian(mean=prior_mean, cov=prior_cov)
lklhd = LinearGaussian(noise_stddev=jnp.sqrt(1e-1))
model = LinearRegression(prior=prior, likelihood=lklhd)

# Define specific alphas to visualize surfaces
alphas = [0.01, 0.1, 0.995]

# High resolution grid for computation
n_grid = 40
x_range = jnp.linspace(-1.0, 1.0, n_grid)
y_range = jnp.linspace(-1.0, 1.0, n_grid)
X, Y = jnp.meshgrid(x_range, y_range)
grid_flat = jnp.stack([X.ravel(), Y.ravel()], axis=1)

print(f"Computing data for {len(alphas)} alpha values...")

# Prepare data storage
mi_results = []

for alpha in alphas:
    mi_vals = jax.vmap(lambda d: model.sibson_mi(d, float(alpha)))(grid_flat)
    mi_results.append(np.array(mi_vals))

mi_matrix = np.array(mi_results)  # (num_alphas, num_points)

# Write plotted data to CSV
csv_path = "examples/linear_regression/data/linreg_design.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = ["xi_1", "xi_2"]
    header += [f"mi_alpha_{alpha:.4f}" for alpha in alphas]
    writer.writerow(header)

    for idx, design in enumerate(grid_flat.tolist()):
        row = design + mi_matrix[:, idx].tolist()
        writer.writerow(row)
        if (idx + 1) % n_grid == 0 and (idx + 1) != len(grid_flat):
            csvfile.write("\n")

print(f"Saved CSV data to '{csv_path}'")

# Plotting
fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4.5))
axes = axes.ravel()

print(f"Generating labeled contour plots...")

for i, alpha in enumerate(alphas):
    ax = axes[i]

    # Extract data for plotting
    mi_vals = mi_matrix[i]
    mi_grid = mi_vals.reshape(n_grid, n_grid)

    # Smooth heatmap background
    ax.imshow(mi_grid, origin="lower", extent=[-1, 1, -1, 1], cmap="gray", interpolation="bicubic", alpha=0.3)

    # Contour lines (dashed)
    cl = ax.contour(
        X,
        Y,
        mi_grid,
        levels=8,
        colors="black",
        linewidths=1.0,
        alpha=0.8,
    )
    ax.clabel(cl, cl.levels[::2], inline=True, fontsize=9, fmt="%.2f")

    ax.set_title(rf"$\alpha = {alpha:.2f}$")
    ax.set_xlabel(r"$\xi_1$")
    ax.set_ylabel(r"$\xi_2$")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.3)

plt.tight_layout()
plt.show()
