"""
Consistency check for power-likelihood Gibbs-EIG in linear regression.

For the conjugate linear-Gaussian model, the power-likelihood Gibbs EIG is an
exact design-independent rescaling of Sibson's alpha-MI at the same order.
This script verifies that relationship on a two-point design grid.
"""

import csv

import jax
from jax import Array
import jax.numpy as jnp
import matplotlib.pyplot as plt

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian, LinearRegression

jax.config.update("jax_enable_x64", True)


def normalize_surface(values: Array) -> Array:
    """Normalize values to [0, 1] for shape-only comparisons."""
    shifted = values - jnp.min(values)
    scale = jnp.max(shifted)
    return jnp.where(scale > 0.0, shifted / scale, shifted)


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    omega = 5e-1
    n_grid = 40

    prior_mean = jnp.array([1.0, 1.0])
    prior_cov = jnp.array([
        [3.0, 2.0],
        [2.0, 2.0],
    ])

    prior = Gaussian(mean=prior_mean, cov=prior_cov)
    likelihood = LinearGaussian(noise_stddev=jnp.sqrt(1e-1))
    model = LinearRegression(prior=prior, likelihood=likelihood)

    x_range = jnp.linspace(-1.0, 1.0, n_grid)
    y_range = jnp.linspace(-1.0, 1.0, n_grid)
    X, Y = jnp.meshgrid(x_range, y_range)
    grid_flat = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    # =========================================================================
    # OBJECTIVE SURFACES
    # =========================================================================
    print("=" * 88)
    print("POWER-GEIG CONSISTENCY CHECK: LINEAR REGRESSION")
    print("=" * 88)
    print(f"Power level ω:                    {omega:.2f}")
    print(f"Grid size:                        {n_grid} x {n_grid}")
    print(f"Noise variance:                   {likelihood.noise_stddev**2:.4f}")
    print()
    print("Computing objective surfaces...")

    sibson_vals = jax.vmap(lambda d: model.sibson_mi(d, omega))(grid_flat)
    power_geig_vals = jax.vmap(lambda d: model.power_geig(d, omega))(grid_flat)

    sibson_norm = normalize_surface(sibson_vals)
    power_geig_norm = normalize_surface(power_geig_vals)
    norm_gap = jnp.abs(sibson_norm - power_geig_norm)

    nonzero_mask = sibson_vals > 1e-12
    scale_estimates = power_geig_vals[nonzero_mask] / sibson_vals[nonzero_mask]
    scale_factor = float(jnp.mean(scale_estimates))
    max_residual = float(jnp.max(jnp.abs(power_geig_vals - scale_factor * sibson_vals)))
    max_norm_gap = float(jnp.max(norm_gap))

    design_sibson = grid_flat[int(jnp.argmax(sibson_vals))]
    design_power_geig = grid_flat[int(jnp.argmax(power_geig_vals))]

    print()
    print("Optimal designs")
    print(f"  Sibson optimum:                 ({design_sibson[0]: .3f}, {design_sibson[1]: .3f})")
    print(f"  Power-GEIG optimum:             ({design_power_geig[0]: .3f}, {design_power_geig[1]: .3f})")
    print(f"  Estimated proportionality:      {scale_factor:.6f}")
    print(f"  Max |GEIG - c * Sibson|:        {max_residual:.6e}")
    print(f"  Max normalized surface gap:     {max_norm_gap:.6e}")

    # =========================================================================
    # SAVE GRID DATA
    # =========================================================================
    csv_path = "examples/linear_regression/data/linreg_geig_consistency.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "xi_1",
            "xi_2",
            "sibson",
            "power_geig",
            "sibson_norm",
            "power_geig_norm",
            "abs_norm_gap",
        ])
        for idx, design in enumerate(grid_flat.tolist()):
            writer.writerow([
                float(design[0]),
                float(design[1]),
                float(sibson_vals[idx]),
                float(power_geig_vals[idx]),
                float(sibson_norm[idx]),
                float(power_geig_norm[idx]),
                float(norm_gap[idx]),
            ])

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    sibson_grid = sibson_norm.reshape(n_grid, n_grid)
    power_geig_grid = power_geig_norm.reshape(n_grid, n_grid)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    im0 = axes[0].imshow(
        sibson_grid,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap="Greys",
        interpolation="bicubic",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].scatter(
        float(design_sibson[0]),
        float(design_sibson[1]),
        color="#B04A1F",
        s=35,
    )
    axes[0].set_title(f"Sibson (normalized), α={omega:.2f}")
    axes[0].set_xlabel(r"$\xi_1$")
    axes[0].set_ylabel(r"$\xi_2$")

    im1 = axes[1].imshow(
        power_geig_grid,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap="Greys",
        interpolation="bicubic",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].scatter(
        float(design_power_geig[0]),
        float(design_power_geig[1]),
        color="#B04A1F",
        s=35,
    )
    axes[1].set_title(f"Power-GEIG (normalized), ω={omega:.2f}")
    axes[1].set_xlabel(r"$\xi_1$")
    axes[1].set_ylabel(r"$\xi_2$")

    axes[2].scatter(sibson_vals, power_geig_vals, s=10, alpha=0.6, color="#111111")
    x_min = float(jnp.min(sibson_vals))
    x_max = float(jnp.max(sibson_vals))
    axes[2].plot(
        [x_min, x_max],
        [scale_factor * x_min, scale_factor * x_max],
        color="#B04A1F",
        lw=2.0,
    )
    axes[2].set_title("Exact proportionality")
    axes[2].set_xlabel("Sibson")
    axes[2].set_ylabel("Power-GEIG")
    axes[2].text(
        0.03,
        0.95,
        f"c = {scale_factor:.4f}\nmax residual = {max_residual:.1e}",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.7", "boxstyle": "round,pad=0.3"},
    )

    fig.colorbar(im0, ax=axes[:2], shrink=0.85, label="Normalized objective")
    for ax in axes:
        ax.grid(False)

    plt.tight_layout()
    plt.show()
