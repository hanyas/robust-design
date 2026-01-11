"""
Linear Gaussian Regression Experiment

A generic implementation for Bayesian linear regression with a 2D parameter vector.
The experiment uses a Gaussian prior over the regression parameters and a Gaussian likelihood
with linear structure in the parameters. The model is:
    theta ~ N(mu, Sigma)
    x | theta, xi ~ N(theta_0 + theta_1 * xi, sigma^2)
where xi is the design, and x is the observed output.
"""
from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr

from jax.scipy import stats

from rdesign.utils import sibson_alpha_mutual_information
from rdesign.distributions import Gaussian


class HeteroLinearGaussian(NamedTuple):
    """
    Heteroscedastic Gaussian linear likelihood for regression.
    
    Args:
        base_noise_std: Base noise standard deviation
        hetero_scale: Scale factor for heteroscedasticity
    """
    base_noise_std: float
    hetero_scale: float = 0.01
    
    def design_matrix(self, design: Array) -> Array:
        """Construct augmented design matrix H(ξ)."""
        design = jnp.atleast_1d(design)
        size = design.shape[0]
        return jnp.column_stack([design, jnp.ones(size)])
    
    def _compute_mean(self, design_point: float, theta: Array) -> float:
        """Compute mean for a single design point: θ₀ + θ₁ ξ"""
        return theta[0] * design_point + theta[1]
    
    def _compute_stddev(self, design_point: float, theta: Array) -> float:
        """
        Compute heteroscedastic standard deviation.
        """
        mean_pred = self._compute_mean(design_point, theta)
        return self.base_noise_std + self.hetero_scale * jnp.abs(mean_pred)
    
    def mean(self, design: Array, theta: Array) -> Array:
        """
        Compute mean vector for all design points.
        
        Args:
            design: Design points, shape (N,)
            theta: Parameters [slope, intercept], shape (2,)
            
        Returns:
            Mean vector, shape (N,)
        """
        H = self.design_matrix(design)
        return H @ theta
    
    def stddev(self, design: Array, theta: Array) -> Array:
        """
        Compute heteroscedastic standard deviation for each design point.
        
        Args:
            design: Design points, shape (N,)
            theta: Parameters, shape (2,)
            
        Returns:
            Standard deviations, shape (N,)
        """
        design = jnp.atleast_1d(design)
        return jax.vmap(self._compute_stddev, in_axes=(0, None))(design, theta)
    
    def cov(self, design: Array, theta: Array) -> Array:
        """
        Compute heteroscedastic covariance matrix.
        
        Args:
            design: Design points, shape (N,)
            theta: Parameters, shape (2,)
            
        Returns:
            Diagonal covariance matrix, shape (N, N)
        """
        std = self.stddev(design, theta)
        return jnp.diag(std ** 2)
    
    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        """
        Sample from heteroscedastic likelihood p(x | θ, ξ).
        
        Args:
            key: Random key
            design: Design points, shape (N,)
            theta: Parameters, shape (2,)
            
        Returns:
            Observations, shape (N,)
        """
        mean = self.mean(design, theta)
        cov = self.cov(design, theta)
        return jr.multivariate_normal(key, mean, cov)
    
    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """
        Compute log p(x | θ, ξ).
        
        Args:
            x: Observations, shape (N,)
            design: Design points, shape (N,)
            theta: Parameters, shape (2,)
            
        Returns:
            Log likelihood (scalar)
        """
        mean = self.mean(design, theta)
        cov = self.cov(design, theta)
        return stats.multivariate_normal.logpdf(x, mean, cov)



# Example usage
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_disable_jit", True)

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg")

    plt.rcParams.update({
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
        "figure.figsize": (12, 8)
    })

    print("=" * 80)
    print("NESTED MONTE CARLO SIBSON α-MI COMPUTATION")
    print("=" * 80)
    
    # Setup
    prior_mean = jnp.array([0., 1.])
    prior_cov = jnp.array([
        [2.0, 0.75],
        [0.75, 1.0]
    ])
    
    prior = Gaussian(mean=prior_mean, cov=prior_cov)
    lklhd = HeteroLinearGaussian(base_noise_std=jnp.sqrt(1e-1), hetero_scale=0.01)
    
    print(f"\nPrior: N({prior_mean}, Σ)")
    print(f"Likelihood: Heteroscedastic (σ_base={lklhd.base_noise_std:.3f})")
    
    # Define alphas and 2D design grid
    alphas = jnp.array([0.01, 0.99])
    
    n_grid = 100
    x = jnp.linspace(-1.0, 1.0, n_grid)
    y = jnp.linspace(-1.0, 1.0, n_grid)
    X, Y = jnp.meshgrid(x, y)
    
    # Monte Carlo parameters
    num_outer = int(1 * 512)
    num_inner = int(1 * 512)
    
    print(f"\nGrid: {n_grid} × {n_grid} = {n_grid**2} design points")
    print(f"Monte Carlo: {num_outer} outer × {num_inner} inner samples")
    print(f"Alphas: {alphas}")
    
    # Flatten the grid for vectorized computation
    designs = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Initialize
    info_gain_vals = jnp.zeros((len(alphas), n_grid, n_grid))
    key = jr.PRNGKey(0)
    
    # Batch processing parameters
    batch_size = 100  # Adjust based on memory
    
    print(f"\nProcessing in batches of {batch_size}...")
    
    for i, alpha in enumerate(alphas):
        print(f"\n{'=' * 80}")
        print(f"Evaluating α = {alpha:.2f}...")
        print(f"{'=' * 80}")
        
        info_gains_flat = jnp.zeros(len(designs))
        
        # Process designs in batches
        num_batches = (len(designs) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(designs))
            batch_designs = designs[batch_start:batch_end]
            
            # Generate keys for this batch
            key, sub_key = jr.split(key)
            batch_keys = jr.split(sub_key, len(batch_designs))
            
            # Vectorize over batch
            batch_gains = jax.vmap(
                lambda k, d: sibson_alpha_mutual_information(
                    k, d, prior, lklhd, num_outer, num_inner, float(alpha)
                )
            )(batch_keys, batch_designs)
            
            info_gains_flat = info_gains_flat.at[batch_start:batch_end].set(batch_gains)
            
            # Progress indicator
            if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
                progress = 100 * (batch_idx + 1) / num_batches
                print(f"  Batch {batch_idx + 1}/{num_batches} ({progress:.1f}%) complete")
        
        # Reshape back to grid
        info_gain_vals = info_gain_vals.at[i].set(info_gains_flat.reshape(n_grid, n_grid))
            
    # ========================================================================
    # Visualization
    # ========================================================================
    
    print(f"\n{'=' * 80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'=' * 80}")
    
    # 2D heatmaps
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4))
    axes = axes.ravel()
    
    for i, (alpha, ax) in enumerate(zip(alphas, axes)):
        im = ax.imshow(
            info_gain_vals[i],
            origin='lower',
            extent=[-1, 1, -1, 1],
            aspect='auto',
            cmap='viridis'
        )
        ax.set_title(fr'$\alpha = {alpha:.2f}$', fontsize=16)
        ax.set_xlabel(r'$\xi_1$', fontsize=14)
        ax.set_ylabel(r'$\xi_2$', fontsize=14)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$I^S_\alpha(\theta; x \mid \xi)$', fontsize=12)
    
    plt.tight_layout()
    plt.show()

    # 3D surface plots
    fig = plt.figure(figsize=(15, 6))
    axes = []
    for i in range(len(alphas)):
        ax = fig.add_subplot(1, len(alphas), i + 1, projection='3d')
        axes.append(ax)
    
    for i, (alpha, ax) in enumerate(zip(alphas, axes)):
        surf = ax.plot_surface(
            X, Y, info_gain_vals[i],
            cmap='viridis',
            edgecolor='none',
            alpha=0.8
        )
        ax.set_xlabel(r'$\xi_1$', fontsize=12)
        ax.set_ylabel(r'$\xi_2$', fontsize=12)
        ax.set_zlabel(r'$I^S_\alpha$', fontsize=12)
        ax.set_title(fr'$\alpha = {alpha:.2f}$', fontsize=14)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    print(f"\n{'=' * 80}")
    print("DONE!")
    print(f"{'=' * 80}")
    
    plt.show()