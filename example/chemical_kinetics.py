"""
Chemical Kinetics Experiment (Michaelis-Menten and Hill Models)

This example implements Bayesian experimental design for a chemical kinetics model:
    - Michaelis-Menten: mean response eta(theta, xi) = Vmax * xi / (xi + V0.5)
    - Hill (misspecified/true): eta_s(theta, xi) = Vmax * xi**s / (xi**s + V0.5**s)

The experiment estimates the parameters theta = (Vmax, V0.5) from noisy measurements at substrate concentrations xi.

"""
import jax
from jax import Array, random, numpy as jnp, scipy as jsc

from rdesign.base import Prior, Model
from rdesign.utils import info_gain

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Qt5Agg")


class UniformPrior(Prior):
    """Uniform prior over (Vmax, V0.5)."""
    def __init__(self, bounds: tuple):
        self.bounds = bounds  # (min, max) for both Vmax and V0.5

    def sample(self, key: Array, num_samples: int) -> Array:
        min_val, max_val = self.bounds
        return random.uniform(key, shape=(num_samples, 2), minval=min_val, maxval=max_val)

    def logpdf(self, x: Array) -> Array:
        min_val, max_val = self.bounds
        range_val = max_val - min_val
        log_probs = jsc.stats.uniform.logpdf(x, loc=min_val, scale=range_val)
        return jnp.sum(log_probs, axis=-1)


class MichaelisMentenModel(Model):
    """Michaelis-Menten model: eta(theta, xi) = Vmax * xi / (xi + V0.5)"""
    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std

    def mean(self, design: Array, theta: Array) -> Array:
        Vmax, V05 = theta[..., 0], theta[..., 1]
        return Vmax * design / (design + V05)

    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        mean = self.mean(design, theta)
        noise = random.normal(key, shape=mean.shape) * self.noise_std
        return mean + noise

    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        mean = self.mean(design, theta)
        log_probs = jsc.stats.norm.logpdf(x, loc=mean, scale=self.noise_std)
        return jnp.sum(log_probs)


class HillModel(Model):
    """Hill model: eta_s(theta, xi) = Vmax * xi**s / (xi**s + V0.5**s)"""
    def __init__(self, s: float = 2.0, noise_std: float = 0.1):
        self.s = s
        self.noise_std = noise_std

    def mean(self, design: Array, theta: Array) -> Array:
        Vmax, V05 = theta[..., 0], theta[..., 1]
        return Vmax * (design ** self.s) / (design ** self.s + V05 ** self.s)

    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        mean = self.mean(design, theta)
        noise = random.normal(key, shape=mean.shape) * self.noise_std
        return mean + noise

    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        mean = self.mean(design, theta)
        log_probs = jsc.stats.norm.logpdf(x, loc=mean, scale=self.noise_std)
        return jnp.sum(log_probs)


def create_kinetics_experiment(
    bounds: tuple = (20.0, 200.0),
    noise_std: float = 0.1,
    s: float = 2.0,
    model_type: str = "michaelis-menten"
) -> tuple[Prior, Model]:
    prior = UniformPrior(bounds)
    if model_type == "michaelis-menten":
        model = MichaelisMentenModel(noise_std=noise_std)
    elif model_type == "hill":
        model = HillModel(s=s, noise_std=noise_std)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return prior, model


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    key = random.PRNGKey(0)
    bounds = (20.0, 200.0)
    noise_std = 5.0
    s = 2.0

    # Main model (used for design)
    prior, model = create_kinetics_experiment(bounds=bounds, noise_std=noise_std, model_type="michaelis-menten")
    # Misspecified/true model (Hill)
    _, true_model = create_kinetics_experiment(bounds=bounds, noise_std=noise_std, s=s, model_type="hill")

    alphas = jnp.array([0.05, 0.95])

    # 1D design grid (single substrate concentration)
    n_grid = 100
    designs = jnp.linspace(0.0, 400.0, n_grid)
    info_gain_vals = jnp.zeros((len(alphas), n_grid))

    num_outer = int(1 * 512)
    num_inner = int(1 * 512)

    for i, alpha in enumerate(alphas):
        print(f"Evaluating alpha = {alpha:.2f}...")
        for j, design in enumerate(designs):
            key, sub_key = random.split(key)
            design_arr = jnp.array([design]).reshape(1, 1)
            info_gain_vals = info_gain_vals.at[i, j].set(
                info_gain(
                    key=sub_key,
                    design=design_arr,
                    prior=prior,
                    model=model,
                    num_outer=num_outer,
                    num_inner=num_inner,
                    alpha=float(alpha)
                )
            )

    # Plot 1D info gain
    fig, axes = plt.subplots(len(alphas), 1, figsize=(8, 6))
    for i, (ax, alpha) in enumerate(zip(axes, alphas)):
        ax.plot(designs, info_gain_vals[i], 'b-', linewidth=2)
        ax.set_ylabel(f'Info gain (alpha={alpha:.2f})')
        ax.set_xlabel('Substrate concentration (xi)')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2D design grid (pair of substrate concentrations)
    print("\nCreating surface plots for paired designs...")
    nx = ny = 40
    x = jnp.linspace(0.0, 400.0, nx)
    y = jnp.linspace(0.0, 400.0, ny)
    X, Y = jnp.meshgrid(x, y)
    info_gain_vals_2d = jnp.zeros((len(alphas), nx, ny))
    key = random.PRNGKey(1)

    for i, alpha in enumerate(alphas):
        print(f"Evaluating alpha = {alpha:.2f} (2D grid)...")
        for j in range(nx):
            for k in range(ny):
                key, sub_key = random.split(key)
                design = jnp.array([X[j, k], Y[j, k]]).reshape(2, 1)
                info_gain_vals_2d = info_gain_vals_2d.at[i, j, k].set(
                    info_gain(
                        key=sub_key,
                        design=design,
                        prior=prior,
                        model=model,
                        num_outer=num_outer,
                        num_inner=num_inner,
                        alpha=float(alpha)
                    )
                )

    fig, axes = plt.subplots(len(alphas), 1, figsize=(8, 10))
    for i, (alpha, ax) in enumerate(zip(alphas, axes)):
        im = ax.imshow(
            info_gain_vals_2d[i],
            origin='lower',
            extent=[0, 400, 0, 400],
            aspect='auto',
            cmap='viridis',
        )
        ax.set_xlabel('Design 1 (xi)')
        ax.set_ylabel('Design 2 (xi)')
        ax.set_title(f'alpha = {alpha:.2f}')
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    # 3D surface plots
    fig = plt.figure(figsize=(15, 8))
    axes = []
    for i in range(len(alphas)):
        ax = fig.add_subplot(1, len(alphas), i + 1, projection='3d')
        axes.append(ax)
    for i, (alpha, ax) in enumerate(zip(alphas, axes)):
        surf = ax.plot_surface(
            X, Y, info_gain_vals_2d[i],
            cmap='viridis',
            edgecolor='none',
            alpha=0.8
        )
        ax.set_xlabel('Design 1 (xi)')
        ax.set_ylabel('Design 2 (xi)')
        ax.set_title(f'alpha = {alpha:.2f}')
    plt.tight_layout()
    plt.show() 