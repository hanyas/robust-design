"""
Linear Gaussian Regression Experiment

A generic implementation for Bayesian linear regression with a 2D parameter vector.
The experiment uses a Gaussian prior over the regression parameters and a Gaussian likelihood
with linear structure in the parameters. The model is:
    theta ~ N(mu, Sigma)
    x | theta, xi ~ N(theta_0 + theta_1 * xi, sigma^2(xi, theta))
where xi is the design (input), and x is the observed output.
"""
import jax
from jax import Array, random, numpy as jnp
from jax.scipy import stats

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian
from rdesign.linear_regression import LinearRegression

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

prior_mean = jnp.array([1., 2.])
prior_cov = jnp.array([
    [3.0, 1.5],
    [1.5, 2.0]
])

key = random.PRNGKey(0)
prior = Gaussian(mean=prior_mean, cov=prior_cov)
model = LinearGaussian(noise_stddev=jnp.sqrt(5e-1))
linreg = LinearRegression(prior=prior, likelihood=model)

# Define alphas and 2D design grid
alphas = jnp.array([0.01, 0.1, 0.995])

n_grid = 40
x = jnp.linspace(-1.0, 1.0, n_grid)
y = jnp.linspace(-1.0, 1.0, n_grid)
X, Y = jnp.meshgrid(x, y)

info_gain_vals = jnp.zeros((len(alphas), n_grid, n_grid))

for i, alpha in enumerate(alphas):
    print(f"Evaluating alpha = {alpha:.2f}...")
    for j in range(n_grid):
        for k in range(n_grid):
            key, sub_key = random.split(key)
            design = jnp.array([X[j, k], Y[j, k]])
            info_gain_vals = info_gain_vals.at[i, j, k].set(
                linreg.sibson_mi(
                    design=design,
                    alpha=float(alpha)
                )
            )

# Plotting
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
    ax.set_title(f'alpha = {alpha:.2f}')
    ax.set_xlabel('Design 1')
    ax.set_ylabel('Design 2')

plt.tight_layout()
plt.show()
# fig = plt.figure(figsize=(15, 12))
# axes = []
# for i in range(len(alphas)):
#     ax = fig.add_subplot(1, 2, i + 1, projection='3d')
#     axes.append(ax)
#
# for i, (alpha, ax) in enumerate(zip(alphas, axes)):
#     surf = ax.plot_surface(
#         X, Y, info_gain_vals[i],
#         cmap='viridis',
#         edgecolor='none',
#         alpha=0.8
#     )
#     ax.set_xlabel('Design 1')
#     ax.set_ylabel('Design 2')
#     ax.set_title(f'alpha = {alpha:.2f}')
#
# plt.tight_layout()
# plt.show()
