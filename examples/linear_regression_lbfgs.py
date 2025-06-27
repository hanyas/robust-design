import numpy as np

import jax
import jax.numpy as jnp

from jaxopt import LBFGSB


@jax.jit
def sibsons_estimator(design, prior_cov, lklhd_cov, order):
    """
    Compute the Sibson mutual information estimator.

    Args:
        design: Array of shape (N,) containing the design points
        prior_cov: Prior covariance matrix of shape (d, d)
        lklhd_cov: Likelihood covariance matrix of shape (1, 1)
        order: Order parameter for the Sibson estimator

    Returns:
        The Sibson mutual information estimate
    """
    N = design.shape[0]
    A = jnp.concatenate([
        jnp.ones((N, 1)),
        jnp.expand_dims(design, axis=1)
    ], axis=1)
    Lambda = A @ prior_cov @ A.T + (lklhd_cov / order) * jnp.eye(N)
    _, logdet = jnp.linalg.slogdet(Lambda)
    return 0.5 * logdet


@jax.jit
def objective(x, prior_cov, lklhd_cov, order):
    """
    Objective function to minimize (negative Sibson estimator).

    Args:
        x: Flattened array of design points
        prior_cov: Prior covariance matrix
        lklhd_cov: Likelihood covariance matrix
        order: Order parameter

    Returns:
        Negative Sibson mutual information estimate
    """
    return - 1.0 * sibsons_estimator(x, prior_cov, lklhd_cov, order)


def optimize_sibson(dim, prior_cov, lklhd_cov, order):
    """
    Optimize the Sibson estimator using JAXopt's LBFGS-B.

    Args:
        dim: Number of design points
        prior_cov: Prior covariance matrix
        lklhd_cov: Likelihood covariance matrix
        order: Order parameter

    Returns:
        Optimized design points and optimization result
    """
    # Initial guess: random points between -1 and 1
    x0 = jnp.array(np.random.uniform(-1, 1, dim))
    bounds = (jnp.full(dim, -1.0), jnp.full(dim, 1.0))  # (lower_bounds, upper_bounds)

    # Create optimizer with adjusted line search parameters
    optimizer = LBFGSB(
        fun=objective,
        maxiter=1000,
        jit=False,
        linesearch="zoom",
        linesearch_init="increase",
        maxls=10,  # maximum line search iterations
        decrease_factor=0.5,
        increase_factor=1.5,
        max_stepsize=1.0,  # Maximum step size
        min_stepsize=1e-3,  # Minimum step size
        tol=1e-4  # Tolerance for convergence
    )

    # Optimize
    result = optimizer.run(
        init_params=x0,
        prior_cov=prior_cov,
        lklhd_cov=lklhd_cov,
        order=order,
        bounds=bounds
    )

    return result.params.reshape(-1), result


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg")

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

    # Example usage
    dim = 2  # Number of design points
    prior_cov = 1. * jnp.eye(2)  # Prior covariance
    lklhd_cov = 1e-1 * jnp.eye(1)  # Likelihood covariance

    # Create meshgrid for 2D plot
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Plot for different alpha values
    # alphas = [1e-2, 1e-1, 0.5, 0.99]
    alphas = [1e-1, 0.99, 1.25, 2.5]

    fig = plt.figure(figsize=(15, 10))
    for idx, alpha in enumerate(alphas):
        # Compute objective values
        info_gain = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                design = jnp.array([X[i, j], Y[i, j]])
                info_gain[i, j] = -1.0 * objective(
                    design,
                    prior_cov,
                    lklhd_cov,
                    alpha
                )

        # Create subplot
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        surf = ax.plot_surface(X, Y, info_gain, cmap='viridis', alpha=0.8)

        # ax = fig.add_subplot(2, 2, idx + 1)
        # im = ax.imshow(
        #     info_gain,
        #     origin='lower',
        #     extent=[-1.0, 1.0, -1.0, 1.0],
        #     cmap='viridis',
        #     alpha=0.8
        # )

        ax.set_xlabel('Design 1')
        ax.set_ylabel('Design 2')
        ax.set_title(f'alpha = {alpha}')

    plt.tight_layout()
    # plt.savefig('linear_regression.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # # Run optimization for one alpha value
    # order = 1e-1  # Order parameter
    # optimal_design, result = optimize_sibson(dim, prior_cov, lklhd_cov, order)
    #
    # print("Optimal design:", optimal_design)
    # print("Optimal value:", - 1.0 * result.state.value)
