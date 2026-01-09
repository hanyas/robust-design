"""
Preference Learning Experiment

Implementation of a preference learning model with censored and corrupted responses.
The model is:
    theta ~ N(mu_theta, sigma_theta^2)
    eta | theta, xi ~ N(xi - theta, sigma_eta^2 (1 + |xi|)²)
    y = f(eta)
where xi is the design, eta is a latent variable, and y is the censored response.
The function f maps eta to [eps, 1-eps] using a piecewise function.
"""
import jax
from jax import Array, random, numpy as jnp
from jax.scipy import stats

from rdesign.base import Prior, Model
from rdesign.utils import info_gain

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Qt5Agg")


class PreferencePrior(Prior):
    """Gaussian prior over the preference point theta."""
    def __init__(self, mean: Array, std: Array):
        """
        Args:
            mean: Prior mean for theta
            std: Prior standard deviation for theta
        """
        self.mean = mean
        self.std = std

    def sample(self, key: Array, num_samples: int) -> Array:
        """Sample theta from univariate Gaussian."""
        return self.mean + self.std * random.normal(key, shape=(num_samples,))


class PreferenceModel(Model):
    """
    Preference model with censored and corrupted responses:
    eta | theta, xi ~ N(xi - theta, sigma_n^2 (1 + |design|)²)
    y = f(eta) where f maps to [eps, 1 - eps]
    """
    def __init__(self, noise_std: Array, epsilon: Array = jnp.array(0.01)):
        self.noise_std = noise_std
        self.epsilon = epsilon

    def _compute_mean(self, design: Array, theta: Array) -> Array:
        """Compute mean of eta given design xi and parameter theta."""
        return design - theta

    def _compute_stddev(self, design: Array, theta: Array) -> Array:
        """Compute standard deviation of eta given design xi."""
        return self.noise_std * (1 + jnp.abs(design))

    def _logit(self, x: Array) -> Array:
        return jnp.log(x) - jnp.log(1 - x)

    def _censoring_function(self, x: Array) -> Array:
        """
        Apply the censoring function f that maps to [eps, 1 - eps].
        Implements the piecewise function
        """

        return jnp.where(
            x <= self._logit(self.epsilon),
            self.epsilon,
            jnp.where(
                x >= self._logit(1 - self.epsilon),
                1 - self.epsilon,
                1 / (1 - jnp.exp(-x))
            )
        )

    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        """
        Sample from the likelihood.
        Args:
            key: JAX PRNGKey
            design: shape (N,)
            theta: shape (1,)
        Returns:
            y: shape (N,)
        """
        mean = jax.vmap(self._compute_mean, in_axes=(0, None))(design, theta)
        stddev = jax.vmap(self._compute_stddev, in_axes=(0, None))(design, theta)

        eta = mean + stddev * random.normal(key)
        return jax.vmap(self._censoring_function)(eta)

    def logpdf(self, y: Array, design: Array, theta: Array) -> Array:
        """
        Log likelihood of y given design and theta.
        """
        mean = jax.vmap(self._compute_mean, in_axes=(0, None))(design, theta)
        stddev = jax.vmap(self._compute_stddev, in_axes=(0, None))(design, theta)

        eps = self.epsilon
        logit_upper = self._logit(eps)
        logit_lower = self._logit(1 - eps)

        x = jnp.log(y / (1. - y))
        log_abs_jac = -jnp.log(y) - jnp.log(1. - y)

        logp = jnp.where(
            y <= eps,
            stats.norm.logcdf(logit_lower, loc=mean, scale=stddev),
            jnp.where(
                y >= 1 - eps,
                stats.norm.logsf(logit_upper, loc=mean, scale=stddev),
                stats.norm.logpdf(x, loc=mean, scale=stddev) + log_abs_jac
            )
        )
        return jnp.sum(logp)


def create_preference_model(
    prior_mean: Array = jnp.zeros(1),
    prior_std: Array = jnp.ones(1),
    noise_std: Array = jnp.ones(1),
    epsilon: Array = jnp.array(0.01),
) -> tuple[Prior, Model]:
    prior = PreferencePrior(prior_mean, prior_std)
    model = PreferenceModel(noise_std=noise_std, epsilon=epsilon)
    return prior, model


# Example usage
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    prior_mean = jnp.array(-20.)
    prior_std = jnp.array(20.0)

    key = random.PRNGKey(0)
    prior, model = create_preference_model(
        prior_mean=prior_mean,
        prior_std=prior_std,
        noise_std=jnp.array(0.2),
        epsilon=jnp.array(.01)
    )

    # Define alphas and design grid
    alphas = jnp.array([0.01, 0.99])
    n_grid = 320
    designs = jnp.linspace(-80.0, 80.0, n_grid)
    info_gain_vals = jnp.zeros((len(alphas), n_grid))

    num_outer = int(1 * 1024)
    num_inner = int(1 * 1024)

    for i, alpha in enumerate(alphas):
        print(f"Evaluating alpha = {alpha:.2f}...")
        for j in range(n_grid):
            key, sub_key = random.split(key)
            design = jnp.array([designs[j]])
            info_gain_vals = info_gain_vals.at[i, j].set(
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

    # Plotting
    plt.figure(figsize=(12, 5))
    for i, alpha in enumerate(alphas):
        plt.subplot(1, 2, i + 1)
        plt.plot(designs, info_gain_vals[i])
        plt.title(f'alpha = {alpha:.2f}')
        plt.xlabel('Design')
        plt.ylabel('Information Gain')
        plt.grid(True)

    plt.tight_layout()
    plt.show()
