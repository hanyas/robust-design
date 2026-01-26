"""
Conjugate Linear Regression for Robust Bayesian Experimental Design
"""

from typing import NamedTuple

from jax import Array
import jax.numpy as jnp
import jax.random as jr

from jax.scipy.stats import multivariate_normal

from rdesign.distributions import Gaussian


class LinearGaussian(NamedTuple):
    """
    Gaussian linear likelihood for regression.

    Args:
        noise_stddev: Observation noise standard deviation σ.
    """
    noise_stddev: float

    def design_matrix(self, design: Array) -> Array:
        """
        Construct the design matrix H(ξ).

        Args:
            design: Raw design array.

        Returns:
            Design matrix H(ξ), shape (N, D)
        """
        size = design.shape[0]
        design = jnp.atleast_1d(design)
        return jnp.column_stack([design, jnp.ones(size)])

    def sample(self, rng_key: Array, design: Array, theta: Array) -> Array:
        """
        Sample from the likelihood p(x | θ, ξ).

        Args:
            rng_key: Random key.
            design: Raw design array.
            theta: Parameters, shape (D,).

        Returns:
            Observations, shape (N,).
        """
        mean = self.mean(design, theta)
        cov = self.cov(design)
        return jr.multivariate_normal(rng_key, mean, cov)

    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """
        Compute log p(x | θ, ξ).

        Args:
            x: Observations, shape (N,).
            design: Raw design array.
            theta: Parameters, shape (D,).

        Returns:
            Log likelihood scalar.
        """

        mean = self.mean(design, theta)
        cov = self.cov(design)
        return multivariate_normal.logpdf(x, mean, cov)

    def mean(self, design: Array, theta: Array) -> Array:
        """
        Compute the mean of p(x | θ, ξ) = H(ξ)θ.

        Args:
            design: Raw design array.
            theta: Parameters, shape (D,).

        Returns:
            Mean, shape (N,).
        """
        H = self.design_matrix(design)
        return H @ theta

    def cov(self, design: Array) -> Array:
        """
        Compute the covariance of p(x | θ, ξ).

        Args:
            design: Raw design array.

        Returns:
            Covariance matrix, shape (N, N).
        """
        num_observations = self.design_matrix(design).shape[0]
        return self.noise_stddev**2 * jnp.eye(num_observations)


class LinearRegression(NamedTuple):
    """
    Conjugate Linear Regression Model for Robust Experimental Design.

    Args:
        prior: Gaussian prior over parameters θ.
        likelihood: Gaussian linear likelihood p(x | θ, ξ).
    """
    prior: Gaussian
    likelihood: LinearGaussian

    # =========================================================================
    # Bayesian Inference
    # =========================================================================

    def get_bayes_posterior(self, x: Array, design: Array) -> Gaussian:
        """
        Compute Bayes posterior distribution p(θ | x, ξ).

        Args:
            x: Observations, shape (N,).
            design: Raw design array.

        Returns:
            Gaussian distribution.
        """
        H = self.likelihood.design_matrix(design)
        prior_prec = jnp.linalg.inv(self.prior.cov)
        lklhd_cov = self.likelihood.cov(design)

        # Posterior precision
        posterior_prec = prior_prec + H.T @ jnp.linalg.inv(lklhd_cov) @ H

        # Posterior covariance
        posterior_cov = jnp.linalg.inv(posterior_prec)

        # Posterior mean
        posterior_mean = posterior_cov @ (
            prior_prec @ self.prior.mean
            + H.T @ jnp.linalg.inv(lklhd_cov) @ x
        )

        return Gaussian(mean=posterior_mean, cov=posterior_cov)

    def get_marginal(self, design: Array) -> Gaussian:
        """
        Compute marginal distribution p(x | ξ).

        Args:
            design: Raw design array.

        Returns:
            Gaussian distribution.
        """
        H = self.likelihood.design_matrix(design)
        return Gaussian(
            mean=H @ self.prior.mean,
            cov=H @ self.prior.cov @ H.T + self.likelihood.cov(design)
        )

    def get_posterior_predictive(self, posterior: Gaussian, design: Array, alpha=float) -> Gaussian:
        """
        Compute posterior predictive distribution p(x_new | x, ξ, ξ_new).

        Args:
            posterior: Posterior distribution.
            design: Raw design array.
            alpha: Misspecification order α ∈ (0, 1).

        Returns:
            Gaussian distribution.
        """
        H = self.likelihood.design_matrix(design)
        return Gaussian(
            mean=H @ posterior.mean,
            cov=H @ posterior.cov @ H.T + self.likelihood.cov(design) / alpha
        )

    # =========================================================================
    # Robust Inference
    # =========================================================================

    def get_tilted_marginal(self, design: Array, alpha: float) -> Gaussian:
        """
        Compute α-tilted marginal distribution pₐ(x | ξ).

        Args:
            design: Raw design array.
            alpha: Misspecification order α ∈ (0, 1).

        Returns:
            Gaussian distribution.
        """
        H = self.likelihood.design_matrix(design)
        return Gaussian(
            mean=H @ self.prior.mean,
            cov=alpha * H @ self.prior.cov @ H.T + self.likelihood.cov(design)
        )

    def get_tilted_posterior(self, x: Array, design: Array, alpha: float) -> Gaussian:
        """
        Compute α-tilted posterior distribution q(θ | x, ξ).

        Args:
            x: Observations, shape (N,).
            design: Raw design array.
            alpha: Misspecification order α ∈ (0, 1).

        Returns:
            Gaussian α-tilted posterior distribution.
        """
        H = self.likelihood.design_matrix(design)
        prior_prec = self.prior.precision()
        lklhd_cov_alpha = self.likelihood.cov(design) / alpha

        # Tilted posterior precision
        posterior_prec = prior_prec + H.T @ jnp.linalg.inv(lklhd_cov_alpha) @ H

        # Tilted posterior covariance
        posterior_cov = jnp.linalg.inv(posterior_prec)

        # Tilted posterior mean
        posterior_mean = posterior_cov @ (
            prior_prec @ self.prior.mean
            + H.T @ jnp.linalg.inv(lklhd_cov_alpha) @ x
        )

        return Gaussian(mean=posterior_mean, cov=posterior_cov)

    def get_tilted_joint(self, design: Array, alpha: float) -> Gaussian:
        """
        Compute α-tilted joint distribution q(θ, x | ξ).

        Args:
            design: Raw design array.
            alpha: Misspecification order α ∈ (0, 1).

        Returns:
            Gaussian α-tilted joint distribution over [θ, x].
        """
        H = self.likelihood.design_matrix(design)
        prior_prec = self.prior.precision()
        lklhd_cov_alpha = self.likelihood.cov(design) / alpha

        # Tilted posterior precision
        posterior_prec = prior_prec + H.T @ jnp.linalg.inv(lklhd_cov_alpha) @ H

        # Tilted posterior covariance
        posterior_cov = jnp.linalg.inv(posterior_prec)

        # Mean: [θ, x]
        mean = jnp.concatenate([
            self.prior.mean,
            H @ self.prior.mean
        ])

        # Covariance blocks
        prior_cov = self.prior.cov
        cross_cov = prior_cov @ H.T
        tilted_cov = H @ prior_cov @ H.T + lklhd_cov_alpha

        # Assemble covariance matrix
        cov = jnp.block([
            [alpha * prior_cov + (1 - alpha) * posterior_cov,   alpha * cross_cov],
            [alpha * cross_cov.T,                               alpha * tilted_cov]
        ])

        return Gaussian(mean=mean, cov=cov)

    # =========================================================================
    # Information Measures
    # =========================================================================

    def shannon_mi(self, design: Array) -> float:
        """
        Compute Shannon's mutual information I(θ; x | ξ).

        Args:
            design: Raw design array.

        Returns:
            Shannon's mutual information in nats.
        """
        num_observations = self.likelihood.design_matrix(design).shape[0]
        predictive = self.get_prior_predictive(design)
        log_det_pred = jnp.linalg.slogdet(predictive.cov)[1]
        log_det_noise = num_observations * jnp.log(self.likelihood.noise_stddev**2)

        return 0.5 * (log_det_pred - log_det_noise)

    def sibson_mi(self, design: Array, alpha: float) -> float:
        """
        Compute Sibson's α-mutual information I^S_α(θ; x | ξ).

        Args:
            design: Raw design array.
            alpha: Misspecification order α ∈ (0, 1).

        Returns:
            Sibson's α-mutual information in nats.
        """
        num_observations = self.likelihood.design_matrix(design).shape[0]
        tilted_marginal = self.get_tilted_marginal(design, alpha)

        log_det_tilted = jnp.linalg.slogdet(tilted_marginal.cov)[1]
        log_det_scaled_noise = num_observations * jnp.log(self.likelihood.noise_stddev**2)

        return 0.5 * (log_det_tilted - log_det_scaled_noise)

    def shannon_ig(self, x: Array, design: Array) -> float:
        """
        Compute information gain for a specific observation.

        Args:
            x: Specific observation, shape (N,).
            design: Raw design array.

        Returns:
            KL divergence from posterior to prior in nats.
        """
        posterior = self.get_bayes_posterior(x, design)
        return posterior.kl_to(self.prior)

    def renyi_ig(self, x: Array, design: Array, alpha: float) -> float:
        """
        Compute robust information gain (Rényi divergence)

        Args:
            x: Specific observation, shape (N,).
            design: Raw design array.
            alpha: Misspecification order α ∈ (0, 1).

        Returns:
            Rényi divergence in nats.
        """
        posterior = self.get_bayes_posterior(x, design)
        return posterior.renyi_to(self.prior, alpha)
