"""
Distribution classes for Bayesian experimental design.

This module provides a base Gaussian class and specific subclasses:
- GaussianPrior: for p(θ)
- GaussianPosterior: for p(θ | x, ξ)
- GaussianPredictive: for p(x | ξ) or p(x_new | x, ξ)

All share the same functionality but have clear semantic names.
"""

from typing import NamedTuple
from jax import Array
import jax.numpy as jnp
import jax.random as jr

from jax.scipy.stats import multivariate_normal
from jax.scipy.special import betaln, gammaln, digamma, logsumexp


class Gaussian(NamedTuple):
    """Multivariate Gaussian distribution N(μ, Σ).
        
    Args:
        mean: Mean vector μ, shape (d,)
        cov: Covariance matrix Σ, shape (d, d)
    """
    mean: Array
    cov: Array
    
    def sample(self, rng_key: Array, num_samples: int) -> Array:
        """Sample from the Gaussian distribution.
        
        Args:
            rng_key: JAX random key
            num_samples: Number of samples to draw
            
        Returns:
            Samples of shape (num_samples, dim)
        """
        return jr.multivariate_normal(rng_key, self.mean, self.cov, shape=(num_samples,))
    
    def logpdf(self, value: Array) -> Array:
        """Compute log probability density.
        
        Args:
            value: Value(s) to evaluate, shape (dim,) or (n, dim)
            
        Returns:
            Log probability, scalar or shape (n,)
        """
        return multivariate_normal.logpdf(value, self.mean, self.cov)
    
    @property
    def dim(self) -> int:
        """Dimension of the distribution."""
        return self.mean.shape[0]
    
    # Utility methods
    
    def entropy(self) -> float:
        """Compute differential entropy H[N(μ, Σ)].
                
        Returns:
            Differential entropy in nats
        """
        d = self.dim
        _, logdet = jnp.linalg.slogdet(self.cov)
        return 0.5 * d * jnp.log(2 * jnp.pi * jnp.e) + 0.5 * logdet
    
    def kl_to(self, other: 'Gaussian') -> float:
        """Compute KL divergence to another Gaussian.
        
        Args:
            other: Target Gaussian distribution
            
        Returns:
            KL divergence in nats
        """
        d = self.dim
        
        # Compute Σ₂⁻¹
        other_cov_inv = jnp.linalg.inv(other.cov)
        
        # Trace term: tr(Σ₂⁻¹ Σ₁)
        trace_term = jnp.trace(other_cov_inv @ self.cov)
        
        # Quadratic term: (μ₂ - μ₁)ᵀ Σ₂⁻¹ (μ₂ - μ₁)
        diff = other.mean - self.mean
        quad_term = diff.T @ other_cov_inv @ diff
        
        # Log determinant term: log|Σ₂| - log|Σ₁|
        _, logdet_self = jnp.linalg.slogdet(self.cov)
        _, logdet_other = jnp.linalg.slogdet(other.cov)
        logdet_term = logdet_other - logdet_self
        
        return 0.5 * (trace_term + quad_term - d + logdet_term)
    
    def renyi_to(self, other: 'Gaussian', alpha: float) -> float:
        """Compute Rényi divergence of order α to another Gaussian.
        
        Args:
            other: Target Gaussian distribution
            alpha: Order parameter α ∈ (0, 1)
            
        Returns:
            Rényi divergence of order α
        """
        d = self.dim 

        # Precision of the geometric mixture
        Sigma_alpha_inv = alpha * self.precision() + (1 - alpha) * other.precision()
        Sigma_alpha = jnp.linalg.inv(Sigma_alpha_inv)

        # Mean of the geometric mixture
        mu_alpha = Sigma_alpha @ (
            alpha * self.precision() @ self.mean
            + (1 - alpha) * other.precision() @ other.mean
        )

        # Log-normalizer helper: 0.5 * (k log(2π) + log|Σ|)
        def log_Z_val(logdet_sigma):
            return 0.5 * d * jnp.log(2 * jnp.pi) + 0.5 * logdet_sigma

        # Log determinant terms
        _, logdet_self = jnp.linalg.slogdet(self.cov)
        _, logdet_other = jnp.linalg.slogdet(other.cov)
        _, logdet_alpha = jnp.linalg.slogdet(Sigma_alpha)

        log_Z_self = log_Z_val(logdet_self)
        log_Z_other = log_Z_val(logdet_other)
        log_Z_alpha = log_Z_val(logdet_alpha)
        
        # Quadratic scaling terms
        quad_term = 0.5 * (
            alpha * self.mean @ self.precision() @ self.mean 
            + (1 - alpha) * other.mean @ other.precision() @ other.mean
            - mu_alpha @ Sigma_alpha_inv @ mu_alpha
        )
        
        # Log integral of
        log_integral = log_Z_alpha - alpha * log_Z_self - (1 - alpha) * log_Z_other - quad_term

        return (1.0 / (alpha - 1.0)) * log_integral

    def precision(self) -> Array:
        """Compute precision matrix.
        
        Returns:
            Precision matrix, shape (dim, dim)
        """
        return jnp.linalg.inv(self.cov)
    
    def mahalanobis(self, value: Array) -> Array:
        """Compute squared Mahalanobis distance.
                
        Args:
            value: Point(s) to evaluate, shape (dim,) or (n, dim)
            
        Returns:
            Squared Mahalanobis distance, scalar or shape (n,)
        """
        diff = jnp.atleast_2d(value - self.mean)
        precision = self.precision()
        return jnp.sum(diff @ precision * diff, axis=1)


class Beta(NamedTuple):
    """
    Beta distribution parameterized by delta (δ) and gamma (γ).

    Args:
        delta: Shape parameter δ > 0, array of shape (2,) for [group_A, group_B]
        gamma: Shape parameter γ > 0, array of shape (2,) for [group_A, group_B]
    """
    delta: Array  # shape (2,) - corresponds to δ in the paper
    gamma: Array   # shape (2,) - corresponds to γ in the paper

    @property
    def mean(self) -> Array:
        """E[θ] = δ / (δ + γ)"""
        return self.delta / (self.delta + self.gamma)

    @property
    def var(self) -> Array:
        """Var[θ] = δγ / [(δ + γ)² (δ + γ + 1)]"""
        total = self.delta + self.gamma
        return (self.delta * self.gamma) / (total ** 2 * (total + 1))

    @property
    def concentration(self) -> Array:
        """Total pseudo-observations δ + γ."""
        return self.delta + self.gamma

    def sample(self, rng_key: Array, shape: tuple = ()) -> Array:
        """Sample from Beta(δ, γ)."""
        return jr.beta(rng_key, self.delta, self.gamma, shape=shape + (2,))

    def log_normalizer(self) -> Array:
        """Log normalizing constant: log B(δ, γ)"""
        return betaln(self.delta, self.gamma)

    def kl_to(self, other: 'Beta') -> Array:
        """
        KL divergence D_KL(self || other).
        """
        psi_self_delta = digamma(self.delta)
        psi_self_gamma = digamma(self.gamma)
        psi_self_total = digamma(self.delta + self.gamma)

        kl = (
            betaln(other.delta, other.gamma) - betaln(self.delta, self.gamma) +
            (self.delta - other.delta) * (psi_self_delta - psi_self_total) +
            (self.gamma - other.gamma) * (psi_self_gamma - psi_self_total)
        )
        return jnp.sum(kl)

    def renyi_to(self, other: 'Beta', alpha: float) -> float:
        """
        Compute Rényi divergence D_α(self || other) for Beta distributions.
        """
        # Effective parameters for the integral
        delta_eff = alpha * (self.delta - 1) + (1 - alpha) * (other.delta - 1) + 1
        gamma_eff = alpha * (self.gamma - 1) + (1 - alpha) * (other.gamma - 1) + 1
                
        log_integral = (
            betaln(delta_eff, gamma_eff) 
            - alpha * betaln(self.delta, self.gamma) 
            - (1 - alpha) * betaln(other.delta, other.gamma)
        )
        
        divergence = (1.0 / (alpha - 1.0)) * log_integral
        return jnp.sum(divergence)