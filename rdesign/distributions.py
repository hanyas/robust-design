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


class Gaussian(NamedTuple):
    """Base multivariate Gaussian distribution N(μ, Σ).
    
    This is the base class for all Gaussian distributions. Use specific
    subclasses for clarity:
    - GaussianPrior: for p(θ)
    - GaussianPosterior: for p(θ | x, ξ)
    - GaussianPredictive: for p(x | ξ) or p(x_new | x, ξ)
    
    Args:
        mean: Mean vector μ, shape (d,)
        cov: Covariance matrix Σ, shape (d, d)
    """
    mean: Array
    cov: Array
    
    def sample(self, key: Array, num_samples: int) -> Array:
        """Sample from the Gaussian distribution.
        
        Args:
            key: JAX random key
            num_samples: Number of samples to draw
            
        Returns:
            Samples of shape (num_samples, dim)
        """
        return jr.multivariate_normal(key, self.mean, self.cov, shape=(num_samples,))
    
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
        
        For a d-dimensional Gaussian:
        H[N(μ, Σ)] = (d/2) log(2πe) + (1/2) log|Σ|
        
        Returns:
            Differential entropy in nats
        """
        d = self.dim
        _, logdet = jnp.linalg.slogdet(self.cov)
        return 0.5 * d * jnp.log(2 * jnp.pi * jnp.e) + 0.5 * logdet
    
    def kl_to(self, other: 'Gaussian') -> float:
        """Compute KL divergence to another Gaussian.
        
        D_KL(self || other) = KL(N(μ₁, Σ₁) || N(μ₂, Σ₂))
        
        Formula:
        D_KL = (1/2) [tr(Σ₂⁻¹ Σ₁) + (μ₂ - μ₁)ᵀ Σ₂⁻¹ (μ₂ - μ₁) - d + log(|Σ₂|/|Σ₁|)]
        
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
        
        d²(x, μ) = (x - μ)ᵀ Σ⁻¹ (x - μ)
        
        Args:
            value: Point(s) to evaluate, shape (dim,) or (n, dim)
            
        Returns:
            Squared Mahalanobis distance, scalar or shape (n,)
        """
        diff = jnp.atleast_2d(value - self.mean)
        precision = self.precision()
        return jnp.sum(diff @ precision * diff, axis=1)
