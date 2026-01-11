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
        
        D_α(self || other) for α ∈ (0, 1)
        
        For Gaussians, the Rényi divergence has a closed form:
        D_α = (1/(2(1-α))) [quad_term + logdet_term]
        
        where:
        - Σ_α = (1-α)Σ₁ + αΣ₂ (geometric mean of covariances)
        - quad_term = (μ₁ - μ₂)ᵀ Σ_α⁻¹ (μ₁ - μ₂)
        - logdet_term = (1-α) log|Σ₁| + α log|Σ₂| - log|Σ_α|
        
        Args:
            other: Target Gaussian distribution
            alpha: Order parameter α ∈ (0, 1)
            
        Returns:
            Rényi divergence of order α
        """
        # Weighted covariance: Σ_α = (1-α)Σ₁ + αΣ₂
        Sigma_alpha = (1 - alpha) * self.cov + alpha * other.cov
        Sigma_alpha_inv = jnp.linalg.inv(Sigma_alpha)
        
        # Quadratic term: (μ₁ - μ₂)ᵀ Σ_α⁻¹ (μ₁ - μ₂)
        diff = self.mean - other.mean
        quad_term = diff.T @ Sigma_alpha_inv @ diff
        
        # Log determinant terms
        _, logdet_self = jnp.linalg.slogdet(self.cov)
        _, logdet_other = jnp.linalg.slogdet(other.cov)
        _, logdet_alpha = jnp.linalg.slogdet(Sigma_alpha)
        
        logdet_term = (1 - alpha) * logdet_self + alpha * logdet_other - logdet_alpha
        
        return (1.0 / (2.0 * (1.0 - alpha))) * (quad_term + logdet_term)
    
    def precision(self) -> Array:
        """Compute precision matrix Λ = Σ⁻¹.
        
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
