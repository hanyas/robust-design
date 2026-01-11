"""
Conjugate Linear Regression for Robust Bayesian Experimental Design

This module implements a conjugate linear regression model with support for:
- Standard Bayesian inference (prior, likelihood, posterior)
- Robust experimental design using Sibson's α-mutual information
- Worst-case generative processes under model misspecification

The design follows the Protocol pattern from base.py with separate Gaussian prior and LinearGaussian likelihood objects.
"""

from typing import NamedTuple

from jax import Array
import jax.numpy as jnp
import jax.random as jr

from rdesign.distributions import Gaussian 



class LinearGaussian(NamedTuple):
    """
    Gaussian linear likelihood for regression.
        
    Args:
        noise_var: Observation noise variance σ²
    """
    noise_var: float

    def design_matrix(self, design: Array) -> Array:
        """
        Construct augmented design matrix H(ξ) = [ξ, 1].
                
        Args:
            design: Design points ξ, shape (N,) or scalar
            
        Returns:
            Design matrix H(ξ), shape (N, 2)
        """
        design = jnp.atleast_1d(design)
        size = design.shape[0]
        return jnp.column_stack([design, jnp.ones(size)])
    
    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        """
        Sample from the likelihood p(x | θ, ξ).
        
        Args:
            key: Random key
            design: Design points, shape (N,)
            theta: Parameters, shape (2,)
            
        Returns:
            Observations, shape (N,)
        """
        mean = self.mean(design, theta)
        cov = self.cov(design)
        return jr.multivariate_normal(key, mean, cov)
    
    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """
        Compute log p(x | θ, ξ).
        
        Args:
            x: Observations, shape (N,)
            design: Design points, shape (N,)
            theta: Parameters, shape (2,)
            
        Returns:
            Log likelihood scalar
        """
        from jax.scipy.stats import multivariate_normal

        mean = self.mean(design, theta)
        cov = self.cov(design)
        return multivariate_normal.logpdf(x, mean, cov)
    
    def mean(self, design: Array, theta: Array) -> Array:
        """
        Compute the mean of p(x | θ, ξ) = H(ξ)θ.
        
        Args:
            design: Design points, shape (N,)
            theta: Parameters, shape (2,)
            
        Returns:
            Mean, shape (N,)
        """
        H = self.design_matrix(design)
        return H @ theta
    
    def cov(self, design: Array, alpha: float = 1.0) -> Array:
        """
        Compute the covariance of p(x | θ, ξ) with α-scaling.
        
        For standard likelihood (α=1): Σ = σ²I
        For tilted likelihood: Σ = (σ²/α)I
        
        Args:
            design: Design points, shape (N,)
            alpha: Misspecification order α ∈ (0, 1], default 1.0
            
        Returns:
            Covariance matrix, shape (N, N)
        """
        size = jnp.atleast_1d(design).shape[0]
        return (self.noise_var / alpha) * jnp.eye(size)




class LinearRegression(NamedTuple):
    """
    Conjugate Linear Regression Model for Robust Experimental Design.
    
    This is a stateless factory that constructs distributions on demand.
    All methods are pure functions with no side effects.
        
    Args:
        prior: Gaussian prior over parameters θ
        likelihood: Gaussian linear likelihood p(x | θ, ξ)
    """
    prior: Gaussian 
    likelihood: LinearGaussian
    
    # =========================================================================
    # Bayesian Inference (α = 1)
    # =========================================================================
    
    def get_bayes_posterior(self, x: Array, design: Array) -> Gaussian:
        """
        Compute Bayes posterior distribution p(θ | x, ξ) = N(μₙ, Σₙ).
        
        Uses conjugate Gaussian update formulas.
        
        Args:
            x: Observations, shape (N,)
            design: Design points, shape (N,)
                        
        Returns:
            Gaussian distribution
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
    
    def get_bayes_predictive(self, design: Array) -> Gaussian:
        """
        Compute prior predictive distribution p(x | ξ).
        
        p(x | ξ) = ∫ p(x | θ, ξ) p(θ) dθ = N(H μ₀, H Σ₀ H^T + σ²I)
        
        Args:
            design: Design points, shape (N,)
            
        Returns:
            Gaussian distribution
        """
        H = self.likelihood.design_matrix(design)
        return Gaussian(
            mean=H @ self.prior.mean,
            cov=H @ self.prior.cov @ H.T + self.likelihood.cov(design)
        )
    
    def get_posterior_predictive(self, posterior: Gaussian, design_new: Array) -> Gaussian:
        """
        Compute posterior predictive distribution p(x_new | x, ξ, ξ_new).
        
        Given a posterior p(θ | x, ξ), compute the predictive distribution
        for new observations at new design points:
        
        p(x_new | x, ξ, ξ_new) = ∫ p(x_new | θ, ξ_new) p(θ | x, ξ) dθ
                                = N(H_new μ_post, H_new Σ_post H_new^T + σ²I)
        
        Args:
            posterior: Posterior distribution p(θ | x, ξ)
            design_new: New design points ξ_new, shape (M,)
            
        Returns:
            Gaussian distribution
        """
        H_new = self.likelihood.design_matrix(design_new)
        return Gaussian(
            mean=H_new @ posterior.mean,
            cov=H_new @ posterior.cov @ H_new.T + self.likelihood.cov(design_new)
        )
    
    # =========================================================================
    # Robust Inference (α < 1)
    # =========================================================================
    
    def get_tilted_marginal(self, design: Array, alpha: float) -> Gaussian:
        """
        Compute α-tilted marginal distribution pₐ(x | ξ).
        
        Used for computing Sibson's α-mutual information.
        
        Args:
            design: Design points, shape (N,)
            alpha: Misspecification order α ∈ (0, 1)
            
        Returns:
            Gaussian distribution
        """
        H = self.likelihood.design_matrix(design)
        return Gaussian(
            mean=H @ self.prior.mean,
            cov=H @ self.prior.cov @ H.T + self.likelihood.cov(design, alpha)
        )
    
    def get_tilted_posterior(self, x: Array, design: Array, alpha: float) -> Gaussian:
        """
        Compute α-tilted posterior distribution q*(θ | x, ξ).
        
        q*(θ | x, ξ) ∝ p(θ) p(x | θ, ξ)^α
        
        Args:
            x: Observations, shape (N,)
            design: Design points, shape (N,)
            alpha: Misspecification order α ∈ (0, 1)
            
        Returns:
            Gaussian α-tilted posterior distribution
        """
        H = self.likelihood.design_matrix(design)
        prior_prec = jnp.linalg.inv(self.prior.cov)
        lklhd_cov = self.likelihood.cov(design, alpha)

        # Tilted posterior precision
        posterior_prec = prior_prec + H.T @ jnp.linalg.inv(lklhd_cov) @ H
        
        # Tilted posterior covariance
        posterior_cov = jnp.linalg.inv(posterior_prec)
        
        # Tilted posterior mean
        posterior_mean = posterior_cov @ (
            prior_prec @ self.prior.mean 
            + H.T @ jnp.linalg.inv(lklhd_cov) @ x
        )
        
        return Gaussian(mean=posterior_mean, cov=posterior_cov)
    
    def get_tilted_joint(self, design: Array, alpha: float) -> Gaussian:
        """
        Compute α-tilted joint distribution q*(θ, x | ξ).
        
        Returns a Gaussian over the augmented space [θ, x].
        
        Args:
            design: Design points, shape (N,)
            alpha: Misspecification order α ∈ (0, 1)
            
        Returns:
            Gaussian α-tilted joint distribution over [θ, x]
        """
        H = self.likelihood.design_matrix(design)
        lklhd_cov = self.likelihood.cov(design, alpha)
        
        # Mean: [θ, x]
        mean = jnp.concatenate([
            self.prior.mean,
            H @ self.prior.mean
        ])
        
        # Covariance blocks
        prior_cov = self.prior.cov
        prior_cov_x = prior_cov @ H.T
        x_cov = H @ prior_cov @ H.T + lklhd_cov
        
        # Assemble covariance matrix
        cov = jnp.block([
            [prior_cov,     prior_cov_x],
            [prior_cov_x.T, x_cov]
        ])
        
        return Gaussian(mean=mean, cov=cov)
    
    # =========================================================================
    # Information Measures
    # =========================================================================

    def shannon_mi(self, design: Array) -> float:
        """
        Compute Shannon's mutual information I(θ; x | ξ).
        
        I(θ; x) = H[p(x|ξ)] - H[p(x|θ,ξ)]
                = (1/2) log|Σ_pred| - (1/2) log|σ²I|
                
        Args:
            design: Design points, shape (N,)
            
        Returns:
            Shannon's mutual information in nats
        """
        size = jnp.atleast_1d(design).shape[0]
        predictive = self.get_bayes_predictive(design)
        log_det_pred = jnp.linalg.slogdet(predictive.cov)[1]
        log_det_noise = size * jnp.log(self.likelihood.noise_var)
        
        return 0.5 * (log_det_pred - log_det_noise)
    
    def sibson_mi(self, design: Array, alpha: float) -> float:
        """
        Compute Sibson's α-mutual information I^S_α(θ; x | ξ).
        
        Robust generalization of Shannon MI for α ∈ (0, 1).
        
        Args:
            design: Design points, shape (N,)
            alpha: Misspecification order α ∈ (0, 1)
            
        Returns:
            Sibson's α-mutual information in nats
        """
        size = jnp.atleast_1d(design).shape[0]
        tilted_marginal = self.get_tilted_marginal(design, alpha)
    
        log_det_tilted = jnp.linalg.slogdet(tilted_marginal.cov)[1]
        log_det_scaled_noise = size * jnp.log(self.likelihood.noise_var / alpha)
        
        return 0.5 * (log_det_tilted - log_det_scaled_noise)
    
    # Backward compatibility aliases
    shannon_mutual_information = shannon_mi
    sibson_alpha_mutual_information = sibson_mi
    
    def shannon_ig(self, x: Array, design: Array) -> float:
        """
        Compute expected information gain for a specific observation.
        
        EIG(x, ξ) = D_KL(p(θ | x, ξ) || p(θ))
        
        Args:
            x: Specific observation, shape (N,)
            design: Design points, shape (N,)
            
        Returns:
            Information gain in nats
        """
        posterior = self.get_bayes_posterior(x, design)
        return posterior.kl_to(self.prior)
    
    def renyi_ig(self, x: Array, design: Array, alpha: float) -> float:
        """
        Compute robust information gain (Rényi divergence).
        
        G_α(x, ξ) = D_α(q*(θ | x, ξ) || p(θ))
        
        Uses the α-tilted posterior q*(θ | x, ξ).
        
        Args:
            x: Specific observation, shape (N,)
            design: Design points, shape (N,)
            alpha: Misspecification order α ∈ (0, 1)
            
        Returns:
            Rényi divergence in nats
        """
        posterior = self.get_tilted_posterior(x, design, alpha)
        return posterior.kl_to(self.prior)
