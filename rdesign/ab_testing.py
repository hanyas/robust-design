"""
A/B Testing Model for Robust Bayesian Experimental Design

This module implements an A/B testing model with support for:
- Standard Bayesian inference (prior, likelihood, posterior)
- Robust experimental design using Sibson's α-mutual information
- Analytical formulas for conjugate Gaussian updates

The design follows the Protocol pattern from base.py with separate Gaussian prior
and ABTestingLikelihood objects.

Model Structure:
- Parameters θ = [θ₀, θ₁]: mean responses for groups 0 and 1
- Design ξ ∈ {0, 1, ..., n}: number of participants assigned to group 0
- Observations x ∈ ℝⁿ: independent Gaussian responses per participant
"""

from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr

from rdesign.distributions import Gaussian


class MaskedLinearGaussian(NamedTuple):
    """
    Gaussian likelihood for A/B testing with independent observations.
    
    Each participant is assigned to one of two groups. Participants in group g
    have observations drawn from N(θ_g, σ_g²).
    
    Args:
        num_participants: Total number of participants n
        observation_std_devs: Standard deviations [σ₀, σ₁] for each group
    """
    num_participants: int
    observation_std_devs: Array
    
    def _group_mask(self, design: Array) -> Array:
        """
        Create boolean mask indicating group assignment.
        
        Args:
            design: Number of participants in group 0 (scalar or array)
            
        Returns:
            Boolean array of shape (num_participants,) where True = group 0
        """
        design = jnp.asarray(design, dtype=jnp.int32)
        design = jnp.clip(design, 0, self.num_participants)
        idx = jnp.arange(self.num_participants)
        return idx < design
    
    def sample(self, key: Array, design: Array, theta: Array) -> Array:
        """
        Sample from the likelihood p(x | θ, ξ).
        
        Args:
            key: Random key
            design: Number of participants in group 0
            theta: Parameters [θ₀, θ₁]
            
        Returns:
            Observations x of shape (num_participants,)
        """
        mean = self.mean(design, theta)
        std = self._observation_std(design)
        return mean + jr.normal(key, (self.num_participants,)) * std
    
    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """
        Compute log p(x | θ, ξ).
        
        Since observations are independent, this is the sum of individual log pdfs.
        
        Args:
            x: Observations, shape (num_participants,)
            design: Number of participants in group 0
            theta: Parameters [θ₀, θ₁]
            
        Returns:
            Log likelihood scalar
        """
        from jax.scipy.stats import norm
        
        mean = self.mean(design, theta)
        scale = self._observation_std(design)
        return jnp.sum(norm.logpdf(x, loc=mean, scale=scale))
    
    def mean(self, design: Array, theta: Array) -> Array:
        """
        Compute the mean of p(x | θ, ξ).
        
        Args:
            design: Number of participants in group 0
            theta: Parameters [θ₀, θ₁]
            
        Returns:
            Mean vector of shape (num_participants,)
        """
        mask = self._group_mask(design)
        return jnp.where(mask, theta[0], theta[1])
    
    def _observation_std(self, design: Array) -> Array:
        """
        Get per-participant observation standard deviations.
        
        Args:
            design: Number of participants in group 0
            
        Returns:
            Standard deviation vector of shape (num_participants,)
        """
        mask = self._group_mask(design)
        return jnp.where(mask, self.observation_std_devs[0], self.observation_std_devs[1])
    
    def cov(self, design: Array, alpha: float = 1.0) -> Array:
        """
        Compute the covariance of p(x | θ, ξ) with α-scaling.
        
        For standard likelihood (α=1): Σ = diag([σ₀², ..., σ₀², σ₁², ..., σ₁²])
        For tilted likelihood: Σ = (1/α) * diag([σ₀², ..., σ₀², σ₁², ..., σ₁²])
        
        Args:
            design: Number of participants in group 0
            alpha: Misspecification order α ∈ (0, 1], default 1.0
            
        Returns:
            Covariance matrix of shape (num_participants, num_participants)
        """
        std = self._observation_std(design)
        return (1.0 / alpha) * jnp.diag(std ** 2)


class ABTesting(NamedTuple):
    """
    A/B Testing Model for Robust Experimental Design.
    
    This is a stateless factory that constructs distributions on demand.
    All methods are pure functions with no side effects.
    
    Args:
        prior: Gaussian prior over parameters θ = [θ₀, θ₁]
        likelihood: A/B testing likelihood p(x | θ, ξ)
    """
    prior: Gaussian
    likelihood: MaskedLinearGaussian
    
    # =========================================================================
    # Bayesian Inference (α = 1)
    # =========================================================================
    
    def get_bayes_posterior(self, x: Array, design: Array) -> Gaussian:
        """
        Compute Bayes posterior distribution p(θ | x, ξ) = N(μₙ, Σₙ).
        
        Uses analytical conjugate Gaussian update formulas. Since observations
        are independent within each group, we can update each θ_g separately.
        
        Args:
            x: Observations, shape (num_participants,)
            design: Number of participants in group 0 (scalar)
            
        Returns:
            Gaussian posterior distribution
        """
        design = jnp.asarray(design, dtype=jnp.int32)
        design = jnp.clip(design, 0, self.likelihood.num_participants)
        
        # Group assignments
        mask = self.likelihood._group_mask(design)
        n_group0 = design
        n_group1 = self.likelihood.num_participants - design
        
        # Prior parameters
        prior_means = self.prior.mean
        prior_vars = jnp.diag(self.prior.cov)
        
        # Observation variances
        obs_vars = self.likelihood.observation_std_devs ** 2
        
        # Posterior variances (closed form for conjugate Gaussian)
        posterior_var0 = 1.0 / (1.0 / prior_vars[0] + n_group0 / obs_vars[0])
        posterior_var1 = 1.0 / (1.0 / prior_vars[1] + n_group1 / obs_vars[1])
        
        # Sum of observations in each group
        sum_group0 = jnp.sum(x * mask)
        sum_group1 = jnp.sum(x * (~mask))
        
        # Posterior means
        posterior_mean0 = posterior_var0 * (prior_means[0] / prior_vars[0] + sum_group0 / obs_vars[0])
        posterior_mean1 = posterior_var1 * (prior_means[1] / prior_vars[1] + sum_group1 / obs_vars[1])
        
        posterior_mean = jnp.array([posterior_mean0, posterior_mean1])
        posterior_cov = jnp.diag(jnp.array([posterior_var0, posterior_var1]))
        
        return Gaussian(mean=posterior_mean, cov=posterior_cov)
    
    def get_bayes_predictive(self, design: Array) -> Gaussian:
        """
        Compute prior predictive distribution p(x | ξ).
        
        p(x | ξ) = ∫ p(x | θ, ξ) p(θ) dθ
        
        Since observations are independent given θ, the predictive distribution
        is a product of independent Gaussians with inflated variances.
        
        Args:
            design: Number of participants in group 0
            
        Returns:
            Gaussian distribution (with diagonal covariance)
        """
        # Mean: E[x | ξ] = E[E[x | θ, ξ]] = E[θ_mask]
        mask = self.likelihood._group_mask(design)
        mean = jnp.where(mask, self.prior.mean[0], self.prior.mean[1])
        
        # Variance: Var[x | ξ] = E[Var[x | θ, ξ]] + Var[E[x | θ, ξ]]
        #                      = σ²_mask + Var[θ_mask]
        prior_vars = jnp.diag(self.prior.cov)
        obs_vars = self.likelihood.observation_std_devs ** 2
        
        var = jnp.where(
            mask,
            obs_vars[0] + prior_vars[0],
            obs_vars[1] + prior_vars[1]
        )
        
        return Gaussian(mean=mean, cov=jnp.diag(var))
    
    def get_posterior_predictive(self, posterior: Gaussian, design_new: Array) -> Gaussian:
        """
        Compute posterior predictive distribution p(x_new | x, ξ, ξ_new).
        
        Given a posterior p(θ | x, ξ), compute the predictive distribution
        for new observations at new design points.
        
        Args:
            posterior: Posterior distribution p(θ | x, ξ)
            design_new: New design (number of participants in group 0)
            
        Returns:
            Gaussian distribution
        """
        # Mean: E[x_new | x, ξ, ξ_new] = E[θ_mask]_posterior
        mask = self.likelihood._group_mask(design_new)
        mean = jnp.where(mask, posterior.mean[0], posterior.mean[1])
        
        # Variance: Var[x_new | x, ξ, ξ_new] = σ²_mask + Var[θ_mask]_posterior
        posterior_vars = jnp.diag(posterior.cov)
        obs_vars = self.likelihood.observation_std_devs ** 2
        
        var = jnp.where(
            mask,
            obs_vars[0] + posterior_vars[0],
            obs_vars[1] + posterior_vars[1]
        )
        
        return Gaussian(mean=mean, cov=jnp.diag(var))
    
    # =========================================================================
    # Robust Inference (α < 1)
    # =========================================================================
    
    def get_tilted_marginal(self, design: Array, alpha: float) -> Gaussian:
        """
        Compute α-tilted marginal distribution p_α(x | ξ).
        
        Used for computing Sibson's α-mutual information.
        
        Args:
            design: Number of participants in group 0
            alpha: Misspecification order α ∈ (0, 1)
            
        Returns:
            Gaussian distribution
        """
        # Mean is unchanged
        mask = self.likelihood._group_mask(design)
        mean = jnp.where(mask, self.prior.mean[0], self.prior.mean[1])
        
        # Variance: Var_α[x | ξ] = σ²_mask/α + Var[θ_mask]
        prior_vars = jnp.diag(self.prior.cov)
        obs_vars = self.likelihood.observation_std_devs ** 2
        
        var = jnp.where(
            mask,
            obs_vars[0] / alpha + prior_vars[0],
            obs_vars[1] / alpha + prior_vars[1]
        )
        
        return Gaussian(mean=mean, cov=jnp.diag(var))
    
    def get_tilted_posterior(self, x: Array, design: Array, alpha: float) -> Gaussian:
        """
        Compute α-tilted posterior distribution q*(θ | x, ξ).
        
        q*(θ | x, ξ) ∝ p(θ) p(x | θ, ξ)^α
        
        Args:
            x: Observations, shape (num_participants,)
            design: Number of participants in group 0
            alpha: Misspecification order α ∈ (0, 1)
            
        Returns:
            Gaussian α-tilted posterior distribution
        """
        design = jnp.asarray(design, dtype=jnp.int32)
        design = jnp.clip(design, 0, self.likelihood.num_participants)
        
        # Group assignments
        mask = self.likelihood._group_mask(design)
        n_group0 = design
        n_group1 = self.likelihood.num_participants - design
        
        # Prior parameters
        prior_means = self.prior.mean
        prior_vars = jnp.diag(self.prior.cov)
        
        # Observation variances
        obs_vars = self.likelihood.observation_std_devs ** 2
        
        # Tilted posterior variances
        posterior_var0 = 1.0 / (1.0 / prior_vars[0] + alpha * n_group0 / obs_vars[0])
        posterior_var1 = 1.0 / (1.0 / prior_vars[1] + alpha * n_group1 / obs_vars[1])
        
        # Sum of observations in each group
        sum_group0 = jnp.sum(x * mask)
        sum_group1 = jnp.sum(x * (~mask))
        
        # Tilted posterior means
        posterior_mean0 = posterior_var0 * (prior_means[0] / prior_vars[0] + alpha * sum_group0 / obs_vars[0])
        posterior_mean1 = posterior_var1 * (prior_means[1] / prior_vars[1] + alpha * sum_group1 / obs_vars[1])
        
        posterior_mean = jnp.array([posterior_mean0, posterior_mean1])
        posterior_cov = jnp.diag(jnp.array([posterior_var0, posterior_var1]))
        
        return Gaussian(mean=posterior_mean, cov=posterior_cov)
    
    def get_tilted_joint(self, design: Array, alpha: float) -> Gaussian:
        """
        Compute α-tilted joint distribution q*(θ, x | ξ).
        
        Returns a Gaussian over the augmented space [θ, x].
        
        Args:
            design: Number of participants in group 0
            alpha: Misspecification order α ∈ (0, 1)
            
        Returns:
            Gaussian α-tilted joint distribution over [θ, x]
        """
        # Group mask
        mask = self.likelihood._group_mask(design)
        
        # Prior parameters
        prior_means = self.prior.mean
        prior_cov = self.prior.cov
        
        # Observation parameters
        obs_vars = self.likelihood.observation_std_devs ** 2
        
        # Mean: [θ, x]
        x_mean = jnp.where(mask, prior_means[0], prior_means[1])
        mean = jnp.concatenate([prior_means, x_mean])
        
        # Covariance blocks
        # Cov[θ, θ] = Σ_prior
        theta_cov = prior_cov
        
        # Cov[θ, x]: each x_i has covariance with its corresponding θ_g
        # If x_i is in group 0: Cov[θ, x_i] = [Var[θ_0], 0]
        # If x_i is in group 1: Cov[θ, x_i] = [0, Var[θ_1]]
        prior_vars = jnp.diag(prior_cov)
        theta_x_cov = jnp.zeros((2, self.likelihood.num_participants))
        theta_x_cov = theta_x_cov.at[0, :].set(jnp.where(mask, prior_vars[0], 0.0))
        theta_x_cov = theta_x_cov.at[1, :].set(jnp.where(mask, 0.0, prior_vars[1]))
        
        # Cov[x, x]: diagonal with group-specific inflated variances
        x_var = jnp.where(
            mask,
            obs_vars[0] / alpha + prior_vars[0],
            obs_vars[1] / alpha + prior_vars[1]
        )
        x_cov = jnp.diag(x_var)
        
        # Assemble full covariance
        cov = jnp.block([
            [theta_cov,      theta_x_cov],
            [theta_x_cov.T,  x_cov]
        ])
        
        return Gaussian(mean=mean, cov=cov)
    
    # =========================================================================
    # Information Measures
    # =========================================================================
    
    def shannon_mi(self, design: Array) -> float:
        """
        Compute Shannon's mutual information I(θ; x | ξ).
        
        For independent observations within groups:
        I(θ; x | ξ) = 0.5 * [log(1 + Var[θ_0] * n_0 / σ²_0) + log(1 + Var[θ_1] * n_1 / σ²_1)]
        
        Args:
            design: Number of participants in group 0
            
        Returns:
            Shannon's mutual information in nats
        """
        design = jnp.asarray(design, dtype=jnp.int32)
        design = jnp.clip(design, 0, self.likelihood.num_participants)
        
        n_group0 = design
        n_group1 = self.likelihood.num_participants - design
        
        # Prior variances and observation variances
        prior_vars = jnp.diag(self.prior.cov)
        obs_vars = self.likelihood.observation_std_devs ** 2
        
        # Exact MI: log(1 + signal-to-noise ratio)
        # Note: n_group is INSIDE the log (scales the SNR, not the MI)
        term0 = jnp.log(1.0 + prior_vars[0] * n_group0 / obs_vars[0])
        term1 = jnp.log(1.0 + prior_vars[1] * n_group1 / obs_vars[1])
        
        return 0.5 * (term0 + term1)
    
    def sibson_mi(self, design: Array, alpha: float) -> float:
        """
        Compute Sibson's α-mutual information I^S_α(θ; x | ξ).
        
        Robust generalization of Shannon MI for α ∈ (0, 1).
        
        For independent observations within groups:
        I^S_α = 0.5 * [log(1 + α * Var[θ_0] * n_0 / σ²_0) + log(1 + α * Var[θ_1] * n_1 / σ²_1)]
        
        Args:
            design: Number of participants in group 0
            alpha: Misspecification order α ∈ (0, 1)
            
        Returns:
            Sibson's α-mutual information in nats
        """
        design = jnp.asarray(design, dtype=jnp.int32)
        design = jnp.clip(design, 0, self.likelihood.num_participants)
        
        n_group0 = design
        n_group1 = self.likelihood.num_participants - design
        
        # Prior variances and observation variances
        prior_vars = jnp.diag(self.prior.cov)
        obs_vars = self.likelihood.observation_std_devs ** 2
        
        # Exact MI: log(1 + signal-to-noise ratio)
        # Note: n_group is INSIDE the log (scales the SNR, not the MI)
        term0 = jnp.log(1.0 + alpha * prior_vars[0] * n_group0 / obs_vars[0])
        term1 = jnp.log(1.0 + alpha * prior_vars[1] * n_group1 / obs_vars[1])
        
        return 0.5 * (term0 + term1)
    
    # Backward compatibility aliases
    shannon_mutual_information = shannon_mi
    sibson_alpha_mutual_information = sibson_mi
    
    def shannon_ig(self, x: Array, design: Array) -> float:
        """
        Compute expected information gain for a specific observation.
        
        EIG(x, ξ) = D_KL(p(θ | x, ξ) || p(θ))
        
        Args:
            x: Specific observation, shape (num_participants,)
            design: Number of participants in group 0
            
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
            x: Specific observation, shape (num_participants,)
            design: Number of participants in group 0
            alpha: Misspecification order α ∈ (0, 1)
            
        Returns:
            Rényi divergence in nats
        """
        posterior = self.get_tilted_posterior(x, design, alpha)
        return posterior.renyi_to(self.prior, alpha)
    
    def optimal_design_shannon(self) -> int:
        """
        Find the optimal design that maximizes Shannon's mutual information.
        
        Returns:
            Optimal number of participants to assign to group 0
        """
        designs = jnp.arange(self.likelihood.num_participants + 1)
        mi_values = jax.vmap(self.shannon_mi)(designs)
        return int(jnp.argmax(mi_values))
    
    def optimal_design_sibson(self, alpha: float) -> int:
        """
        Find the optimal design that maximizes Sibson's α-mutual information.
        
        Args:
            alpha: Misspecification order α ∈ (0, 1)
            
        Returns:
            Optimal number of participants to assign to group 0
        """
        designs = jnp.arange(self.likelihood.num_participants + 1)
        mi_values = jax.vmap(lambda d: self.sibson_mi(d, alpha))(designs)
        return int(jnp.argmax(mi_values))