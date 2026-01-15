"""
Conjugate Beta-Binomial A/B Testing for Robust Bayesian Experimental Design

This module implements a conjugate Beta-Binomial A/B testing model with support for:
- Standard Bayesian inference (prior, likelihood, posterior)
- Robust experimental design using Sibson's α-mutual information
- Worst-case generative processes under model misspecification
"""

from typing import NamedTuple, Tuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr

from jax.scipy.special import betaln, gammaln, digamma, logsumexp

from rdesign.distributions import Beta


class BetaBinomial(NamedTuple):
    """
    Beta-Binomial distribution
    Represents the marginal likelihood or predictive distribution.
    """
    trials: Array  # shape (2,) for [n_a, n_b]
    delta: Array   # shape (2,)
    gamma: Array   # shape (2,)

    def logpmf(self, x: Array) -> float:
        """
        Compute log probability mass function.
        """
        # Log Binomial coefficient
        log_binom = (
            gammaln(self.trials + 1)
            - gammaln(x + 1)
            - gammaln(self.trials - x + 1)
        )

        # Log Beta function ratio
        log_beta_ratio = (
            betaln(self.delta + x, self.gamma + self.trials - x) -
            betaln(self.delta, self.gamma)
        )

        # Sum over groups (independence)
        return jnp.sum(jnp.where(self.trials > 0, log_binom + log_beta_ratio, 0.0))

    def sample(self, rng_key: Array, shape: tuple = ()) -> Array:
        """Sample from Beta-Binomial via hierarchical sampling."""
        key_theta, key_x = jr.split(rng_key)
        # 1. Sample theta from Beta
        theta = jr.beta(key_theta, self.delta, self.gamma, shape=shape + (2,))
        # 2. Sample x from Binomial using theta
        return jr.binomial(key_x, self.trials.astype(int), theta)


class TiltedBetaBinomial(NamedTuple):
    """
    α-tilted Marginal distribution p_α(x | ξ).
    """
    trials: Array  # shape (2,)
    delta: Array   # shape (2,)
    gamma: Array   # shape (2,)
    alpha: float

    def _log_Z_alpha_k(self, x_k: Array, n_k: Array, delta_k: float, gamma_k: float) -> Array:
        """
        Compute unnormalized log probability term log Z_α(k, n) for a single group.
        """
        # Ensure we don't have NaNs by masking
        mask = (x_k <= n_k) & (x_k >= 0)
        safe_x = jnp.where(mask, x_k, 0.0)
        safe_n_minus_x = jnp.where(mask, n_k - x_k, 0.0)

        log_binom = (
            gammaln(n_k + 1)
            - gammaln(safe_x + 1)
            - gammaln(safe_n_minus_x + 1)
        )
        log_beta_term = (
            betaln(delta_k + self.alpha * safe_x, gamma_k + self.alpha * safe_n_minus_x) -
            betaln(delta_k, gamma_k)
        )
        res = self.alpha * log_binom + log_beta_term
        return jnp.where(mask, res, -jnp.inf)

    def log_pmf(self, x: Array) -> Array:
        """
        Compute normalized log p_α(x).
        """
        # Calculate unnormalized log probs for the specific x
        log_Z_a = self._log_Z_alpha_k(x[0], self.trials[0], self.delta[0], self.gamma[0])
        log_Z_b = self._log_Z_alpha_k(x[1], self.trials[1], self.delta[1], self.gamma[1])

        # Calculate partition functions Z_alpha(xi) for normalization
        # We must sum Z_alpha(k)^(1/alpha) over all possible k
        # For individual log_pmf we use int(n) as support
        norm_a = self._compute_log_normalizer(self.trials[0], self.delta[0], self.gamma[0])
        norm_b = self._compute_log_normalizer(self.trials[1], self.delta[1], self.gamma[1])

        # log p_alpha(x) = (1/alpha) * log Z_alpha(x) - log_normalizer
        log_prob_a = (1.0 / self.alpha) * log_Z_a - norm_a
        log_prob_b = (1.0 / self.alpha) * log_Z_b - norm_b

        return log_prob_a + log_prob_b

    def _compute_log_normalizer(self, n: Array, delta: float, gamma: float, max_n: int = None) -> Array:
        """Compute log sum_k Z_α(k)^(1/α)."""
        if max_n is None:
            # Fallback to concrete support if max_n is not provided
            # This will fail under vmap if n is a tracer
            if n <= 0: return jnp.array(0.0)
            n_int = int(n)
            k_vals = jnp.arange(n_int + 1, dtype=jnp.float32)
        else:
            # JAX-friendly version for vmap
            k_vals = jnp.arange(max_n + 1, dtype=jnp.float32)

        log_Z_vals = self._log_Z_alpha_k(k_vals, n, delta, gamma)
        return logsumexp(log_Z_vals / self.alpha)

    def sample(self, rng_key: Array) -> Array:
        """
        Sample from the discrete tilted marginal.
        """
        def sample_single_group(k, n, d, g):
            if n <= 0: return 0.0
            k_vals = jnp.arange(int(n) + 1, dtype=jnp.float32)
            # Unnormalized log probs in the tilted distribution: (1/alpha) * log Z_alpha
            logits = self._log_Z_alpha_k(k_vals, n, d, g) / self.alpha
            idx = jr.categorical(k, logits)
            return idx.astype(jnp.float32)

        key_a, key_b = jr.split(rng_key)
        x_a = sample_single_group(key_a, self.trials[0], self.delta[0], self.gamma[0])
        x_b = sample_single_group(key_b, self.trials[1], self.delta[1], self.gamma[1])
        return jnp.array([x_a, x_b])


class Binomial(NamedTuple):
    """
    Binomial likelihood for A/B testing.
    """
    num_participants: int

    def get_trials(self, design: Array) -> Array:
        """Get number of trials (n_a, n_b) for each group."""
        n_a = design
        n_b = self.num_participants - design
        return jnp.array([n_a, n_b], dtype=jnp.float32)


class ABTesting(NamedTuple):
    """
    Conjugate Beta-Binomial A/B Testing Model for Robust Experimental Design.

    Args:
        prior: Beta prior
        likelihood: Binomial
    """
    prior: Beta
    likelihood: Binomial

    # =========================================================================
    # Bayesian Inference
    # =========================================================================

    def get_bayes_posterior(self, x: Array, design: Array) -> Beta:
        """
        Compute Bayes posterior p(θ | x, ξ).

        Args:
            x: Observations [x_a, x_b], shape (2,)
            design: Number of participants in group A

        Returns:
            Beta posterior distribution
        """
        trials = self.likelihood.get_trials(design)
        x = jnp.asarray(x, dtype=jnp.float32)

        return Beta(
            delta=self.prior.delta + x,
            gamma=self.prior.gamma + trials - x
        )

    def get_prior_predictive(self, design: Array) -> BetaBinomial:
        """
        Compute prior predictive p(x | ξ).
        Returns a BetaBinomial distribution.
        """
        trials = self.likelihood.get_trials(design)
        return BetaBinomial(
            trials=trials,
            delta=self.prior.delta,
            gamma=self.prior.gamma
        )

    def get_posterior_predictive(self, posterior: Beta, design: Array) -> BetaBinomial:
        """
        Compute posterior predictive p(x_new | x, ξ, ξ_new).
        This is a BetaBinomial parameterized by the posterior parameters.
        """
        trials = self.likelihood.get_trials(design)
        return BetaBinomial(
            trials=trials,
            delta=posterior.delta,
            gamma=posterior.gamma
        )

    # =========================================================================
    # Robust Inference
    # =========================================================================

    def get_tilted_posterior(self, x: Array, design: Array, alpha: Array) -> Beta:
        """
        Compute α-tilted posterior q*(θ | x, ξ) ∝ p(θ) p(x | θ, ξ)^α.

        Args:
            x: Observations [x_a, x_b], shape (2,)
            design: Number of participants in group A
            alpha: Misspecification order α ∈ (0, 1]

        Returns:
            Beta α-tilted posterior distribution
        """
        trials = self.likelihood.get_trials(design)
        x = jnp.asarray(x, dtype=jnp.float32)

        return Beta(
            delta=self.prior.delta + alpha * x,
            gamma=self.prior.gamma + alpha * (trials - x)
        )

    def get_tilted_marginal(self, design: Array, alpha: Array) -> TiltedBetaBinomial:
        """
        Compute α-tilted marginal p_α(x | ξ).
        Returns a TiltedBetaBinomial distribution object.
        """
        trials = self.likelihood.get_trials(design)
        return TiltedBetaBinomial(
            trials=trials,
            delta=self.prior.delta,
            gamma=self.prior.gamma,
            alpha=alpha
        )

    def sample_tilted_joint(self, rng_key: Array, design: Array, alpha: Array) -> Tuple[Array, Array]:
        """
        Sample from the worst-case joint distribution q*(θ, x | ξ).

        1. Sample x ~ p_α(x | ξ) (Tilted Marginal)
        2. Sample θ ~ q*(θ | x, ξ) (Tilted Posterior)
        """
        key_x, key_theta = jr.split(rng_key)

        # 1. Sample x from tilted marginal
        marginal = self.get_tilted_marginal(design, alpha)
        x = marginal.sample(key_x)

        # 2. Sample theta from tilted posterior conditioned on x
        posterior = self.get_tilted_posterior(x, design, alpha)
        theta = posterior.sample(key_theta)

        return theta, x

    # =========================================================================
    # Information Measures
    # =========================================================================

    def shannon_mi(self, design: Array) -> Array:
        """
        Compute Shannon's mutual information I(θ; x | ξ).
        """
        trials = self.likelihood.get_trials(design)
        n_a, n_b = trials[0], trials[1]
        max_n = self.likelihood.num_participants

        def _group_shannon_mi(n, d, g):
            k_vals = jnp.arange(max_n + 1, dtype=jnp.float32)
            mask = (k_vals <= n) & (k_vals >= 0)

            # log p(x) for Beta-Binomial
            safe_n_minus_k = jnp.maximum(0, n - k_vals)
            log_px = (
                gammaln(n + 1) - gammaln(k_vals + 1) - gammaln(safe_n_minus_k + 1)
                + betaln(d + k_vals, g + safe_n_minus_k) - betaln(d, g)
            )
            px = jnp.where(mask, jnp.exp(log_px), 0.0)

            # D_KL( Beta(d+x, g+n-x) || Beta(d, g) )
            d_post = d + k_vals
            g_post = g + n - k_vals

            # Use safe values for the evaluation branch to avoid NaNs from digamma/betaln
            # for k_vals outside the support [0, n], which are then masked out.
            safe_d_post = jnp.where(mask, d_post, 1.0)
            safe_g_post = jnp.where(mask, g_post, 1.0)
            safe_total_post = safe_d_post + safe_g_post

            # KL divergence for Beta distributions
            kl = (
                betaln(d, g) - betaln(safe_d_post, safe_g_post) +
                (d_post - d) * (digamma(safe_d_post) - digamma(safe_total_post)) +
                (g_post - g) * (digamma(safe_g_post) - digamma(safe_total_post))
            )

            return jnp.sum(jnp.where(mask, px * kl, 0.0))

        mi_a = _group_shannon_mi(n_a, self.prior.delta[0], self.prior.gamma[0])
        mi_b = _group_shannon_mi(n_b, self.prior.delta[1], self.prior.gamma[1])
        return mi_a + mi_b

    def sibson_mi(self, design: Array, alpha: Array) -> Array:
        """
        Compute Sibson's α-mutual information I^S_α(θ; x | ξ).
        """
        # Handle alpha near 1.0 stably by returning Shannon MI
        is_shannon = jnp.abs(alpha - 1.0) < 1e-4

        def _sibson_mi():
            # Clip alpha to avoid exactly 1.0 and 0.0
            safe_alpha = jnp.clip(alpha, 1e-6, 1.0 - 1e-3)
            trials = self.likelihood.get_trials(design)
            n_a, n_b = trials[0], trials[1]
            max_n = self.likelihood.num_participants

            marginal = self.get_tilted_marginal(design, safe_alpha)

            log_norm_a = marginal._compute_log_normalizer(n_a, self.prior.delta[0], self.prior.gamma[0], max_n=max_n)
            log_norm_b = marginal._compute_log_normalizer(n_b, self.prior.delta[1], self.prior.gamma[1], max_n=max_n)

            return safe_alpha / (safe_alpha - 1.0) * (log_norm_a + log_norm_b)

        return jnp.where(is_shannon, self.shannon_mi(design), _sibson_mi())

    def shannon_ig(self, x: Array, design: Array) -> Array:
        """
        Compute information gain for a specific observation.

        Args:
            x: Specific observation [x_a, x_b], shape (2,)
            design: Number of participants in group A

        Returns:
            Information gain in nats
        """
        posterior = self.get_bayes_posterior(x, design)
        return posterior.kl_to(self.prior)

    def renyi_ig(self, x: Array, design: Array, alpha: Array) -> Array:
        """
        Compute robust information gain G_α(x, ξ).

        Args:
            x: Specific observation [x_a, x_b], shape (2,)
            design: Number of participants in group A
            alpha: Misspecification order α ∈ (0, 1]

        Returns:
            Robust information gain in nats
        """
        posterior = self.get_tilted_posterior(x, design, alpha)
        return posterior.renyi_to(self.prior, alpha)

    # =========================================================================
    # Design Optimization
    # =========================================================================

    def optimal_design_shannon(self) -> Array:
        """Find design that maximizes Shannon MI."""
        designs = jnp.arange(self.likelihood.num_participants + 1)
        mi_values = jax.vmap(self.shannon_mi)(designs)
        return jnp.argmax(mi_values)

    def optimal_design_sibson(self, alpha: Array) -> Array:
        """Find design that maximizes Sibson MI."""
        designs = jnp.arange(self.likelihood.num_participants + 1)
        mi_values = jax.vmap(lambda d: self.sibson_mi(d, alpha))(designs)
        return jnp.argmax(mi_values)
