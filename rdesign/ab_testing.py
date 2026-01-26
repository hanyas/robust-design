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

        # Sum over groups
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
    Optimized for JAX with vectorization over groups.
    """
    trials: Array  # shape (2,)
    delta: Array   # shape (2,)
    gamma: Array   # shape (2,)
    alpha: float
    max_n: int

    def _log_tilted_kernel(self, k: Array, n: Array, d: Array, g: Array) -> Array:
        """
        Computes log[ (Binom(n, k)^α * Beta(...) / Beta(...))^(1/α) ].
        This simplifies to: log_binom(n, k) + (1/α) * log_beta_ratio
        """
        # Safe masking for support
        mask = (k >= 0) & (k <= n)
        safe_k = jnp.where(mask, k, 0.0)
        safe_nmk = jnp.where(mask, n - safe_k, 0.0)

        # Log Binomial term
        log_binom = gammaln(n + 1) - gammaln(safe_k + 1) - gammaln(safe_nmk + 1)

        # Log Beta ratio term
        # Paper Appendix C.2: B(δ + αx, γ + α(n-x)) / B(δ, γ)
        log_beta_ratio = (
            betaln(d + self.alpha * safe_k, g + self.alpha * safe_nmk) -
            betaln(d, g)
        )

        # Combine: log(Z^(1/α)) = log(Z) / α
        # = (α * log_binom + log_beta_ratio) / α
        # = log_binom + log_beta_ratio / α
        return jnp.where(mask, log_binom + (log_beta_ratio / self.alpha), -jnp.inf)

    def log_normalizer(self, n: Array, d: Array, g: Array) -> Array:
            """
            Compute log Z_α = log sum_k (p(k|...)^α)^(1/α).
            Public method required for Sibson MI.
            """
            # 1. Create grid of all possible outcomes [0...max_n]
            k_grid = jnp.arange(self.max_n + 1, dtype=jnp.float32)

            # 2. Compute unnormalized kernel for every k
            # We broadcast n, d, g against the grid k
            log_kernel = self._log_tilted_kernel(k_grid, n, d, g)

            # 3. Sum over k
            return logsumexp(log_kernel)

    def logpmf(self, x: Array) -> Array:
        """Compute normalized log p_α(x)."""

        def standard_logpmf(val):
            # Standard Beta-Binomial
            log_binom = gammaln(self.trials + 1) - gammaln(val + 1) - gammaln(self.trials - val + 1)
            log_beta = betaln(self.delta + val, self.gamma + self.trials - val) - betaln(self.delta, self.gamma)
            return jnp.sum(jnp.where(self.trials > 0, log_binom + log_beta, 0.0))

        def tilted_logpmf(val):
            # 1. Compute unnormalized log-probs for the specific x
            # vmap over groups (trials, delta, gamma) and input x
            log_unnorm = jax.vmap(self._log_tilted_kernel)(val, self.trials, self.delta, self.gamma)

            # 2. Compute log-normalizers
            # Grid of all possible outcomes [0...max_n]
            k_grid = jnp.arange(self.max_n + 1, dtype=jnp.float32)

            # vmap grid computation over groups
            # Result shape: (2, max_n+1)
            log_grid = jax.vmap(lambda n, d, g: self._log_tilted_kernel(k_grid, n, d, g))(
                self.trials, self.delta, self.gamma
            )

            log_Z = logsumexp(log_grid, axis=-1)

            return jnp.sum(log_unnorm - log_Z)

        # Branch for stability near alpha=1
        return jax.lax.cond(jnp.abs(self.alpha - 1.0) < 1e-3, standard_logpmf, tilted_logpmf, x)

    def sample(self, rng_key: Array) -> Array:
        """Sample from the discrete tilted marginal."""

        def standard_sample(k):
            key_theta, key_x = jr.split(k)
            theta = jr.beta(key_theta, self.delta, self.gamma)
            return jr.binomial(key_x, self.trials.astype(int), theta)

        def tilted_sample(k):
            k_grid = jnp.arange(self.max_n + 1, dtype=jnp.float32)

            # Helper to sample one group
            def sample_group(key, n, d, g):
                logits = self._log_tilted_kernel(k_grid, n, d, g)
                return jr.categorical(key, logits).astype(jnp.float32)

            # vmap sampling over groups
            keys = jr.split(k, 2)
            return jax.vmap(sample_group)(keys, self.trials, self.delta, self.gamma)

        return jax.lax.cond(jnp.abs(self.alpha - 1.0) < 1e-3, standard_sample, tilted_sample, rng_key)


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

    def sample(self, rng_key: Array, design: Array, theta: Array) -> Array:
        """Sample from Binomial(n, θ)."""
        trials = self.get_trials(design)
        return jr.binomial(rng_key, trials.astype(int), theta)

    def logpdf(self, x: Array, design: Array, theta: Array) -> Array:
        """Compute log p(x | design, θ)."""
        trials = self.get_trials(design)
        return jnp.sum(jax.scipy.stats.binom.logpmf(x, trials, theta))


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

    def get_marginal(self, design: Array) -> BetaBinomial:
        """
        Compute marginal p(x | ξ).
        Returns a BetaBinomial distribution.
        """
        trials = self.likelihood.get_trials(design)
        return BetaBinomial(
            trials=trials,
            delta=self.prior.delta,
            gamma=self.prior.gamma
        )

    def get_posterior_predictive(self, posterior: Beta, design: Array, alpha: float = 1.0) -> BetaBinomial:
        """
        Compute posterior predictive p(x_new | x, ξ, ξ_new).
        This is a BetaBinomial parameterized by the posterior parameters.
        """
        trials = self.likelihood.get_trials(design)
        return TiltedBetaBinomial(
            trials=trials,
            delta=posterior.delta,
            gamma=posterior.gamma,
            alpha=alpha,
            max_n=self.likelihood.num_participants
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
            alpha=alpha,
            max_n=self.likelihood.num_participants
        )

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

            # Use safe values to avoid NaNs from digamma/betaln
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

        trials = self.likelihood.get_trials(design)
        n_a, n_b = trials[0], trials[1]

        # 1. Define the log-normalizer as a pure function of alpha
        def log_normalizer_fn(a):
            # Re-instantiate marginal with dynamic alpha to trace gradients
            marginal = self.get_tilted_marginal(design, a)

            # Compute log Z for both groups
            lz_a = marginal.log_normalizer(n_a, self.prior.delta[0], self.prior.gamma[0])
            lz_b = marginal.log_normalizer(n_b, self.prior.delta[1], self.prior.gamma[1])
            return lz_a + lz_b

        # 2. Define the exact Shannon MI as the gradient at 1.0
        def shannon_via_grad():
            return jax.grad(log_normalizer_fn)(1.0)

        # 3. Define the standard Sibson calculation
        def sibson_standard():
            # Clip safe_alpha to avoid division by zero in the non-limit branch
            safe_alpha = jnp.where(jnp.isclose(alpha, 1.0), 0.5, alpha)
            log_z = log_normalizer_fn(safe_alpha)
            return (safe_alpha / (safe_alpha - 1.0)) * log_z

        # 4. Switch based on proximity to 1.0
        is_shannon = jnp.isclose(alpha, 1.0, atol=1e-3)

        return jax.lax.cond(
            is_shannon,
            lambda _: shannon_via_grad(),
            lambda _: sibson_standard(),
            operand=None
        )

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
        posterior = self.get_bayes_posterior(x, design)
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
