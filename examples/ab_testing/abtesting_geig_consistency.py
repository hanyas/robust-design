"""
Consistency check for power-likelihood Gibbs-EIG in A/B testing.

This example compares two design objectives for the same tempering level ω:

1. Exact power-GEIG, built from the unnormalized power likelihood p(y | θ, ξ)^ω
2. Ordinary EIG under the properly normalized tilted model
       q_ω(y | θ, ξ) ∝ p(y | θ, ξ)^ω

If power-GEIG were consistent for the normalized tilted model, these two curves
would induce the same optimal allocation. In the Beta-Binomial A/B setting they
need not agree because the normalization factor depends on the design.
"""

import csv
from typing import Tuple

import numpy as np

import jax
from jax import Array
import jax.numpy as jnp
from jax.scipy.special import betaln, digamma, gammaln, logsumexp

import matplotlib.pyplot as plt

from rdesign.distributions import Beta
from rdesign.ab_testing import ABTesting, Binomial

jax.config.update("jax_enable_x64", True)


def normalize_curve(values: Array) -> Array:
    """Normalize a curve to [0, 1] for shape-only comparisons."""
    shifted = values - jnp.min(values)
    scale = jnp.max(shifted)
    return jnp.where(scale > 0.0, shifted / scale, shifted)


def gauss_legendre_unit_interval(num_points: int) -> Tuple[Array, Array]:
    """Quadrature nodes and weights on [0, 1]."""
    nodes, weights = np.polynomial.legendre.leggauss(num_points)
    return (
        jnp.asarray(0.5 * (nodes + 1.0), dtype=jnp.float64),
        jnp.asarray(0.5 * weights, dtype=jnp.float64),
    )


def power_geig_group(n: Array, delta: Array, gamma: Array, omega: float, max_n: int) -> Tuple[Array, Array]:
    """
    Exact one-arm power-likelihood Gibbs-EIG ingredients.

    Returns:
        group_gain: sum_k m_omega(k) [E_q[ω log p(k|θ)] - log m_omega(k)]
        group_mass: sum_k m_omega(k)
    """
    k_vals = jnp.arange(max_n + 1, dtype=jnp.float64)
    mask = (k_vals >= 0) & (k_vals <= n)

    safe_k = jnp.where(mask, k_vals, 0.0)
    safe_n_minus_k = jnp.where(mask, n - safe_k, 0.0)

    log_binom = (
        gammaln(n + 1.0)
        - gammaln(safe_k + 1.0)
        - gammaln(safe_n_minus_k + 1.0)
    )

    a = delta + omega * safe_k
    b = gamma + omega * safe_n_minus_k

    safe_log_m = (
        omega * log_binom
        + betaln(a, b)
        - betaln(delta, gamma)
    )
    safe_log_m = jnp.where(mask, safe_log_m, 0.0)

    m_vals = jnp.where(mask, jnp.exp(safe_log_m), 0.0)

    expected_log_like = (
        omega * log_binom
        + omega * safe_k * (digamma(a) - digamma(a + b))
        + omega * safe_n_minus_k * (digamma(b) - digamma(a + b))
    )
    expected_log_like = jnp.where(mask, expected_log_like, 0.0)

    group_gain = jnp.sum(jnp.where(mask, m_vals * (expected_log_like - safe_log_m), 0.0))
    group_mass = jnp.sum(m_vals)
    return group_gain, group_mass


def power_geig(model: ABTesting, design: Array, omega: float) -> Array:
    """Exact power-GEIG for the two-arm Beta-Binomial model."""
    trials = model.likelihood.get_trials(design)
    max_n = model.likelihood.num_participants

    gain_a, mass_a = power_geig_group(
        trials[0], model.prior.delta[0], model.prior.gamma[0], omega, max_n
    )
    gain_b, mass_b = power_geig_group(
        trials[1], model.prior.delta[1], model.prior.gamma[1], omega, max_n
    )
    return gain_a * mass_b + mass_a * gain_b


def beta_quadrature_weights(theta_nodes: Array, quad_weights: Array, delta: Array, gamma: Array) -> Array:
    """Quadrature weights for expectations under Beta(delta, gamma)."""
    log_prior = (
        (delta - 1.0) * jnp.log(theta_nodes)
        + (gamma - 1.0) * jnp.log1p(-theta_nodes)
        - betaln(delta, gamma)
    )
    weights = quad_weights * jnp.exp(log_prior)
    return weights / jnp.sum(weights)


def normalized_tilted_eig_group(
    n: Array,
    delta: Array,
    gamma: Array,
    omega: float,
    theta_nodes: Array,
    quad_weights: Array,
    max_n: int,
) -> Array:
    """
    Ordinary EIG under the properly normalized tilted model q_ω for one arm.

    q_ω(k | θ, n) ∝ Binomial(k; n, θ)^ω

    The expectation over θ ~ Beta(delta, gamma) is approximated by
    Gauss-Legendre quadrature on [0, 1].
    """
    prior_weights = beta_quadrature_weights(theta_nodes, quad_weights, delta, gamma)

    k_vals = jnp.arange(max_n + 1, dtype=jnp.float64)
    mask = (k_vals >= 0) & (k_vals <= n)

    safe_k = jnp.where(mask, k_vals, 0.0)
    safe_n_minus_k = jnp.where(mask, n - safe_k, 0.0)

    log_binom = (
        gammaln(n + 1.0)
        - gammaln(safe_k + 1.0)
        - gammaln(safe_n_minus_k + 1.0)
    )

    theta = theta_nodes[:, None]
    log_p = (
        log_binom[None, :]
        + safe_k[None, :] * jnp.log(theta)
        + safe_n_minus_k[None, :] * jnp.log1p(-theta)
    )
    log_q_unnorm = jnp.where(mask[None, :], omega * log_p, -jnp.inf)

    log_normalizer = logsumexp(log_q_unnorm, axis=1, keepdims=True)
    log_q = log_q_unnorm - log_normalizer
    q_vals = jnp.where(mask[None, :], jnp.exp(log_q), 0.0)

    marginal_q = prior_weights @ q_vals
    safe_log_marginal_q = jnp.where(
        mask,
        jnp.log(jnp.clip(marginal_q, 1e-300)),
        0.0,
    )
    safe_log_q = jnp.where(mask[None, :], log_q, 0.0)

    return jnp.sum(
        prior_weights[:, None] * q_vals * (safe_log_q - safe_log_marginal_q[None, :])
    )


def normalized_tilted_eig(
    model: ABTesting,
    design: Array,
    omega: float,
    theta_nodes: Array,
    quad_weights: Array,
) -> Array:
    """
    Ordinary EIG under the normalized tilted model q_ω for the two-arm problem.

    Because both the prior and q_ω factorize across arms, the total EIG is the
    sum of the two one-arm EIG terms.
    """
    trials = model.likelihood.get_trials(design)
    max_n = model.likelihood.num_participants

    eig_a = normalized_tilted_eig_group(
        trials[0],
        model.prior.delta[0],
        model.prior.gamma[0],
        omega,
        theta_nodes,
        quad_weights,
        max_n,
    )
    eig_b = normalized_tilted_eig_group(
        trials[1],
        model.prior.delta[1],
        model.prior.gamma[1],
        omega,
        theta_nodes,
        quad_weights,
        max_n,
    )
    return eig_a + eig_b


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    num_participants = 50
    omega = 5e-1
    num_quad_points = 80

    # A prior setting where the GEIG and EIG(q_ω) optima visibly differ.
    prior_delta = jnp.array([1.0, 10.0])
    prior_gamma = jnp.array([5.0, 10.0])

    prior = Beta(delta=prior_delta, gamma=prior_gamma)
    likelihood = Binomial(num_participants=num_participants)
    model = ABTesting(prior=prior, likelihood=likelihood)

    designs = jnp.arange(num_participants + 1)
    theta_nodes, quad_weights = gauss_legendre_unit_interval(num_quad_points)

    # =========================================================================
    # OBJECTIVE CURVES
    # =========================================================================
    print("=" * 80)
    print("POWER-GEIG CONSISTENCY CHECK: A/B TESTING")
    print("=" * 80)
    print(f"Participants:              {num_participants}")
    print(f"Power level ω:            {omega:.2f}")
    print(f"Quadrature points:        {num_quad_points}")
    print("Prior")
    print(f"  Group A: Beta({prior_delta[0]:.0f}, {prior_gamma[0]:.0f})")
    print(f"  Group B: Beta({prior_delta[1]:.0f}, {prior_gamma[1]:.0f})")
    print()
    print("Computing objective curves...")

    power_geig_vals = jnp.array(
        [power_geig(model, int(design), omega) for design in designs.tolist()]
    )
    tilted_eig_vals = jnp.array(
        [
            normalized_tilted_eig(model, int(design), omega, theta_nodes, quad_weights)
            for design in designs.tolist()
        ]
    )

    design_power_geig = int(jnp.argmax(power_geig_vals))
    design_tilted_eig = int(jnp.argmax(tilted_eig_vals))

    tilted_eig_opt = float(tilted_eig_vals[design_tilted_eig])
    tilted_eig_at_geig = float(tilted_eig_vals[design_power_geig])
    tilted_eig_gap = tilted_eig_opt - tilted_eig_at_geig

    power_geig_opt = float(power_geig_vals[design_power_geig])
    power_geig_at_tilted = float(power_geig_vals[design_tilted_eig])
    power_geig_gap = power_geig_opt - power_geig_at_tilted

    print("Selected designs")
    print(f"  Power-GEIG optimum:          n_A = {design_power_geig}")
    print(f"  EIG(q_ω) optimum:            n_A = {design_tilted_eig}")
    print(f"  Allocation gap:              {abs(design_power_geig - design_tilted_eig)}")
    print()
    print("Cross-evaluation")
    print(f"  EIG(q_ω) at GEIG design:     {tilted_eig_at_geig:.6f}")
    print(f"  EIG(q_ω) optimum:            {tilted_eig_opt:.6f}")
    print(f"  EIG(q_ω) loss from GEIG:     {tilted_eig_gap:.6f}")
    print(f"  Relative loss:               {100 * tilted_eig_gap / tilted_eig_opt:.2f}%")
    print()
    print(f"  GEIG at EIG(q_ω) design:     {power_geig_at_tilted:.6f}")
    print(f"  GEIG optimum:                {power_geig_opt:.6f}")
    print(f"  GEIG loss from EIG(q_ω):     {power_geig_gap:.6f}")

    # =========================================================================
    # SAVE DATA
    # =========================================================================
    csv_path = "examples/ab_testing/data/abtesting_geig_consistency.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "n_A",
            "power_geig",
            "tilted_eig",
            "power_geig_norm",
            "tilted_eig_norm",
            "tilted_eig_suboptimality",
            "is_power_geig_opt",
            "is_tilted_eig_opt",
        ])

        power_geig_norm = normalize_curve(power_geig_vals)
        tilted_eig_norm = normalize_curve(tilted_eig_vals)
        tilted_eig_suboptimality = jnp.max(tilted_eig_vals) - tilted_eig_vals

        for idx, n_a in enumerate(designs.tolist()):
            writer.writerow([
                n_a,
                float(power_geig_vals[idx]),
                float(tilted_eig_vals[idx]),
                float(power_geig_norm[idx]),
                float(tilted_eig_norm[idx]),
                float(tilted_eig_suboptimality[idx]),
                int(idx == design_power_geig),
                int(idx == design_tilted_eig),
            ])

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = {
        "Power-GEIG": "#B04A1F",
        "Tilted-EIG": "#111111",
    }

    # 1. Compare the objective shapes directly
    axes[0].plot(
        designs,
        normalize_curve(power_geig_vals),
        color=colors["Power-GEIG"],
        lw=2.5,
        label=f"Power-GEIG (ω={omega:.2f})",
    )
    axes[0].plot(
        designs,
        normalize_curve(tilted_eig_vals),
        color=colors["Tilted-EIG"],
        lw=2.5,
        label=f"EIG($q_\\omega$), ω={omega:.2f}",
    )
    axes[0].axvline(design_power_geig, color=colors["Power-GEIG"], ls="--", lw=1.5)
    axes[0].axvline(design_tilted_eig, color=colors["Tilted-EIG"], ls="--", lw=1.5)
    axes[0].set_title("Objective Curves")
    axes[0].set_xlabel("$n_A$")
    axes[0].set_ylabel("Normalized utility")
    axes[0].set_xlim(0, num_participants)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=9)

    # 2. Show how suboptimal the GEIG choice is under the true normalized model
    tilted_eig_suboptimality = jnp.max(tilted_eig_vals) - tilted_eig_vals
    axes[1].plot(
        designs,
        tilted_eig_suboptimality,
        color=colors["Tilted-EIG"],
        lw=2.5,
    )
    axes[1].scatter(
        [design_power_geig, design_tilted_eig],
        [
            float(tilted_eig_suboptimality[design_power_geig]),
            float(tilted_eig_suboptimality[design_tilted_eig]),
        ],
        color=[colors["Power-GEIG"], colors["Tilted-EIG"]],
        s=60,
        zorder=3,
    )
    axes[1].axvline(design_power_geig, color=colors["Power-GEIG"], ls="--", lw=1.5)
    axes[1].axvline(design_tilted_eig, color=colors["Tilted-EIG"], ls="--", lw=1.5)
    axes[1].set_title("EIG($q_\\omega$) Suboptimality")
    axes[1].set_xlabel("$n_A$")
    axes[1].set_ylabel("Gap to optimal EIG($q_\\omega$)")
    axes[1].set_xlim(0, num_participants)
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()
