"""
Posterior predictive performance under an alpha-tilted Binomial true process.

This benchmark keeps the same tilted true process as abtesting_elpd_titled.py,
but compares two design-selection rules for the same robustness order alpha:
"""
from typing import NamedTuple

import csv

import numpy as np

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln

import matplotlib.pyplot as plt

from rdesign.distributions import Beta
from rdesign.ab_testing import ABTesting, Binomial

jax.config.update("jax_enable_x64", True)


class TiltedBinomial(NamedTuple):
    """
    Generates data from the alpha-tilted Binomial likelihood p_alpha(x | theta).

    Args:
        num_participants: Total participants (defines the support size)
        alpha: Tilting parameter
    """
    num_participants: int
    alpha: float = 1.0

    def get_trials(self, design: Array) -> Array:
        """Get trial counts [n_A, n_B] from design (n_A)."""
        n_a = design
        n_b = self.num_participants - design
        return jnp.array([n_a, n_b], dtype=jnp.float32)

    def _log_tilted_prob(self, k: Array, n: Array, theta: Array) -> Array:
        """Compute the unnormalized tilted log probability."""
        mask = (k >= 0) & (k <= n)
        safe_k = jnp.where(mask, k, 0.0)
        safe_nmk = jnp.where(mask, n - safe_k, 0.0)

        log_comb = gammaln(n + 1) - gammaln(safe_k + 1) - gammaln(safe_nmk + 1)

        theta = jnp.clip(theta, 1e-6, 1.0 - 1e-6)
        log_lklhd = safe_k * jnp.log(theta) + safe_nmk * jnp.log(1.0 - theta)

        return jnp.where(mask, self.alpha * (log_comb + log_lklhd), -jnp.inf)

    def sample(self, rng_key: Array, design: Array, theta_true: Array) -> Array:
        """Sample observations x from the tilted likelihood."""
        trials = self.get_trials(design)
        k_grid = jnp.arange(self.num_participants + 1, dtype=jnp.float32)

        def sample_group(key, n, theta):
            logits = self._log_tilted_prob(k_grid, n, theta)
            return jr.categorical(key, logits).astype(jnp.float32)

        keys = jr.split(rng_key, 2)
        return jax.vmap(sample_group)(keys, trials, theta_true)


def generate_uniform_design(rng_key: Array, num_participants: int) -> Array:
    """Uniform random design in {0, ..., num_participants}."""
    return jr.randint(rng_key, shape=(), minval=0, maxval=num_participants + 1)


def custom_split(rng_key: Array, num_splits: int) -> tuple[Array, Array]:
    """Split RNG key into num_splits parts."""
    keys = jr.split(rng_key, num_splits)
    return keys[0], keys[1:]


def build_alpha_grid() -> Array:
    """Log-spaced robustness orders from 1e-2 to 1."""
    return jnp.asarray(np.geomspace(1e-2, 1.0, 21), dtype=jnp.float64)


def summarize_elpd(elpd_values: Array, num_trials: int) -> tuple[Array, Array, Array, Array]:
    """Mean, standard error, and central 90% interval across trials."""
    mean = jnp.mean(elpd_values, axis=1)
    std = jnp.std(elpd_values, axis=1)
    se = std / jnp.sqrt(num_trials)
    q05 = jnp.percentile(elpd_values, 5, axis=1)
    q95 = jnp.percentile(elpd_values, 95, axis=1)
    return mean, se, q05, q95


if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    rng_key = jr.PRNGKey(42)

    # Experiment settings
    num_participants = 1000
    num_trials = int(5 * 1024)
    num_test_points = 32

    alpha_values = build_alpha_grid()
    num_alphas = len(alpha_values)

    # True Process (Tilted Binomial)
    alpha_true = 1e-1
    proc_prior = Beta(
        delta=jnp.array([1.0, 500.0]),
        gamma=jnp.array([1.0, 500.0]),
    )
    proc_lklhd = TiltedBinomial(
        num_participants=num_participants,
        alpha=alpha_true,
    )

    # Model (Beta-Binomial)
    mdl_prior = Beta(
        delta=jnp.array([1.0, 5.0]),
        gamma=jnp.array([1.0, 5.0]),
    )
    mdl_lklhd = Binomial(num_participants=num_participants)
    model = ABTesting(prior=mdl_prior, likelihood=mdl_lklhd)

    print("=" * 100)
    print("PREDICTIVE PERFORMANCE EXPERIMENT UNDER TILTED TRUE PROCESS: A/B TESTING")
    print("=" * 100)
    print("TRUE PROCESS: alpha-tilted Binomial")
    print(f"  Participants:    {num_participants}")
    print(f"  True alpha:      {alpha_true:.2f}")
    print("\nMODEL: Beta-Binomial")
    print("  Robust learning:  q(θ|x) ∝ p(θ) p(x|θ)^α")
    print("Comparison rule")
    print("  For each α, compare the ELPD induced by:")
    print("    1. the Sibson-selected design")
    print("    2. the Power-GEIG-selected design")
    print(f"\nAlpha values:      {num_alphas}")
    print(f"Test points:       {num_test_points}")
    print(f"Number of trials:  {num_trials}")
    print("=" * 100)

    # ============================================================================
    # OPTIMAL DESIGNS PER ALPHA
    # ============================================================================
    print("\n" + "=" * 100)
    print("FINDING SIBSON AND POWER-GEIG DESIGNS")
    print("=" * 100)

    designs = jnp.arange(num_participants + 1)
    sibson_matrix = jax.vmap(
        lambda alpha: jax.vmap(lambda d: model.sibson_mi(d, alpha))(designs)
    )(alpha_values)
    geig_matrix = jax.vmap(
        lambda alpha: jax.vmap(lambda d: model.power_geig(d, alpha))(designs)
    )(alpha_values)

    sibson_designs = jnp.argmax(sibson_matrix, axis=1)
    geig_designs = jnp.argmax(geig_matrix, axis=1)
    allocation_gaps = jnp.abs(sibson_designs - geig_designs)

    for alpha_target in [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]:
        idx = jnp.argmin(jnp.abs(alpha_values - alpha_target))
        print(
            f"  α = {float(alpha_values[idx]):.4f}: "
            f"Sibson n_A = {int(sibson_designs[idx])}, "
            f"GEIG n_A = {int(geig_designs[idx])}, "
            f"gap = {int(allocation_gaps[idx])}"
        )

    # ============================================================================
    # RUN EXPERIMENTS
    # ============================================================================
    print("\n" + "=" * 100)
    print("RUNNING ELPD COMPARISON")
    print("=" * 100)

    @jax.jit
    def evaluate_all_alphas(
        alpha_vals: Array,
        sibson_design_vals: Array,
        geig_design_vals: Array,
        x_sibson: Array,
        x_geig: Array,
        ds_test: Array,
        xs_test: Array,
    ) -> tuple[Array, Array]:
        """Evaluate held-out ELPD for both design rules across all alpha values."""

        def evaluate_alpha(alpha, design_sibson, design_geig, x_sibson_val, x_geig_val):
            posterior_sibson = model.get_tilted_posterior(x_sibson_val, design_sibson, alpha)
            log_preds_sibson = jax.vmap(
                lambda d, x: model.get_posterior_predictive(
                    posterior_sibson, d, alpha=alpha
                ).logpmf(x)
            )(ds_test, xs_test)

            posterior_geig = model.get_tilted_posterior(x_geig_val, design_geig, alpha)
            log_preds_geig = jax.vmap(
                lambda d, x: model.get_posterior_predictive(
                    posterior_geig, d, alpha=alpha
                ).logpmf(x)
            )(ds_test, xs_test)

            return jnp.mean(log_preds_sibson), jnp.mean(log_preds_geig)

        return jax.vmap(evaluate_alpha)(
            alpha_vals,
            sibson_design_vals,
            geig_design_vals,
            x_sibson,
            x_geig,
        )

    elpd_sibson = jnp.zeros((num_alphas, num_trials))
    elpd_geig = jnp.zeros((num_alphas, num_trials))

    print(f"\nRunning {num_trials} trials...")

    for trial in range(num_trials):
        if (trial + 1) % 200 == 0 or trial == 0:
            print(f"[{trial + 1:4d}/{num_trials}]  Evaluating...")

        rng_key, sub_key = jr.split(rng_key)
        theta_true = proc_prior.sample(sub_key, 1)[0]

        rng_key, sibson_keys = custom_split(rng_key, num_alphas + 1)
        x_sibson = jax.vmap(lambda k, d: proc_lklhd.sample(k, d, theta_true))(
            sibson_keys, sibson_designs
        )

        rng_key, geig_keys = custom_split(rng_key, num_alphas + 1)
        x_geig = jax.vmap(lambda k, d: proc_lklhd.sample(k, d, theta_true))(
            geig_keys, geig_designs
        )

        rng_key, test_design_keys = custom_split(rng_key, num_test_points + 1)
        designs_test = jax.vmap(
            lambda k: generate_uniform_design(k, num_participants)
        )(test_design_keys)

        rng_key, test_obs_keys = custom_split(rng_key, num_test_points + 1)
        x_test = jax.vmap(lambda k, d: proc_lklhd.sample(k, d, theta_true))(
            test_obs_keys, designs_test
        )

        elpds_sibson, elpds_geig = evaluate_all_alphas(
            alpha_values,
            sibson_designs,
            geig_designs,
            x_sibson,
            x_geig,
            designs_test,
            x_test,
        )

        elpd_sibson = elpd_sibson.at[:, trial].set(elpds_sibson)
        elpd_geig = elpd_geig.at[:, trial].set(elpds_geig)

    print("✓ Complete")

    # ============================================================================
    # ANALYSIS
    # ============================================================================
    print("\n" + "=" * 100)
    print("RESULTS: ELPD BY DESIGN SELECTION RULE")
    print("=" * 100)

    mean_elpd_sibson, se_elpd_sibson, q05_sibson, q95_sibson = summarize_elpd(
        elpd_sibson, num_trials
    )
    mean_elpd_geig, se_elpd_geig, q05_geig, q95_geig = summarize_elpd(
        elpd_geig, num_trials
    )
    geig_minus_sibson = mean_elpd_geig - mean_elpd_sibson

    print(
        f"\n{'α':>8} | {'Sibson n_A':>10} | {'GEIG n_A':>8} | "
        f"{'Sibson ELPD':>20} | {'GEIG ELPD':>20} | {'GEIG - Sibson':>14}"
    )
    print("-" * 95)
    for alpha_target in [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]:
        idx = jnp.argmin(jnp.abs(alpha_values - alpha_target))
        print(
            f"{float(alpha_values[idx]):>8.3f} | "
            f"{int(sibson_designs[idx]):>10d} | "
            f"{int(geig_designs[idx]):>8d} | "
            f"{float(mean_elpd_sibson[idx]):>10.4f} ± {float(se_elpd_sibson[idx]):<7.4f} | "
            f"{float(mean_elpd_geig[idx]):>10.4f} ± {float(se_elpd_geig[idx]):<7.4f} | "
            f"{float(geig_minus_sibson[idx]):>14.4f}"
        )

    geig_wins = int(jnp.sum(geig_minus_sibson > 0.0))
    sibson_wins = int(jnp.sum(geig_minus_sibson < 0.0))
    ties = int(num_alphas - geig_wins - sibson_wins)
    print("\nSummary")
    print(f"  GEIG higher mean ELPD at:    {geig_wins} alpha values")
    print(f"  Sibson higher mean ELPD at:  {sibson_wins} alpha values")
    print(f"  Equal mean ELPD at:          {ties} alpha values")

    # ============================================================================
    # CSV + VISUALIZATION
    # ============================================================================
    csv_path = "examples/ab_testing/data/abtesting_elpd_titled_geig.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "alpha",
            "sibson_design",
            "geig_design",
            "allocation_gap",
            "mean_elpd_sibson",
            "se_elpd_sibson",
            "q05_elpd_sibson",
            "q95_elpd_sibson",
            "mean_elpd_geig",
            "se_elpd_geig",
            "q05_elpd_geig",
            "q95_elpd_geig",
            "geig_minus_sibson",
        ])
        for idx in range(num_alphas):
            writer.writerow([
                float(alpha_values[idx]),
                int(sibson_designs[idx]),
                int(geig_designs[idx]),
                int(allocation_gaps[idx]),
                float(mean_elpd_sibson[idx]),
                float(se_elpd_sibson[idx]),
                float(q05_sibson[idx]),
                float(q95_sibson[idx]),
                float(mean_elpd_geig[idx]),
                float(se_elpd_geig[idx]),
                float(q05_geig[idx]),
                float(q95_geig[idx]),
                float(geig_minus_sibson[idx]),
            ])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(alpha_values, mean_elpd_sibson, "-", color="0.15", lw=2.5, label="Sibson design")
    axes[0].fill_between(alpha_values, q05_sibson, q95_sibson, alpha=0.2, color="0.15")
    axes[0].plot(alpha_values, mean_elpd_geig, "-", color="0.45", lw=2.5, label="Power-GEIG design")
    axes[0].fill_between(alpha_values, q05_geig, q95_geig, alpha=0.2, color="0.45")
    axes[0].axvline(1.0, color="gray", linestyle=":", lw=1.5, label="Nominal (α=1)")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("ELPD", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].plot(alpha_values, sibson_designs, "-", color="0.15", lw=2.5, label="Sibson choice")
    axes[1].plot(alpha_values, geig_designs, "-", color="0.45", lw=2.5, label="Power-GEIG choice")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("α", fontsize=12)
    axes[1].set_ylabel("Chosen n_A", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
