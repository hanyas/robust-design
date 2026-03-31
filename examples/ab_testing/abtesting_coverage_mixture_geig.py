"""
Posterior Calibration Assessment for Misspecified A/B Testing

This variant matches abtesting_coverage_mixture.py except that the robust
design is chosen with power-GEIG rather than Sibson MI.
"""
from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import betainc

import csv
import matplotlib.pyplot as plt

from rdesign.distributions import Beta
from rdesign.ab_testing import ABTesting, Binomial

jax.config.update("jax_enable_x64", True)


class BetaMixtureBinomial(NamedTuple):
    """
    Binomial likelihood with an always-on latent rate mixture.

    For each group, we first sample p ~ Beta(kappa * theta, kappa * (1 - theta))
    and then draw x ~ Binomial(n, p). The mean remains theta, while smaller
    kappa values induce stronger overdispersion and more extreme counts.
    """
    num_participants: int
    mixture_concentration: float = 10.0

    def get_trials(self, design: Array) -> Array:
        """Get trial counts [n_A, n_B] from design (n_A)."""
        n_a = design
        n_b = self.num_participants - design
        return jnp.array([n_a, n_b], dtype=jnp.float32)

    def sample(self, rng_key: Array, design: Array, theta_true: Array) -> Array:
        """Sample observations from the mixture model."""
        trials = self.get_trials(design)
        keys = jr.split(rng_key, 2)

        def sample_group(key, n, theta):
            key_rate, key_obs = jr.split(key)
            theta = jnp.clip(theta, 1e-6, 1.0 - 1e-6)

            alpha = jnp.maximum(self.mixture_concentration * theta, 1e-6)
            beta = jnp.maximum(self.mixture_concentration * (1.0 - theta), 1e-6)
            latent_rate = jr.beta(key_rate, alpha, beta)

            return jr.binomial(key_obs, n.astype(int), latent_rate).astype(jnp.float32)

        return jax.vmap(sample_group)(keys, trials, theta_true)


def compute_marginal_cdf(posterior: Beta, theta_true: Array) -> Array:
    """Compute the CDF value of the true parameter under the posterior."""
    u_a = betainc(posterior.delta[0], posterior.gamma[0], theta_true[0])
    u_b = betainc(posterior.delta[1], posterior.gamma[1], theta_true[1])
    return jnp.array([u_a, u_b])


def compute_coverage_at_level(cdf_values: Array, level: float) -> float:
    """
    Check what fraction of true parameters fall within the central credible interval
    of probability mass `level`.
    """
    lower = (1.0 - level) / 2.0
    upper = 1.0 - lower

    in_interval = (cdf_values >= lower) & (cdf_values <= upper)
    return jnp.mean(in_interval)


def compute_calibration_curve(cdf_values: Array, levels: Array) -> Array:
    """Compute actual coverage for a list of expected coverage levels."""
    return jax.vmap(lambda l: compute_coverage_at_level(cdf_values, l))(levels)


def generate_random_design(rng_key: Array, num_participants: int) -> int:
    """Randomly allocate participants to Group A."""
    return jr.randint(rng_key, shape=(), minval=0, maxval=num_participants + 1)


if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    rng_key = jr.PRNGKey(42)

    # Experiment settings
    num_participants = 1000
    num_trials = 10_000

    # Models: Beta-Binomial
    alpha_nominal = 1.0
    alpha_robust = 1e-2

    # True Process (Beta-mixture Binomial)
    mixture_concentration = 50.0
    proc_prior = Beta(
        delta=jnp.array([1.0, 500.0]),
        gamma=jnp.array([1.0, 500.0]),
    )
    proc_lklhd = BetaMixtureBinomial(
        num_participants=num_participants,
        mixture_concentration=mixture_concentration,
    )

    # Model (Beta-Binomial)
    mdl_prior = Beta(
        delta=jnp.array([1.0, 5.0]),
        gamma=jnp.array([1.0, 5.0]),
    )

    mdl_lklhd = Binomial(num_participants=num_participants)
    model = ABTesting(prior=mdl_prior, likelihood=mdl_lklhd)

    # ============================================================================
    # OPTIMAL DESIGNS
    # ============================================================================
    print("Finding optimal designs...")
    design_opt_nominal = model.optimal_design_shannon()
    designs = jnp.arange(num_participants + 1)
    geig_values = jax.vmap(lambda d: model.power_geig(d, alpha_robust))(designs)
    design_opt_robust = jnp.argmax(geig_values)

    print(f"Optimal n_A - Nominal, α={alpha_nominal:.1f}: {design_opt_nominal}")
    print(f"Optimal n_A - Robust (Power-GEIG), α={alpha_robust:.2f}: {design_opt_robust}")

    print("=" * 80)
    print("COVERAGE COMPARISON: A/B TESTING")
    print("=" * 80)
    print("TRUE PROCESS: Beta-mixture Binomial")
    print(f"  Participants:           {num_participants}")
    print(f"  Mixture concentration:  {mixture_concentration:.1f}")
    print("\nMODELS: Beta-Binomial")
    print(f"  Nominal α={alpha_nominal:.1f}:    Bayes Posterior")
    print(f"  Robust  α={alpha_robust:.2f}:    Robust Posterior")
    print(f"\nNumber of trials:        {num_trials}")
    print("=" * 80)

    # ============================================================================
    # RUN EXPERIMENTS
    # ============================================================================
    print(f"\nRunning {num_trials} trials...")

    def run_single_trial(rng_key):
        key_theta, key_design, key_obs_opt_nom, key_obs_opt_rob, key_obs_rand = jr.split(rng_key, 5)

        # Sample true parameter from prior
        theta_true = proc_prior.sample(key_theta, 1)[0]

        # --- Optimal Designs ---
        x_opt_nom = proc_lklhd.sample(key_obs_opt_nom, design_opt_nominal, theta_true)
        x_opt_rob = proc_lklhd.sample(key_obs_opt_rob, design_opt_robust, theta_true)

        # --- Random Design ---
        design_rand = generate_random_design(key_design, num_participants)
        x_rand = proc_lklhd.sample(key_obs_rand, design_rand, theta_true)

        # Compute tilted posteriors
        post_opt_nom = model.get_tilted_posterior(x_opt_nom, design_opt_nominal, alpha_nominal)
        post_opt_rob = model.get_tilted_posterior(x_opt_rob, design_opt_robust, alpha_robust)
        post_rand_nom = model.get_tilted_posterior(x_rand, design_rand, alpha_nominal)
        post_rand_rob = model.get_tilted_posterior(x_rand, design_rand, alpha_robust)

        return jnp.array([
            compute_marginal_cdf(post_opt_nom, theta_true),
            compute_marginal_cdf(post_opt_rob, theta_true),
            compute_marginal_cdf(post_rand_nom, theta_true),
            compute_marginal_cdf(post_rand_rob, theta_true)
        ])

    # Batch run all trials
    trial_keys = jr.split(rng_key, num_trials)
    results = jax.jit(jax.vmap(run_single_trial))(trial_keys)

    # Extract and flatten results for aggregate curves
    cdf_opt_nominal = results[:, 0, :].flatten()
    cdf_opt_robust = results[:, 1, :].flatten()
    cdf_rand_nominal = results[:, 2, :].flatten()
    cdf_rand_robust = results[:, 3, :].flatten()

    # ============================================================================
    # COMPUTE METRICS
    # ============================================================================
    print("\n" + "=" * 80)
    print("COMPUTING CALIBRATION METRICS")
    print("=" * 80)

    expected_levels = jnp.linspace(0.05, 0.99, 25)

    curve_opt_nom = compute_calibration_curve(cdf_opt_nominal, expected_levels)
    curve_opt_rob = compute_calibration_curve(cdf_opt_robust, expected_levels)
    curve_rand_nom = compute_calibration_curve(cdf_rand_nominal, expected_levels)
    curve_rand_rob = compute_calibration_curve(cdf_rand_robust, expected_levels)

    csv_path = "examples/ab_testing/data/abtesting_coverage_mixture_geig.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "expected_levels",
            "curve_opt_nom",
            "curve_opt_rob",
            "curve_rand_nom",
            "curve_rand_rob",
        ])
        for row in zip(
            expected_levels.tolist(),
            curve_opt_nom.tolist(),
            curve_opt_rob.tolist(),
            curve_rand_nom.tolist(),
            curve_rand_rob.tolist(),
        ):
            writer.writerow(row)

    cov_95_opt_nom = compute_coverage_at_level(cdf_opt_nominal, 0.95)
    cov_95_opt_rob = compute_coverage_at_level(cdf_opt_robust, 0.95)
    cov_95_rand_nom = compute_coverage_at_level(cdf_rand_nominal, 0.95)
    cov_95_rand_rob = compute_coverage_at_level(cdf_rand_robust, 0.95)

    # ============================================================================
    # RESULTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("RESULTS: 95% Credible Interval Coverage")
    print("=" * 80)

    print("\nOPTIMIZED DESIGN (Aggregate)")
    print("-" * 30)
    print(f"  Nominal (α={alpha_nominal:.1f}):     {100 * cov_95_opt_nom:.1f}%")
    print(f"  Robust  (α={alpha_robust:.2f}):     {100 * cov_95_opt_rob:.1f}%")

    print("\nRANDOM DESIGN (Aggregate)")
    print("-" * 30)
    print(f"  Nominal (α={alpha_nominal:.1f}):     {100 * cov_95_rand_nom:.1f}%")
    print(f"  Robust  (α={alpha_robust:.2f}):     {100 * cov_95_rand_rob:.1f}%")

    # ============================================================================
    # COMPARISON TABLE
    # ============================================================================
    table_levels = jnp.array([0.25, 0.50, 0.75, 0.95])
    print("\nCOVERAGE COMPARISON TABLE")
    print("-" * 73)
    print(f"{'Design':<16} | " + " | ".join([f"{100 * l:7.0f}%" for l in table_levels]))
    print("-" * 73)

    rob_opt_covs = [compute_coverage_at_level(cdf_opt_robust, l) for l in table_levels]
    rob_rand_covs = [compute_coverage_at_level(cdf_rand_robust, l) for l in table_levels]
    nom_opt_covs = [compute_coverage_at_level(cdf_opt_nominal, l) for l in table_levels]
    nom_rand_covs = [compute_coverage_at_level(cdf_rand_nominal, l) for l in table_levels]

    print(f"{'Robust + Optimal':<16} | " + " | ".join([f"{100 * c:7.1f}%" for c in rob_opt_covs]))
    print(f"{'Robust + Uniform':<16} | " + " | ".join([f"{100 * c:7.1f}%" for c in rob_rand_covs]))
    print("-" * 73)
    print(f"{'Nominal + Optimal':<16} | " + " | ".join([f"{100 * c:7.1f}%" for c in nom_opt_covs]))
    print(f"{'Nominal + Uniform':<16} | " + " | ".join([f"{100 * c:7.1f}%" for c in nom_rand_covs]))
    print("-" * 73)

    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    print("\nVISUALIZATION")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot([0, 1], [0, 1], ':', color='0.2', lw=2, label='Perfect')
    ax.plot(expected_levels, curve_opt_nom, '--', color='0.1', lw=2.5, label=f'Nominal w/ Optimal (α={alpha_nominal:.1f})')
    ax.plot(expected_levels, curve_opt_rob, '-', color='0.1', lw=2.5, label=f'Robust w/ Power-GEIG Optimal (α={alpha_robust:.2f})')
    ax.plot(expected_levels, curve_rand_nom, '--', color='0.5', lw=2.5, label=f'Nominal w/ Uniform (α={alpha_nominal:.1f})')
    ax.plot(expected_levels, curve_rand_rob, '-', color='0.5', lw=2.5, label=f'Robust w/ Uniform (α={alpha_robust:.2f})')

    ax.set_xlabel('Expected Coverage', fontsize=12)
    ax.set_ylabel('Actual Coverage', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
