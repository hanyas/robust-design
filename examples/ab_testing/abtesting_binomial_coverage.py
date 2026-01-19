"""
Posterior Calibration Assessment for Misspecified A/B Testing

TRUE PROCESS: Heavey-Tailed Binomial
MODELS: BetaBinomial

This script evaluates posterior calibration when the true data-generating process
has overdispersion but we fit a standard Binomial model.
"""
from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import betainc
import csv

from rdesign.distributions import Beta
from rdesign.ab_testing import ABTesting, Binomial

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


class HeavyTailBinomial(NamedTuple):
    """
    True generative process: Heavy Tail Binomial with asymmetric overdispersion.

    Args:
        num_participants: Total participants per experiment
        concentration_a: Overdispersion for Group A
        concentration_b: Overdispersion for Group B
    """
    num_participants: int
    concentration_a: float = 100.0
    concentration_b: float = 100.0

    def get_trials(self, design: Array) -> Array:
        """Get trial counts [n_A, n_B] from design (n_A)."""
        n_a = design
        n_b = self.num_participants - design
        return jnp.array([n_a, n_b], dtype=jnp.float32)

    def sample(self, rng_key: Array, design: Array, theta_true: Array) -> Array:
        """
        Sample observations x with overdispersion.
        """
        trials = self.get_trials(design)
        key_a, key_b = jr.split(rng_key)

        def sample_effective_rate(key, mean, conc):
            conc = jnp.maximum(conc, 1e-4)
            a = mean * conc
            b = (1.0 - mean) * conc
            return jr.beta(key, a, b)

        theta_eff_a = sample_effective_rate(key_a, theta_true[0], self.concentration_a)
        theta_eff_b = sample_effective_rate(key_b, theta_true[1], self.concentration_b)

        # Sample observed conversions using the jittered rates
        x_a = jr.binomial(key_a, trials[0].astype(int), theta_eff_a)
        x_b = jr.binomial(key_b, trials[1].astype(int), theta_eff_b)

        return jnp.array([x_a, x_b], dtype=jnp.float32)


def compute_marginal_cdf(posterior: Beta, theta_true: Array) -> Array:
    """
    Compute the CDF value of the true parameter under the posterior.
    """
    # jax.scipy.special.betainc computes the regularized incomplete beta function
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

    # Check if CDF value is within boundsyou get teh
    in_interval = (cdf_values >= lower) & (cdf_values <= upper)
    return jnp.mean(in_interval)


def compute_calibration_curve(cdf_values: Array, levels: Array) -> Array:
    """Compute actual coverage for a list of expected coverage levels."""
    return jax.vmap(lambda l: compute_coverage_at_level(cdf_values, l))(levels)


def generate_random_design(rng_key: Array, num_participants: int) -> int:
    """Randomly allocate participants to Group A."""
    return jr.randint(rng_key, shape=(), minval=0, maxval=num_participants)


if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    rng_key = jr.PRNGKey(42)

    # True Process: Heavy Tail Binomial
    concentration_a = 5.0  # High noise
    concentration_b = 2000.0  # Low noise
    num_participants = 1000

    # Models: Beta-Binomial
    alpha_nominal = 1.0
    alpha_robust = 0.05

    # Experiment settings
    num_trials = 100_000

    # Prior Setup
    prior = Beta(
        delta=jnp.array([1.0, 1.0]),
        gamma=jnp.array([50.0, 10.0])
    )

    # Instantiate Classes
    process = HeavyTailBinomial(
        num_participants=num_participants,
        concentration_a=concentration_a,
        concentration_b=concentration_b
    )
    model = Binomial(num_participants=num_participants)
    abtest = ABTesting(prior=prior, likelihood=model)

    # ============================================================================
    # OPTIMAL DESIGNS
    # ============================================================================
    print("Finding optimal designs...")
    design_opt_nominal = abtest.optimal_design_shannon()
    design_opt_robust = abtest.optimal_design_sibson(alpha_robust)

    print(f"Optimal n_A - Nominal, α={alpha_nominal:.1f}: {design_opt_nominal}")
    print(f"Optimal n_A - Robust,  α={alpha_robust:.1f}: {design_opt_robust}")

    print("=" * 80)
    print("COVERAGE COMPARISON: A/B TESTING")
    print("=" * 80)
    print("TRUE PROCESS: Heavy Tail Binomial (Asymmetric)")
    print(f"  Concentration A: {concentration_a}")
    print(f"  Concentration B: {concentration_b}")
    print(f"  Participants:    {num_participants}")
    print("\nMODELS: Beta-Binomial")
    print(f"  Nominal α={alpha_nominal:.1f}:    Standard Bayesian Update")
    print(f"  Robust  α={alpha_robust:.1f}:    Tilted Posterior")
    print(f"\nNumber of trials:    {num_trials}")
    print("=" * 80)

    # ============================================================================
    # RUN EXPERIMENTS
    # ============================================================================
    print(f"\nRunning {num_trials} trials...")

    def run_single_trial(rng_key):
        key_theta, key_design, key_obs_opt_nom, key_obs_opt_rob, key_obs_rand = jr.split(rng_key, 5)

        # Sample True Parameter from prior
        theta_true = prior.sample(key_theta)

        # --- Optimal Designs ---
        x_opt_nom = process.sample(key_obs_opt_nom, design_opt_nominal, theta_true)
        x_opt_rob = process.sample(key_obs_opt_rob, design_opt_robust, theta_true)

        # --- Random Design ---
        design_rand = generate_random_design(key_design, num_participants)
        x_rand = process.sample(key_obs_rand, design_rand, theta_true)

        # Compute Bayes Posteriors
        post_opt_nom = abtest.get_tilted_posterior(x_opt_nom, design_opt_nominal, alpha_nominal)
        post_opt_rob = abtest.get_tilted_posterior(x_opt_rob, design_opt_robust, alpha_robust)
        post_rand_nom = abtest.get_tilted_posterior(x_rand, design_rand, alpha_nominal)
        post_rand_rob = abtest.get_tilted_posterior(x_rand, design_rand, alpha_robust)

        # Compute CDF values
        return jnp.array([
            compute_marginal_cdf(post_opt_nom, theta_true),
            compute_marginal_cdf(post_opt_rob, theta_true),
            compute_marginal_cdf(post_rand_nom, theta_true),
            compute_marginal_cdf(post_rand_rob, theta_true)
        ])

    # Batch run all trials
    trial_keys = jr.split(rng_key, num_trials)
    results = jax.jit(jax.vmap(run_single_trial))(trial_keys)

    # Extract and flatten results
    # results shape: (num_trials, 4, 2) -> (opt_nom, opt_rob, rand_nom, rand_rob)
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

    # Write plotted data to CSV
    csv_path = "examples/ab_testing/abtesting_binomial_coverage.csv"
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

    # Specific check at 95%
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

    print("\nOPTIMIZED DESIGN")
    print("-" * 30)
    print(f"  Nominal (α={alpha_nominal:.1f}):     {100*cov_95_opt_nom:.1f}%")
    print(f"  Robust  (α={alpha_robust:.1f}):     {100*cov_95_opt_rob:.1f}%")

    print("\nRANDOM DESIGN")
    print("-" * 30)
    print(f"  Nominal (α={alpha_nominal:.1f}):     {100*cov_95_rand_nom:.1f}%")
    print(f"  Robust  (α={alpha_robust:.1f}):     {100*cov_95_rand_rob:.1f}%")

    # ============================================================================
    # COMPARISON TABLE
    # ============================================================================
    table_levels = jnp.linspace(0.1, 1.0, 10)
    print("\nCOVERAGE COMPARISON TABLE")
    print("-" * 115)
    print(f"{'Design':<16} | " + " | ".join([f"{l:8.1f}" for l in table_levels]))
    print("-" * 115)

    # Robust evaluations
    rob_opt_covs = [compute_coverage_at_level(cdf_opt_robust, l) for l in table_levels]
    rob_rand_covs = [compute_coverage_at_level(cdf_rand_robust, l) for l in table_levels]

    # Nominal evaluations
    nom_opt_covs = [compute_coverage_at_level(cdf_opt_nominal, l) for l in table_levels]
    nom_rand_covs = [compute_coverage_at_level(cdf_rand_nominal, l) for l in table_levels]

    print(f"{'Robust + Optimal':<16} | " + " | ".join([f"{100*c:8.1f}%" for c in rob_opt_covs]))
    print(f"{'Robust + Uniform':<16} | " + " | ".join([f"{100*c:8.1f}%" for c in rob_rand_covs]))
    print("-" * 115)
    print(f"{'Nominal + Optimal':<16} | " + " | ".join([f"{100*c:8.1f}%" for c in nom_opt_covs]))
    print(f"{'Nominal + Uniform':<16} | " + " | ".join([f"{100*c:8.1f}%" for c in nom_rand_covs]))
    print("-" * 115)

    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    print("\nVISUALIZATION")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], ':', color='0.2', lw=2, label='Perfect')

    # Curves
    # Optimal designs (solid lines, dark colors)
    ax.plot(expected_levels, curve_opt_nom, '--', color='0.1', lw=2.5, label=f'Nominal w/ Optimal (α={alpha_nominal:.1f})')
    ax.plot(expected_levels, curve_opt_rob, '-', color='0.1', lw=2.5, label=f'Robust w/ Optimal (α={alpha_robust:.1f})')

    # Random designs (dashed lines, lighter colors)
    ax.plot(expected_levels, curve_rand_nom, '--', color='0.5', lw=2.5, label=f'Nominal w/ Uniform (α={alpha_nominal:.1f})')
    ax.plot(expected_levels, curve_rand_rob, '-', color='0.5', lw=2.5, label=f'Robust w/ Uniform (α={alpha_robust:.1f})')

    ax.set_xlabel('Expected Coverage', fontsize=12)
    ax.set_ylabel('Actual Coverage', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
