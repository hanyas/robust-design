"""
Posterior Calibration under Random Designs Across an α Sweep.
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
    rng_key = jr.PRNGKey(42)

    # Experiment settings
    num_participants = 100
    num_trials = 10_000

    # α grid: quarter-decade from 10^{-5/4} to 10^0 (approx 0.056 -> 1)
    alpha_values = 10.0 ** jnp.linspace(-1.25, 0.0, 5)
    num_alphas = alpha_values.shape[0]

    # True Process (Beta-mixture Binomial)
    mixture_concentration = 10.0
    proc_prior = Beta(
        delta=jnp.array([100.0, 1.0]),
        gamma=jnp.array([100.0, 1.0]),
    )
    proc_lklhd = BetaMixtureBinomial(
        num_participants=num_participants,
        mixture_concentration=mixture_concentration,
    )

    # Model (Beta-Binomial)
    mdl_prior = Beta(
        delta=jnp.array([10.0, 1.0]),
        gamma=jnp.array([10.0, 1.0]),
    )

    mdl_lklhd = Binomial(num_participants=num_participants)
    model = ABTesting(prior=mdl_prior, likelihood=mdl_lklhd)

    print("=" * 80)
    print("COVERAGE SWEEP UNDER RANDOM DESIGNS: A/B TESTING")
    print("=" * 80)
    print("TRUE PROCESS: Beta-mixture Binomial")
    print(f"  Participants:           {num_participants}")
    print(f"  Mixture concentration:  {mixture_concentration:.1f}")
    print("\nMODEL: Beta-Binomial (uniform random designs only)")
    print(f"  α values:        {[float(a) for a in alpha_values]}")
    print(f"  Number of trials: {num_trials}")
    print("=" * 80)

    print(f"\nRunning {num_trials} trials...")

    def run_single_trial(rng_key):
        """For one trial, return CDF values per α (shape (num_alphas, 2))."""
        key_theta, key_design, key_obs = jr.split(rng_key, 3)

        # Sample true parameter from prior
        theta_true = proc_prior.sample(key_theta, 1)[0]

        # Random design and observations (shared across α values)
        design_rand = generate_random_design(key_design, num_participants)
        x_rand = proc_lklhd.sample(key_obs, design_rand, theta_true)

        # Tilted posterior CDF for each α in the grid
        def cdf_for_alpha(alpha):
            posterior = model.get_tilted_posterior(x_rand, design_rand, alpha)
            return compute_marginal_cdf(posterior, theta_true)

        return jax.vmap(cdf_for_alpha)(alpha_values)

    # Batch run all trials
    trial_keys = jr.split(rng_key, num_trials)
    results = jax.jit(jax.vmap(run_single_trial))(trial_keys)
    # results shape: (num_trials, num_alphas, 2)

    # Flatten across trials and arms for each α
    cdf_per_alpha = [results[:, i, :].flatten() for i in range(num_alphas)]

    print("\n" + "=" * 80)
    print("COMPUTING CALIBRATION METRICS")
    print("=" * 80)

    expected_levels = jnp.linspace(0.05, 0.99, 25)
    curves = [compute_calibration_curve(cdf, expected_levels) for cdf in cdf_per_alpha]

    csv_path = "examples/ab_testing/data/abtesting_coverage_mixture.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["expected_levels"] + [f"alpha_{float(a):.4f}" for a in alpha_values]
        writer.writerow(header)
        for level_idx, level in enumerate(expected_levels.tolist()):
            row = [level] + [curves[i][level_idx].item() for i in range(num_alphas)]
            writer.writerow(row)

    print("\n" + "=" * 80)
    print("COVERAGE COMPARISON TABLE")
    print("=" * 80)
    table_levels = jnp.array([0.25, 0.50, 0.75, 0.95])
    print(f"{'α':>8} | " + " | ".join([f"{100 * l:7.0f}%" for l in table_levels]))
    print("-" * 73)
    for i, alpha in enumerate(alpha_values):
        covs = [compute_coverage_at_level(cdf_per_alpha[i], l) for l in table_levels]
        print(f"{float(alpha):>8.4f} | " + " | ".join([f"{100 * c:7.1f}%" for c in covs]))

    print("\nVISUALIZATION")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot([0, 1], [0, 1], ":", color="0.2", lw=2, label="Perfect")

    # Grayscale: small α (dark) -> large α (light)
    for i, alpha in enumerate(alpha_values):
        shade = 0.10 + 0.65 * (i / max(num_alphas - 1, 1))
        ax.plot(
            expected_levels,
            curves[i],
            "-",
            color=str(shade),
            lw=2.5,
            label=f"α = {float(alpha):.3f}",
        )

    ax.set_xlabel("Expected Coverage", fontsize=12)
    ax.set_ylabel("Actual Coverage", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
