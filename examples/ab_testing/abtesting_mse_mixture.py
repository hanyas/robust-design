"""
Posterior Mean MSE under a Beta-mixture Binomial true process.
"""

from typing import NamedTuple

import csv

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr

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


def generate_uniform_design(rng_key: Array, num_participants: int) -> Array:
    """Uniform random design in {0, ..., num_participants}."""
    return jr.randint(rng_key, shape=(), minval=0, maxval=num_participants + 1)


def custom_split(rng_key: Array, num_splits: int) -> tuple[Array, Array]:
    """Split RNG key into num_splits parts."""
    keys = jr.split(rng_key, num_splits)
    return keys[0], keys[1:]


def summarize(values: Array) -> tuple[Array, Array, Array]:
    """Mean and central 80% interval (q10, q90) across trials."""
    mean = jnp.mean(values, axis=1)
    q10 = jnp.percentile(values, 10, axis=1)
    q90 = jnp.percentile(values, 90, axis=1)
    return mean, q10, q90


if __name__ == "__main__":
    rng_key = jr.PRNGKey(42)

    num_participants = 100
    num_trials = 10_000

    alpha_values = 10.0 ** jnp.linspace(-1.25, 0.0, 5)
    num_alphas = len(alpha_values)

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

    # Model (Beta-Binomial) — asymmetric so the optimal design varies with α.
    mdl_prior = Beta(
        delta=jnp.array([10.0, 1.0]),
        gamma=jnp.array([10.0, 1.0]),
    )
    mdl_lklhd = Binomial(num_participants=num_participants)
    model = ABTesting(prior=mdl_prior, likelihood=mdl_lklhd)

    print("=" * 100)
    print("POSTERIOR MEAN RMSE EXPERIMENT UNDER BETA-MIXTURE TRUE PROCESS: A/B TESTING")
    print("=" * 100)
    print("TRUE PROCESS: Beta-mixture Binomial")
    print(f"  Participants:           {num_participants}")
    print(f"  Mixture concentration:  {mixture_concentration:.1f}")
    print("\nMODEL: Beta-Binomial")
    print("  Robust learning:  q(θ|x) ∝ p(θ) p(x|θ)^α")
    print("Metric")
    print("  sqrt( mean_i (E[θ|x,ξ]_i − θ_true_i)² )  per trial  (RMSE)")
    print("Comparison rule")
    print("  For each α, compare:")
    print("    1. the Sibson-selected design")
    print("    2. a uniform random design")
    print(f"\nAlpha values:       {num_alphas}")
    print(f"Number of trials:   {num_trials}")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("FINDING SIBSON DESIGNS")
    print("=" * 100)

    designs = jnp.arange(num_participants + 1)
    sibson_matrix = jax.vmap(lambda alpha: jax.vmap(lambda d: model.sibson_mi(d, alpha))(designs))(alpha_values)
    sibson_designs = jnp.argmax(sibson_matrix, axis=1)

    for idx in range(num_alphas):
        print(f"  α = {float(alpha_values[idx]):.4f}: Sibson n_A = {int(sibson_designs[idx])}")

    print("\n" + "=" * 100)
    print("RUNNING RMSE COMPARISON")
    print("=" * 100)

    @jax.jit
    def evaluate_all_alphas(
        alpha_vals: Array,
        sibson_design_vals: Array,
        x_sibson: Array,
        x_unif: Array,
        design_unif: Array,
        theta_true: Array,
    ) -> tuple[Array, Array]:
        """Per-trial RMSE of the posterior mean for both designs across α."""

        def evaluate_alpha(alpha, design_sibson, x_sibson_val):
            posterior_sibson = model.get_tilted_posterior(x_sibson_val, design_sibson, alpha)
            posterior_unif = model.get_tilted_posterior(x_unif, design_unif, alpha)

            theta_hat_sibson = posterior_sibson.delta / (posterior_sibson.delta + posterior_sibson.gamma)
            theta_hat_unif = posterior_unif.delta / (posterior_unif.delta + posterior_unif.gamma)

            rmse_sibson = jnp.sqrt(jnp.mean((theta_hat_sibson - theta_true) ** 2))
            rmse_unif = jnp.sqrt(jnp.mean((theta_hat_unif - theta_true) ** 2))
            return rmse_sibson, rmse_unif

        return jax.vmap(evaluate_alpha)(alpha_vals, sibson_design_vals, x_sibson)

    rmse_sibson = jnp.zeros((num_alphas, num_trials))
    rmse_uniform = jnp.zeros((num_alphas, num_trials))

    print(f"\nRunning {num_trials} trials...")

    for trial in range(num_trials):
        if (trial + 1) % 200 == 0 or trial == 0:
            print(f"[{trial + 1:4d}/{num_trials}]  Evaluating...")

        rng_key, sub_key = jr.split(rng_key)
        theta_true = proc_prior.sample(sub_key, 1)[0]

        rng_key, sibson_keys = custom_split(rng_key, num_alphas + 1)
        x_sibson = jax.vmap(lambda k, d: proc_lklhd.sample(k, d, theta_true))(sibson_keys, sibson_designs)

        rng_key, sub_key = jr.split(rng_key)
        design_uniform = generate_uniform_design(sub_key, num_participants)
        rng_key, sub_key = jr.split(rng_key)
        x_uniform = proc_lklhd.sample(sub_key, design_uniform, theta_true)

        r_sibson, r_unif = evaluate_all_alphas(
            alpha_values,
            sibson_designs,
            x_sibson,
            x_uniform,
            design_uniform,
            theta_true,
        )

        rmse_sibson = rmse_sibson.at[:, trial].set(r_sibson)
        rmse_uniform = rmse_uniform.at[:, trial].set(r_unif)

    print("✓ Complete")

    print("\n" + "=" * 100)
    print("RESULTS: Posterior Mean RMSE ACROSS α")
    print("=" * 100)

    mean_opt, q10_opt, q90_opt = summarize(rmse_sibson)
    mean_unif, q10_unif, q90_unif = summarize(rmse_uniform)

    print(f"\n{'α':>8} | {'Optimal: mean [q10, q90]':>32} | {'Uniform: mean [q10, q90]':>32}")
    print("-" * 82)
    for idx in range(num_alphas):
        print(
            f"{float(alpha_values[idx]):>8.3f} | "
            f"{float(mean_opt[idx]):>8.4f} [{float(q10_opt[idx]):.4f}, {float(q90_opt[idx]):.4f}] | "
            f"{float(mean_unif[idx]):>8.4f} [{float(q10_unif[idx]):.4f}, {float(q90_unif[idx]):.4f}]"
        )

    csv_path = "examples/ab_testing/data/abtesting_mse_mixture.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["alpha", "mean_opt", "q10_opt", "q90_opt", "mean_unif", "q10_unif", "q90_unif"])
        for idx in range(num_alphas):
            writer.writerow(
                [
                    float(alpha_values[idx]),
                    float(mean_opt[idx]),
                    float(q10_opt[idx]),
                    float(q90_opt[idx]),
                    float(mean_unif[idx]),
                    float(q10_unif[idx]),
                    float(q90_unif[idx]),
                ]
            )
