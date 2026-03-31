"""
Posterior Predictive Performance for Misspecified Linear Regression

This benchmark compares Sibson-selected and power-GEIG-selected two-point
designs under a Student-t true process.
"""
from typing import NamedTuple

import csv

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian, LinearRegression

jax.config.update("jax_enable_x64", True)


class LinearStudentT(NamedTuple):
    """
    Student-t linear likelihood for regression.

    Args:
        noise_stddev: Noise standard deviation
        df: Degrees of freedom
    """
    df: float
    noise_stddev: float

    def design_matrix(self, design: Array) -> Array:
        """Construct augmented design matrix H(xi) = [xi, 1]."""
        design = jnp.atleast_1d(design)
        size = design.shape[0]
        return jnp.column_stack([design, jnp.ones(size)])

    def mean(self, design: Array, theta: Array) -> Array:
        """Compute mean for a single design point."""
        h = self.design_matrix(design)
        return h @ theta

    def sample(self, rng_key: Array, design: Array, theta: Array) -> Array:
        """Sample from Student-t likelihood."""
        size = design.shape[0]

        loc = self.mean(design, theta)
        scale_matrix = self.noise_stddev**2 * jnp.eye(size)

        rng_key, key1, key2 = jr.split(rng_key, 3)
        chi2_samples = jr.gamma(key1, self.df / 2, shape=(size,)) * 2

        l_factor = jnp.linalg.cholesky(scale_matrix)
        z = jr.normal(key2, (size,))
        normal_samples = loc + l_factor @ z

        scale_factors = jnp.sqrt(self.df / chi2_samples)
        return loc + (normal_samples - loc) * scale_factors


def generate_sibson_optimal_design(num_designs: int) -> Array:
    """Known Sibson-optimal two-point boundary design."""
    return jnp.array([(-1.0) ** (i + 1) for i in range(num_designs)])


def genereate_geig_optimal_design(num_designs: int) -> Array:
    """Power-GEIG-optimal design, equal to the Sibson optimum here."""
    return generate_sibson_optimal_design(num_designs)


def generate_uniform_design(rng_key: Array, num_designs: int) -> Array:
    """Generate uniform random design in [-1, 1]."""
    return jr.uniform(rng_key, shape=(num_designs,), minval=-1.0, maxval=1.0)


def custom_split(rng_key: Array, num_splits: int) -> tuple[Array, Array]:
    """Split the RNG key into num_splits parts."""
    keys = jr.split(rng_key, num_splits)
    return keys[0], keys[1:]


def build_alpha_grid() -> Array:
    """Log-spaced robustness orders from 1e-2 to 1."""
    decades = 10.0 ** jnp.arange(-2, 0)
    return jnp.concatenate([jnp.arange(1, 11) * d for d in decades])


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
    # Configuration
    # ============================================================================
    rng_key = jr.PRNGKey(42)

    alpha_values = build_alpha_grid()
    num_alphas = len(alpha_values)

    num_design_points = 2
    num_test_points = 32
    num_trials = 1024

    dim_theta = 2
    base_stddev = jnp.sqrt(1e-2)

    # True process (Student-t)
    df_true = 5.0
    proc_prior_mean = jnp.array([1.5, -1.5])
    proc_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
    proc_prior = Gaussian(mean=proc_prior_mean, cov=proc_prior_cov)

    proc_stddev = base_stddev * 2 / 5e-1
    proc_lklhd = LinearStudentT(df=df_true, noise_stddev=proc_stddev)

    # Model (Gaussian)
    mdl_prior_mean = jnp.array([1.5, -1.5])
    mdl_prior_cov = jnp.diag(jnp.array([1.0, 1.0]))
    mdl_prior = Gaussian(mean=mdl_prior_mean, cov=mdl_prior_cov)

    mdl_lklhd = LinearGaussian(noise_stddev=base_stddev)
    model = LinearRegression(prior=mdl_prior, likelihood=mdl_lklhd)

    print("=" * 96)
    print("PREDICTIVE PERFORMANCE EXPERIMENT: STUDENT-T TRUE PROCESS")
    print("=" * 96)
    print(f"TRUE PROCESS: Student-t with df={df_true}")
    print(f"  Noise: σ² = {proc_stddev**2:.4f}")
    print("\nMODEL: LinearGaussian")
    print("  Robust learning:   q(theta|x) ∝ p(theta) p(x|theta)^alpha")
    print("Comparison rule")
    print("  For each alpha, compare Sibson-selected and Power-GEIG-selected designs")
    print(f"\nDesign points:      {num_design_points}")
    print("Design selection:   closed-form boundary optimum")
    print(f"Test points:        {num_test_points}")
    print(f"Number of trials:   {num_trials}")
    print("=" * 96)

    # ============================================================================
    # Optimal designs per alpha
    # ============================================================================
    print("\n" + "=" * 96)
    print("USING CLOSED-FORM SIBSON AND POWER-GEIG DESIGNS")
    print("=" * 96)

    sibson_design = generate_sibson_optimal_design(num_design_points)
    geig_design = genereate_geig_optimal_design(num_design_points)
    sibson_designs = jnp.repeat(sibson_design[None, :], num_alphas, axis=0)
    geig_designs = jnp.repeat(geig_design[None, :], num_alphas, axis=0)
    design_gaps = jnp.linalg.norm(sibson_designs - geig_designs, axis=1)

    for alpha_target in [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]:
        idx = jnp.argmin(jnp.abs(alpha_values - alpha_target))
        design_s = sibson_designs[idx]
        design_g = geig_designs[idx]
        print(
            f"  alpha = {float(alpha_values[idx]):.4f}: "
            f"Sibson = ({float(design_s[0]): .3f}, {float(design_s[1]): .3f}), "
            f"GEIG = ({float(design_g[0]): .3f}, {float(design_g[1]): .3f}), "
            f"gap = {float(design_gaps[idx]):.6f}"
        )

    # ============================================================================
    # Run experiments
    # ============================================================================
    print("\n" + "=" * 96)
    print("RUNNING ELPD COMPARISON")
    print("=" * 96)

    @jax.jit
    def evaluate_all_alphas(
        alpha_vals: Array,
        sibson_design_vals: Array,
        geig_design_vals: Array,
        y_sibson: Array,
        y_geig: Array,
        xi_test: Array,
        y_test: Array,
    ) -> tuple[Array, Array]:
        """Evaluate held-out ELPD for both design rules across all alpha values."""

        def evaluate_alpha(alpha, xi_sibson, xi_geig, y_sibson_i, y_geig_i):
            posterior_sibson = model.get_tilted_posterior(y_sibson_i, xi_sibson, alpha)
            log_preds_sibson = jax.vmap(
                lambda xi_i, y_i: model.get_posterior_predictive(
                    posterior_sibson, xi_i, alpha
                ).logpdf(y_i)
            )(xi_test, y_test)

            posterior_geig = model.get_tilted_posterior(y_geig_i, xi_geig, alpha)
            log_preds_geig = jax.vmap(
                lambda xi_i, y_i: model.get_posterior_predictive(
                    posterior_geig, xi_i, alpha
                ).logpdf(y_i)
            )(xi_test, y_test)

            return jnp.mean(log_preds_sibson), jnp.mean(log_preds_geig)

        return jax.vmap(evaluate_alpha)(
            alpha_vals,
            sibson_design_vals,
            geig_design_vals,
            y_sibson,
            y_geig,
        )

    elpd_sibson = jnp.zeros((num_alphas, num_trials))
    elpd_geig = jnp.zeros((num_alphas, num_trials))

    print(f"\nRunning {num_trials} trials...")

    for trial in range(num_trials):
        if (trial + 1) % 200 == 0 or trial == 0:
            print(f"  Trial {trial + 1}/{num_trials}...")

        rng_key, sub_key = jr.split(rng_key)
        theta_true = proc_prior.sample(sub_key, 1)[0]

        rng_key, obs_keys = custom_split(rng_key, num_alphas + 1)
        y_sibson = jax.vmap(lambda k, xi: proc_lklhd.sample(k, xi, theta_true))(
            obs_keys, sibson_designs
        )
        y_geig = jax.vmap(lambda k, xi: proc_lklhd.sample(k, xi, theta_true))(
            obs_keys, geig_designs
        )

        rng_key, test_design_keys = custom_split(rng_key, num_test_points + 1)
        xi_test = jax.vmap(lambda k: generate_uniform_design(k, num_design_points))(
            test_design_keys
        )

        rng_key, test_obs_keys = custom_split(rng_key, num_test_points + 1)
        y_test = jax.vmap(lambda k, xi: proc_lklhd.sample(k, xi, theta_true))(
            test_obs_keys, xi_test
        )

        elpds_sibson, elpds_geig = evaluate_all_alphas(
            alpha_values,
            sibson_designs,
            geig_designs,
            y_sibson,
            y_geig,
            xi_test,
            y_test,
        )

        elpd_sibson = elpd_sibson.at[:, trial].set(elpds_sibson)
        elpd_geig = elpd_geig.at[:, trial].set(elpds_geig)

    print("✓ Complete")

    # ============================================================================
    # Analysis
    # ============================================================================
    print("\n" + "=" * 96)
    print("RESULTS: ELPD BY DESIGN SELECTION RULE")
    print("=" * 96)

    mean_elpd_sibson, se_elpd_sibson, q05_sibson, q95_sibson = summarize_elpd(
        elpd_sibson, num_trials
    )
    mean_elpd_geig, se_elpd_geig, q05_geig, q95_geig = summarize_elpd(
        elpd_geig, num_trials
    )
    geig_minus_sibson = mean_elpd_geig - mean_elpd_sibson

    print(
        f"\n{'alpha':>8} | {'Sibson design':>27} | {'GEIG design':>27} | "
        f"{'Sibson ELPD':>20} | {'GEIG ELPD':>20} | {'GEIG - Sibson':>14}"
    )
    print("-" * 136)
    for alpha_target in [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]:
        idx = jnp.argmin(jnp.abs(alpha_values - alpha_target))
        design_s = sibson_designs[idx]
        design_g = geig_designs[idx]
        print(
            f"{float(alpha_values[idx]):>8.3f} | "
            f"({float(design_s[0]):>6.3f}, {float(design_s[1]):>6.3f}){'':>10} | "
            f"({float(design_g[0]):>6.3f}, {float(design_g[1]):>6.3f}){'':>10} | "
            f"{float(mean_elpd_sibson[idx]):>10.4f} ± {float(se_elpd_sibson[idx]):<7.4f} | "
            f"{float(mean_elpd_geig[idx]):>10.4f} ± {float(se_elpd_geig[idx]):<7.4f} | "
            f"{float(geig_minus_sibson[idx]):>14.4f}"
        )

    # ============================================================================
    # Visualization + CSV
    # ============================================================================
    csv_path = "examples/linear_regression/data/linreg_elpd_studentt_geig.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "alpha",
            "sibson_x1",
            "sibson_x2",
            "geig_x1",
            "geig_x2",
            "design_gap_l2",
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
                float(sibson_designs[idx, 0]),
                float(sibson_designs[idx, 1]),
                float(geig_designs[idx, 0]),
                float(geig_designs[idx, 1]),
                float(design_gaps[idx]),
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

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)

    axes[0].plot(alpha_values, mean_elpd_sibson, "-", color="0.15", lw=2.5, label="Sibson design")
    axes[0].fill_between(alpha_values, q05_sibson, q95_sibson, alpha=0.2, color="0.15")
    axes[0].plot(alpha_values, mean_elpd_geig, "-", color="0.45", lw=2.5, label="Power-GEIG design")
    axes[0].fill_between(alpha_values, q05_geig, q95_geig, alpha=0.2, color="0.45")
    axes[0].axvline(1.0, color="gray", linestyle=":", lw=1.5, label="Nominal (alpha=1)")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("ELPD", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].plot(alpha_values, design_gaps, "-", color="0.2", lw=2.5)
    axes[1].axvline(1.0, color="gray", linestyle=":", lw=1.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("alpha", fontsize=12)
    axes[1].set_ylabel("L2 design gap", fontsize=12)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
