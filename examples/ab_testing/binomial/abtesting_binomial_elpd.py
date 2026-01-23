"""
Posterior Predictive Performance for Misspecified A/B Testing

TRUE PROCESS: Heavy-Tailed Binomial
MODELS: Beta-Binomial with α-tilted posterior and predictive
"""

from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import betainc, gammaln

import csv
import matplotlib.pyplot as plt

from rdesign.distributions import Beta
from rdesign.ab_testing import ABTesting, Binomial

jax.config.update("jax_enable_x64", True)


class TiltedBinomial(NamedTuple):
    """
    Generates data from the α-tilted Binomial likelihood p_α(x | θ).
    
    Args:
        num_participants: Total participants (defines the support size)
        alpha: Tilting parameter. 
    """
    num_participants: int
    alpha: float = 1.0

    def get_trials(self, design: Array) -> Array:
        """Get trial counts [n_A, n_B] from design (n_A)."""
        n_a = design
        n_b = self.num_participants - design
        return jnp.array([n_a, n_b], dtype=jnp.float32)

    def _log_tilted_prob(self, k: Array, n: Array, theta: Array) -> Array:
        """
        Computes unnormalized log probability: α * log(Binomial PMF)
        """
        # Safe masking for support [0, n]
        mask = (k >= 0) & (k <= n)
        safe_k = jnp.where(mask, k, 0.0)
        safe_nmk = jnp.where(mask, n - safe_k, 0.0)
        
        # Log Binomial Coefficient: log(nCx)
        log_comb = gammaln(n + 1) - gammaln(safe_k + 1) - gammaln(safe_nmk + 1)
        
        # Log Likelihood: x log(θ) + (n-x) log(1-θ)
        # We clamp theta to avoid log(0)
        theta = jnp.clip(theta, 1e-6, 1.0 - 1e-6)
        log_lklhd = safe_k * jnp.log(theta) + safe_nmk * jnp.log(1.0 - theta)
        
        # Tilted Log Prob: α * (log_comb + log_lklhd)
        return jnp.where(mask, self.alpha * (log_comb + log_lklhd), -jnp.inf)

    def sample(self, rng_key: Array, design: Array, theta_true: Array) -> Array:
        """
        Sample observations x from the tilted likelihood.
        """
        trials = self.get_trials(design)
        
        # Grid of all possible outcomes [0, 1, ..., N]
        k_grid = jnp.arange(self.num_participants + 1, dtype=jnp.float32)
        
        def sample_group(key, n, theta):
            # 1. Compute unnormalized logits for all k
            logits = self._log_tilted_prob(k_grid, n, theta)
            
            # 2. Sample from Categorical
            # JAX's categorical handles the normalization (log-sum-exp) internally
            return jr.categorical(key, logits).astype(jnp.float32)

        # Vectorize sampling over the two groups (A and B)
        keys = jr.split(rng_key, 2)
        
        # vmap applies sample_group to (key[0], trials[0], theta[0]) and (...[1]...)
        return jax.vmap(sample_group)(keys, trials, theta_true)


def generate_uniform_design(rng_key: Array, num_participants: int) -> Array:
    """Uniform random design in {0, ..., num_participants}."""
    return jr.randint(rng_key, shape=(), minval=0, maxval=num_participants + 1)


def custom_split(rng_key: Array, num_splits: int) -> Array:
    """Split RNG key into num_splits parts."""
    keys = jr.split(rng_key, num_splits)
    return keys[0], keys[1:]


# ============================================================================
# Configuration
# ============================================================================

rng_key = jr.PRNGKey(42)

# Experiment settings
num_participants = 2500
num_trials = 10_000
num_test_points = 32

# Alpha values to evaluate
decades = 10.0 ** jnp.arange(-2, 0)
alpha_values = jnp.concatenate([jnp.arange(1, 11) * d for d in decades])
num_alphas = len(alpha_values)

# True Process (Tilted Binomial)
alpha_true = 1e-1
proc_prior = Beta(
    delta=jnp.array([1, 5000.0]),
    gamma=jnp.array([1, 5000.0])
)
proc_lklhd = TiltedBinomial(
    num_participants=num_participants, 
    alpha=alpha_true
)

# Model (Beta-Binomial)
mdl_prior = Beta(
    delta=jnp.array([2., 50.0]),
    gamma=jnp.array([2., 50.0])
)

mdl_lklhd = Binomial(num_participants=num_participants)
model = ABTesting(prior=mdl_prior, likelihood=mdl_lklhd)

print("=" * 80)
print("PREDICTIVE PERFORMANCE EXPERIMENT: A/B TESTING")
print("=" * 80)
print("TRUE PROCESS: Heavy-Tailed Binomial (Asymmetric)")
print(f"  Participants:    {num_participants}")
print("\nMODEL: Beta-Binomial")
print("  Robust learning:   q(θ|x) ∝ p(θ) p(x|θ)^α")
print("\nTest points:       {0}".format(num_test_points))
print("Number of trials:  {0}".format(num_trials))
print("=" * 80)

# ============================================================================
# Optimal Designs per α
# ============================================================================

print("\n" + "=" * 80)
print("FINDING OPTIMAL DESIGNS")
print("=" * 80)

designs = jnp.arange(num_participants + 1)
vmap_sibson = jax.vmap(jax.vmap(model.sibson_mi, in_axes=(0, None)), in_axes=(None, 0))
mi_matrix = vmap_sibson(designs, alpha_values)
optimal_designs = jnp.argmax(mi_matrix, axis=1)

for i, alpha in enumerate(alpha_values):
    if i in (0, len(alpha_values) // 2, len(alpha_values) - 1):
        print(f"  α = {alpha:.4f}... optimal n_A = {optimal_designs[i]}")

# ============================================================================
# Run Experiments
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING EXPERIMENTS")
print("=" * 80)


@jax.jit
def evaluate_all_alphas(
    alpha_vals: Array,
    opt_designs: Array,
    x_opt: Array,
    x_unif: Array,
    design_unif: Array,
    ds_test: Array,
    xs_test: Array,
):
    """JIT-ted evaluation of all alphas to avoid per-trial recompilation."""

    def evaluate_alpha(alpha, design_opt, x_opt_val):
        posterior_opt = model.get_tilted_posterior(x_opt_val, design_opt, alpha)
        log_preds_opt = jax.vmap(
            lambda d, x: model.get_posterior_predictive(posterior_opt, d, alpha=alpha).logpmf(x)
        )(ds_test, xs_test)

        posterior_unif = model.get_tilted_posterior(x_unif, design_unif, alpha)
        log_preds_unif = jax.vmap(
            lambda d, x: model.get_posterior_predictive(posterior_unif, d, alpha=alpha).logpmf(x)
        )(ds_test, xs_test)

        return jnp.mean(log_preds_opt), jnp.mean(log_preds_unif)

    return jax.vmap(evaluate_alpha)(alpha_vals, opt_designs, x_opt)


elpd_optimal = jnp.zeros((num_alphas, num_trials))
elpd_uniform = jnp.zeros((num_alphas, num_trials))

print(f"\nRunning {num_trials} trials...")

for trial in range(num_trials):
    if (trial + 1) % 200 == 0 or trial == 0:
        print(f"[{trial+1:4d}/{num_trials}]  Evaluating...")

    # Sample true parameter from prior
    rng_key, sub_key = jr.split(rng_key)
    theta_true = proc_prior.sample(sub_key)

    # Optimal designs: sample data for each α-specific design
    rng_key, sub_keys = custom_split(rng_key, num_alphas + 1)
    x_optimal = jax.vmap(lambda k, d: proc_lklhd.sample(k, d, theta_true))(
        sub_keys, optimal_designs
    )

    # Uniform design (single random allocation)
    rng_key, sub_key = jr.split(rng_key)
    design_uniform = generate_uniform_design(sub_key, num_participants)
    rng_key, sub_key = jr.split(rng_key)
    x_uniform = proc_lklhd.sample(sub_key, design_uniform, theta_true)

    # Test designs and observations
    rng_key, sub_keys = custom_split(rng_key, num_test_points + 1)
    designs_test = jax.vmap(lambda k: generate_uniform_design(k, num_participants))(
        sub_keys
    )

    rng_key, sub_keys = custom_split(rng_key, num_test_points + 1)
    x_test = jax.vmap(lambda k, d: proc_lklhd.sample(k, d, theta_true))(
        sub_keys, designs_test
    )

    # Evaluate all alphas using the JIT-ted function
    elpds_opt, elpds_unif = evaluate_all_alphas(
        alpha_values,
        optimal_designs,
        x_optimal,
        x_uniform,
        design_uniform,
        designs_test,
        x_test,
    )

    elpd_optimal = elpd_optimal.at[:, trial].set(elpds_opt)
    elpd_uniform = elpd_uniform.at[:, trial].set(elpds_unif)

print("✓ Complete")

# ============================================================================
# Analysis
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS: ELPD across α")
print("=" * 80)

mean_elpd_opt = jnp.mean(elpd_optimal, axis=1)
std_elpd_opt = jnp.std(elpd_optimal, axis=1)
se_elpd_opt = std_elpd_opt / jnp.sqrt(num_trials)
mean_elpd_unif = jnp.mean(elpd_uniform, axis=1)
std_elpd_unif = jnp.std(elpd_uniform, axis=1)
se_elpd_unif = std_elpd_unif / jnp.sqrt(num_trials)

print(f"\n{'α':>8} | {'Optimal (mean ± SE)':>20} | {'Uniform (mean ± SE)':>20}")
print("-" * 55)
for alpha_target in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    idx = jnp.argmin(jnp.abs(alpha_values - alpha_target))
    print(
        f"{alpha_values[idx]:>8.3f} | "
        f"{mean_elpd_opt[idx]:>10.4f} ± {se_elpd_opt[idx]:<7.4f} | "
        f"{mean_elpd_unif[idx]:>10.4f} ± {se_elpd_unif[idx]:<7.4f}"
    )

# ============================================================================
# Visualization + CSV
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

q05_opt = jnp.percentile(elpd_optimal, 5, axis=1)
q95_opt = jnp.percentile(elpd_optimal, 95, axis=1)
q05_unif = jnp.percentile(elpd_uniform, 5, axis=1)
q95_unif = jnp.percentile(elpd_uniform, 95, axis=1)

csv_path = "examples/ab_testing/data/abtesting_binomial_elpd.csv"
with open(csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        "alpha",
        "mean_opt",
        "se_opt",
        "q05_opt",
        "q95_opt",
        "mean_unif",
        "se_unif",
        "q05_unif",
        "q95_unif",
    ])
    for idx in range(num_alphas):
        writer.writerow([
            float(alpha_values[idx]),
            float(mean_elpd_opt[idx]),
            float(se_elpd_opt[idx]),
            float(q05_opt[idx]),
            float(q95_opt[idx]),
            float(mean_elpd_unif[idx]),
            float(se_elpd_unif[idx]),
            float(q05_unif[idx]),
            float(q95_unif[idx]),
        ])

ax.plot(alpha_values, mean_elpd_opt, "-", color="0.1", lw=2.5, label="Optimal")
ax.fill_between(alpha_values, q05_opt, q95_opt, alpha=0.2, color="0.1")

ax.plot(alpha_values, mean_elpd_unif, "-", color="0.5", lw=2.5, label="Uniform")
ax.fill_between(alpha_values, q05_unif, q95_unif, alpha=0.2, color="0.5")

ax.axvline(1.0, color="gray", linestyle=":", lw=1.5, label="Nominal (α=1)")

ax.set_xscale("log")
ax.set_xlabel("α", fontsize=12)
ax.set_ylabel("ELPD", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
