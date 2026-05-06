"""
Sensitivity sweep over the PAC-Bayes precision parameter lambda.
"""

import csv

import jax
import optax

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian
from rdesign.linear_regression import LinearRegression

from rdesign.base import REPSConfig
from rdesign.utils import nested_mc_sibson_mi_with_marginal as nested_mc_sibson_mi
from rdesign.policy import BoxGaussian
from rdesign.reps import optimize_policy
from rdesign.bijectors import Tanh

jax.config.update("jax_enable_x64", True)


class HighDimLinearGaussian(LinearGaussian):

    def design_matrix(self, design: Array) -> Array:
        return design.reshape(1, -1)


DIM = 10
NOISE_STDDEV = 1e-1
ALPHA = 0.1

NUM_INNER = 16
NUM_REPS_SAMPLES = 4096

vars = jnp.linspace(1.0, 10.0, DIM)
prior = Gaussian(mean=jnp.zeros(DIM), cov=jnp.diag(vars))
lklhd = HighDimLinearGaussian(noise_stddev=NOISE_STDDEV)
model = LinearRegression(prior=prior, likelihood=lklhd)

optimizer = optax.adam(learning_rate=1e-2)
policy = BoxGaussian(
    dim=DIM,
    bijector=Tanh(),
    init_mean=lambda k, s: jr.normal(k, s) * 0.1,
    init_log_std=lambda _, s: jnp.zeros(s),
)


def run_policy_search(num_outer: int, lmbda: float, seed: int = 42):
    config = REPSConfig(
        num_samples=256,
        batch_size=32,
        max_iterations=1000,
        log_interval=10000,  # Silence logs
        verbose=False,
        lmbda=lmbda,
    )

    def objective_fn(key, design, log_prob, lmbda):
        mi = nested_mc_sibson_mi(
            rng_key=key,
            model=model,
            design=design,
            num_outer=num_outer,
            num_inner=NUM_INNER,
            alpha=ALPHA,
        )
        return mi - (1.0 / lmbda) * log_prob

    rng_key = jr.PRNGKey(seed)
    init_key, run_key = jr.split(rng_key)
    state = policy.init_state(init_key, optimizer)

    result = optimize_policy(
        rng_key=run_key,
        func=objective_fn,
        policy=policy,
        state=state,
        config=config,
    )
    return result.state


@jax.jit
def sample_reps_designs(sample_keys: Array, params) -> Array:
    return jax.vmap(lambda k: policy.sample(k, params))(sample_keys)


@jax.jit
def compute_relative_regret(designs: Array) -> Array:
    true_opt = jnp.ones(DIM)
    true_max_val = model.sibson_mi(true_opt, ALPHA)
    found_vals = jax.vmap(lambda d: model.sibson_mi(d, ALPHA))(designs)
    return (true_max_val - found_vals) / true_max_val


LAMBDA_GRID = jnp.logspace(2, 8, 7)
N_VALUES = [64, 128, 256]

results = []

print(f"Starting lambda sensitivity sweep")
print(f"  lambda grid: {LAMBDA_GRID.shape[0]} points in [{float(LAMBDA_GRID[0]):.1e}, {float(LAMBDA_GRID[-1]):.1e}]")
print(f"  N values: {N_VALUES}")
print(f"  Total cells: {LAMBDA_GRID.shape[0] * len(N_VALUES)}")

for num_outer in N_VALUES:
    for lmbda in LAMBDA_GRID:
        lmbda_float = float(lmbda)
        print(f"Processing N={num_outer}, lambda={lmbda_float:.3e}...")

        state = run_policy_search(num_outer=num_outer, lmbda=lmbda_float)

        sample_key = jr.PRNGKey(42)
        sample_keys = jr.split(sample_key, NUM_REPS_SAMPLES)
        reps_designs = sample_reps_designs(sample_keys, state.params)
        reps_rel_regrets = compute_relative_regret(reps_designs)

        reps_mean = jnp.mean(reps_rel_regrets)
        reps_p10, reps_p50, reps_p90 = jnp.percentile(reps_rel_regrets, jnp.array([10, 50, 90]))

        print(
            f"  N={num_outer}, lambda={lmbda_float:.3e}: "
            f"REPS mean={float(reps_mean):.4f}, p50={float(reps_p50):.4f}"
        )

        results.append(
            {
                "lambda": lmbda_float,
                "N": num_outer,
                "reps_mean": float(reps_mean),
                "reps_p10": float(reps_p10),
                "reps_p50": float(reps_p50),
                "reps_p90": float(reps_p90),
            }
        )

# Save to CSV
csv_path = "examples/linear_regression/data/linreg_sweep_lambda.csv"
fieldnames = ["lambda", "N", "reps_mean", "reps_p10", "reps_p50", "reps_p90"]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Results saved to {csv_path}")
