"""
Sweep over sample sizes comparing GD and REPS for linear regression design.
"""

import csv
import math
from functools import partial

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

LAMBDA = 1e6
ALPHA = 0.1  # Fixed alpha

NUM_GD_TRIALS = 256
NUM_REPS_SAMPLES = 256

vars = jnp.linspace(1.0, 10.0, DIM)
prior = Gaussian(mean=jnp.zeros(DIM), cov=jnp.diag(vars))
lklhd = HighDimLinearGaussian(noise_stddev=NOISE_STDDEV)
model = LinearRegression(prior=prior, likelihood=lklhd)

optimizer = optax.adam(learning_rate=1e-2)
bijector = Tanh()
policy = BoxGaussian(
    dim=DIM,
    bijector=Tanh(),
    init_mean=lambda k, s: jr.normal(k, s) * 0.1,
    init_log_std=lambda _, s: jnp.zeros(s),
)


@partial(jax.jit, static_argnames=("num_outer", "num_inner"))
def run_gradient_descent_jit(rng_keys: Array, num_outer: int, num_inner: int) -> Array:

    def optimize_single(rng_key):
        init_key, opt_key = jr.split(rng_key)
        params = jr.normal(init_key, (DIM,))
        opt_state = optimizer.init(params)

        def objective_fn(p, key):
            xi = bijector.forward(p)
            return -nested_mc_sibson_mi(
                rng_key=key,
                model=model,
                design=xi,
                num_outer=num_outer,
                num_inner=num_inner,
                alpha=ALPHA,
            )

        def step(carry, _):
            p, state, k = carry
            obj_key, next_key = jr.split(k)
            val, grads = jax.value_and_grad(objective_fn)(p, obj_key)
            updates, new_state = optimizer.update(grads, state)
            new_p = optax.apply_updates(p, updates)
            return (new_p, new_state, next_key), val

        (final_p, _, _), _ = jax.lax.scan(step, (params, opt_state, opt_key), jnp.arange(1000))
        return bijector.forward(final_p)

    return jax.vmap(optimize_single)(rng_keys)


def run_gradient_descent(
    num_outer: int,
    num_inner: int,
    num_trials: int,
    seed: int = 42,
):
    keys = jr.split(jr.PRNGKey(seed), num_trials)
    return run_gradient_descent_jit(keys, num_outer, num_inner)


def run_policy_search(num_outer: int, num_inner: int, seed: int = 42):
    config = REPSConfig(
        num_samples=256,
        batch_size=32,
        max_iterations=1000,
        log_interval=10000,  # Silence logs
        verbose=False,
        lmbda=LAMBDA,
    )

    def objective_fn(key, design, log_prob, lmbda):
        mi = nested_mc_sibson_mi(
            rng_key=key,
            model=model,
            design=design,
            num_outer=num_outer,
            num_inner=num_inner,
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
    return policy, result.state


@jax.jit
def sample_reps_designs(sample_keys: Array, params) -> Array:
    return jax.vmap(lambda k: policy.sample(k, params))(sample_keys)


@jax.jit
def compute_regret(designs: Array) -> Array:
    true_opt = jnp.ones(DIM)
    true_max_val = model.sibson_mi(true_opt, ALPHA)
    found_vals = jax.vmap(lambda d: model.sibson_mi(d, ALPHA))(designs)
    return true_max_val - found_vals


N_values = [16, 32, 64, 128, 256]
M_fixed = 16

results = []

print(f"Starting sweep over N values: {N_values} with fixed M={M_fixed}")
print(f"Fixed Alpha = {ALPHA}")

for num_outer in N_values:
    num_inner = M_fixed
    budget = num_outer * num_inner
    print(f"Processing Budget T={budget} (N={num_outer}, M={num_inner})...")

    # GD
    print("  Running GD...")
    gd_designs = run_gradient_descent(num_outer, num_inner, num_trials=NUM_GD_TRIALS)
    gd_regrets = compute_regret(gd_designs)
    gd_mean = jnp.mean(gd_regrets)
    gd_p10, gd_p50, gd_p90 = jnp.percentile(gd_regrets, jnp.array([10, 50, 90]))

    # REPS
    print("  Running REPS...")
    _, state = run_policy_search(num_outer, num_inner)
    sample_key = jr.PRNGKey(42)
    sample_keys = jr.split(sample_key, NUM_REPS_SAMPLES)
    reps_designs = sample_reps_designs(sample_keys, state.params)
    reps_regrets = compute_regret(reps_designs)
    reps_mean = jnp.mean(reps_regrets)
    reps_p10, reps_p50, reps_p90 = jnp.percentile(reps_regrets, jnp.array([10, 50, 90]))

    print(f"  Result: GD mean={gd_mean:.4f}, REPS mean={reps_mean:.4f}")

    results.append(
        {
            "budget": budget,
            "N": num_outer,
            "M": num_inner,
            "gd_mean": float(gd_mean),
            "gd_p10": float(gd_p10),
            "gd_p50": float(gd_p50),
            "gd_p90": float(gd_p90),
            "reps_mean": float(reps_mean),
            "reps_p10": float(reps_p10),
            "reps_p50": float(reps_p50),
            "reps_p90": float(reps_p90),
        }
    )

# Save to CSV
csv_path = "examples/linear_regression/data/linreg_sweep_samples.csv"
fieldnames = [
    "budget",
    "N",
    "M",
    "gd_mean",
    "gd_p10",
    "gd_p50",
    "gd_p90",
    "reps_mean",
    "reps_p10",
    "reps_p50",
    "reps_p90",
]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Results saved to {csv_path}")
