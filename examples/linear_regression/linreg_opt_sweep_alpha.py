"""
Sweep over alpha values comparing GD and REPS for linear regression design.
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
from rdesign.utils import nested_mc_sibson_mi
from rdesign.policy import BoxGaussian
from rdesign.reps import optimize_policy
from rdesign.bijectors import Tanh

jax.config.update("jax_enable_x64", True)

# --- Problem Setup ---
class HighDimLinearGaussian(LinearGaussian):

    def design_matrix(self, design: Array) -> Array:
        return design.reshape(1, -1)


DIM = 10
NOISE_STDDEV = 1e-1

LAMBDA = 1e6
NUM_OUTER = 16
NUM_INNER = 16

NUM_TRIALS_GD = 256
NUM_SAMPLES_REPS = 256

vars = jnp.linspace(1.0, 10.0, DIM)
prior = Gaussian(mean=jnp.zeros(DIM), cov=jnp.diag(vars))
lklhd = HighDimLinearGaussian(noise_stddev=NOISE_STDDEV)
model = LinearRegression(prior=prior, likelihood=lklhd)

# --- Functions ---

def run_gradient_descent(alpha: float, num_trials: int, seed: int = 42):
    optimizer = optax.adam(learning_rate=1e-2)
    bijector = Tanh()

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
                num_outer=NUM_OUTER,
                num_inner=NUM_INNER,
                alpha=alpha
            )

        def step(carry, i):
            p, state, k = carry
            obj_key, next_key = jr.split(k)
            val, grads = jax.value_and_grad(objective_fn)(p, obj_key)
            updates, new_state = optimizer.update(grads, state)
            new_p = optax.apply_updates(p, updates)
            return (new_p, new_state, next_key), val

        (final_p, _, _), _ = jax.lax.scan(
            step, (params, opt_state, opt_key), jnp.arange(1000)
        )
        return bijector.forward(final_p)

    keys = jr.split(jr.PRNGKey(seed), num_trials)
    return jax.jit(jax.vmap(optimize_single))(keys)


def run_policy_search(alpha: float, seed: int = 42):
    policy = BoxGaussian(
        dim=DIM,
        bijector=Tanh(),
        init_mean=lambda k, s: jr.normal(k, s) * 0.1,
        init_log_std=lambda k, s: jnp.zeros(s)
    )

    config = REPSConfig(
        num_samples=256,
        batch_size=32,
        max_iterations=1000,
        log_interval=10000, # Silence logs
        verbose=False,
        lmbda=LAMBDA
    )

    def objective_fn(key, design, log_prob, lmbda):
        mi = nested_mc_sibson_mi(
            rng_key=key,
            model=model,
            design=design,
            num_outer=NUM_OUTER,
            num_inner=NUM_INNER,
            alpha=alpha
        )
        return mi - (1.0 / lmbda) * log_prob

    rng_key = jr.PRNGKey(seed)
    init_key, run_key = jr.split(rng_key)
    optimizer = optax.adam(learning_rate=1e-2)
    policy_state = policy.init_state(init_key, optimizer)

    result = optimize_policy(
        rng_key=run_key,
        func=objective_fn,
        policy=policy,
        policy_state=policy_state,
        config=config
    )
    return policy, result.policy_state

def compute_relative_regret(designs, alpha):
    true_opt = jnp.ones(DIM)
    true_max_val = model.sibson_mi(true_opt, alpha)
    found_vals = jax.vmap(lambda d: model.sibson_mi(d, alpha))(designs)
    return (true_max_val - found_vals) / true_max_val

# --- Main Sweep ---
alphas = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 0.9995]
results = []

print(f"Starting sweep over alphas: {alphas}")

for alpha in alphas:
    print(f"Processing alpha={alpha}...")

    # GD
    gd_designs = run_gradient_descent(alpha, num_trials=NUM_TRIALS_GD)
    gd_rel_regrets = compute_relative_regret(gd_designs, alpha)
    gd_mean = jnp.mean(gd_rel_regrets)
    gd_p10, gd_p50, gd_p90 = jnp.percentile(gd_rel_regrets, jnp.array([10, 50, 90]))

    # REPS
    policy, policy_state = run_policy_search(alpha)
    sample_key = jr.PRNGKey(42)
    sample_keys = jr.split(sample_key, NUM_SAMPLES_REPS)
    reps_designs = jax.vmap(lambda k: policy.sample(k, policy_state.params))(sample_keys)
    reps_rel_regrets = compute_relative_regret(reps_designs, alpha)
    reps_mean = jnp.mean(reps_rel_regrets)
    reps_p10, reps_p50, reps_p90 = jnp.percentile(reps_rel_regrets, jnp.array([10, 50, 90]))

    print(f"  alpha={alpha}: GD mean={gd_mean:.4f}, REPS mean={reps_mean:.4f}")

    results.append({
        "alpha": alpha,
        "gd_mean": float(gd_mean),
        "gd_p10": float(gd_p10),
        "gd_p50": float(gd_p50),
        "gd_p90": float(gd_p90),
        "reps_mean": float(reps_mean),
        "reps_p10": float(reps_p10),
        "reps_p50": float(reps_p50),
        "reps_p90": float(reps_p90),
    })

# Save to CSV
csv_path = "examples/linear_regression/data/linreg_sweep_alpha.csv"
fieldnames = ["alpha", "gd_mean", "gd_p10", "gd_p50", "gd_p90", "reps_mean", "reps_p10", "reps_p50", "reps_p90"]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Results saved to {csv_path}")
