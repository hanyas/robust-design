"""
Evaluate the squared bias, variance, MSE, and relative squared error of the MC estimator for Sibson's MI against the sampling budget.
"""
from functools import partial
import csv
import os

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
import matplotlib.pyplot as plt

from rdesign.distributions import Gaussian
from rdesign.linear_regression import LinearGaussian
from rdesign.linear_regression import LinearRegression
from rdesign.utils import nested_mc_sibson_mi

jax.config.update("jax_enable_x64", True)


# --- Settings ---
DIM = 10
NOISE_STDDEV = 1e-1

ALPHAS = jnp.array([0.01, 0.1, 0.9995])
NUM_REPEATS = 256

# Budget Generation: 1..9 * 10^k
budgets = []
for k in range(2, 5): # 10^2 to 10^4
    for d in range(1, 10):
        budgets.append(d * 10**k)
budgets.append(10**5)

# --- Model Setup ---
class HighDimLinearGaussian(LinearGaussian):
    def design_matrix(self, design: Array) -> Array:
        return design.reshape(1, -1)

variances = jnp.linspace(1.0, 10.0, DIM)
prior = Gaussian(mean=jnp.zeros(DIM), cov=jnp.diag(variances))
lklhd = HighDimLinearGaussian(noise_stddev=NOISE_STDDEV)
model = LinearRegression(prior=prior, likelihood=lklhd)

# Ground Truth
design = jnp.ones(DIM)
# Vectorize over alphas for ground truth
true_vals = jax.vmap(lambda a: model.sibson_mi(design, a))(ALPHAS)
print("Ground Truth Sibson MI values:")
for a, val in zip(ALPHAS, true_vals):
    print(f"  alpha={a:.3f}: {val:.6f}")

# --- Helper Functions ---

@partial(jax.jit, static_argnums=(1, 2))
def run_estimation_batch(rng_key, num_outer, num_inner, alphas_arr):
    """Run a batch of estimations for all alphas."""
    keys = jr.split(rng_key, NUM_REPEATS)

    def single_est(k):
        def per_alpha(a):
            return nested_mc_sibson_mi(
                rng_key=k,
                model=model,
                design=design,
                num_outer=num_outer,
                num_inner=num_inner,
                alpha=a
            )
        return jax.vmap(per_alpha)(alphas_arr)

    return jax.vmap(single_est)(keys)

# --- Main Loop ---
results = []
print(f"Starting evaluation on {len(budgets)} budget points for {len(ALPHAS)} alphas...")

seed_counter = 0

for T in budgets:
    seed_counter += 1
    rng_key = jr.PRNGKey(seed_counter)

    # Strategy: N = M
    N = int(round(jnp.sqrt(T)))
    M = int(round(T / N)) if N > 0 else 0

    if N > 0 and M > 0:
        # estimates shape: (NUM_REPEATS, NUM_ALPHAS)
        est = run_estimation_batch(rng_key, N, M, ALPHAS)

        # Calculate metrics for each alpha
        for i, alpha in enumerate(ALPHAS):
            # est[:, i] corresponds to alpha
            estimates_alpha = est[:, i]
            true_val = true_vals[i]

            # Bias^2 = (E[est] - true)^2
            mean_est = jnp.mean(estimates_alpha)
            bias_sq = (mean_est - true_val)**2
            bias_abs = jnp.abs(mean_est - true_val)

            # Variance = E[(est - E[est])^2]
            variance = jnp.var(estimates_alpha)

            # MSE = E[(est - true)^2] = Bias^2 + Variance
            mse = jnp.mean((estimates_alpha - true_val)**2)

            results.append({
                "budget": T,
                "alpha": float(alpha),
                "N": N,
                "M": M,
                "bias_sq": float(bias_sq),
                "bias_abs": float(bias_abs),
                "variance": float(variance),
                "mse": float(mse),
            })

    print(f"Budget {T}: Done.")

# --- Save Results ---
csv_path = "examples/linear_regression/data/linreg_mc_error.csv"
fieldnames = ["budget", "alpha", "N", "M", "bias_sq", "bias_abs", "variance", "mse"]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Results saved to {csv_path}")

# --- Plotting ---
print("Generating plots...")
# Organize data for plotting
plot_data = {}
for a in ALPHAS:
    val = float(a)
    plot_data[val] = {'budgets': [], 'bias_sq': [], 'bias_abs': [], 'variance': [], 'mse': []}

for row in results:
    a = row['alpha']
    plot_data[a]['budgets'].append(row['budget'])
    plot_data[a]['bias_sq'].append(row['bias_sq'])
    plot_data[a]['bias_abs'].append(row['bias_abs'])
    plot_data[a]['variance'].append(row['variance'])
    plot_data[a]['mse'].append(row['mse'])

# Create 4 subplots: Bias^2, Bias Abs, Variance, MSE
fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
metrics = ['bias_sq', 'bias_abs', 'variance', 'mse']
titles = ['Squared Bias', 'Absolute Bias', 'Variance', 'Mean Squared Error']
colors = ["blue", "red", "green", "purple"]

for i, metric in enumerate(metrics):
    ax = axes[i]
    for j, alpha in enumerate(ALPHAS):
        val = float(alpha)
        data = plot_data[val]

        if not data['budgets']:
            continue

        bs = data['budgets']
        vals = data[metric]
        c = colors[j % len(colors)]
        label_str = f"Alpha = {val:.3f}"

        ax.plot(bs, vals, label=label_str, color=c, marker='o', markersize=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(titles[i])
    ax.set_xlabel("Budget T")
    ax.set_ylabel(metric)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    if i == 0:
        ax.legend()

plt.show()