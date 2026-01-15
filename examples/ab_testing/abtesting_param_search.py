import jax
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple
from rdesign.distributions import Beta
from rdesign.ab_testing import ABTesting, Binomial
from jax.scipy.special import betainc

jax.config.update("jax_enable_x64", True)

# --- The core logic extracted for massive vectorization ---

def compute_coverage(rng_key, sA, sB, N, a, gA, gB, num_trials):
    """
    Runs num_trials for a SINGLE hyperparameter setting, but fully JIT-able.
    """
    prior = Beta(delta=jnp.array([1.0, 1.0]), gamma=jnp.array([gA, gB]))
    model = Binomial(num_participants=N)
    abtest = ABTesting(prior=prior, likelihood=model)
    
    # These use vmap internally but are fast
    nA_opt_rob = abtest.optimal_design_sibson(a)

    def run_single_trial(key):
        key_theta, key_design, key_obs_opt, key_obs_rand = jr.split(key, 4)
        
        # Sample
        theta_true = prior.sample(key_theta)
        
        # Optimal Design Trial
        trials_opt = model.get_trials(nA_opt_rob)
        
        def sample_bb(key, n, mean, conc):
            conc = jnp.maximum(conc, 1e-4)
            a_eff = mean * conc
            b_eff = (1.0 - mean) * conc
            rate = jr.beta(key, a_eff, b_eff)
            return jr.binomial(key, n.astype(int), rate)
        
        x_opt = jnp.array([
            sample_bb(key_obs_opt, trials_opt[0], theta_true[0], sA),
            sample_bb(key_obs_opt, trials_opt[1], theta_true[1], sB)
        ])
        
        # Random Design Trial
        nA_rand = jr.randint(key_design, (), 0, N)
        trials_rand = model.get_trials(nA_rand)
        x_rand = jnp.array([
            sample_bb(key_obs_rand, trials_rand[0], theta_true[0], sA),
            sample_bb(key_obs_rand, trials_rand[1], theta_true[1], sB)
        ])

        # Posteriors
        post_opt = abtest.get_tilted_posterior(x_opt, nA_opt_rob, a)
        post_rand = abtest.get_tilted_posterior(x_rand, nA_rand, a)

        # CDF values
        u_opt = betainc(post_opt.delta, post_opt.gamma, theta_true)
        u_rand = betainc(post_rand.delta, post_rand.gamma, theta_true)
        
        return jnp.stack([u_opt, u_rand])

    trial_keys = jr.split(rng_key, num_trials)
    all_cdfs = jax.vmap(run_single_trial)(trial_keys) # (num_trials, 2, 2)
    
    # Calculate coverage for 50% interval
    in_opt = (all_cdfs[:, 0, :] >= 0.25) & (all_cdfs[:, 0, :] <= 0.75)
    in_rand = (all_cdfs[:, 1, :] >= 0.25) & (all_cdfs[:, 1, :] <= 0.75)
    
    return jnp.mean(in_opt), jnp.mean(in_rand), nA_opt_rob

# --- Grid Generation ---
# Reasoning:
# 1. Asymmetric noise (sA != sB) should force the robust design to be more selective.
# 2. Large disparity in prior (gA vs gB) makes the information gain asymmetric.
# 3. Small N emphasizes the impact of design.

sA_vals = jnp.array([5.0, 20.0, 100.0]) # More extreme noise
sB_vals = jnp.array([2000.0]) # Even cleaner
N_vals = jnp.array([100, 300, 1000])
a_vals = jnp.array([0.01, 0.05, 0.1, 0.2]) # Smaller alphas
gA_vals = jnp.array([1.1, 5.0, 50.0]) # Extreme prior disparity
gB_vals = jnp.array([10.0])

# Create a flattened grid
SA, SB, A, GA, GB = jnp.meshgrid(sA_vals, sB_vals, a_vals, gA_vals, gB_vals, indexing='ij')
grid_sA = SA.flatten()
grid_sB = SB.flatten()
grid_a = A.flatten()
grid_ga = GA.flatten()
grid_gb = GB.flatten()

num_trials = 10000
rng_key = jr.PRNGKey(42)

print(f"Starting vectorized search over {len(grid_sA)} hyperparam combinations per N...")

best_gap = -1.0
best_config = None

for N in N_vals:
    print(f"Checking N={int(N)}...")
    def batch_experiment(sA, sB, a, ga, gb, key):
        return compute_coverage(key, sA, sB, int(N), a, ga, gb, num_trials)
    
    grid_keys = jr.split(rng_key, len(grid_sA))
    v_cov_opt, v_cov_rand, v_nA = jax.jit(jax.vmap(batch_experiment))(grid_sA, grid_sB, grid_a, grid_ga, grid_gb, grid_keys)
    
    for i in range(len(grid_sA)):
        cov_opt = float(v_cov_opt[i])
        cov_rand = float(v_cov_rand[i])
        gap = cov_opt - cov_rand
        
        # Criteria: Both conservative (>0.50), and Opt more conservative than Rand
        if cov_opt > 0.50 and cov_rand > 0.50 and gap > 0.01: # Use a minimum gap of 1%
            if gap > best_gap:
                best_gap = gap
                best_config = {
                    'sA': float(grid_sA[i]),
                    'sB': float(grid_sB[i]),
                    'N': int(N),
                    'alpha': float(grid_a[i]),
                    'gammaA': float(grid_ga[i]),
                    'gammaB': float(grid_gb[i]),
                    'cov_opt': cov_opt,
                    'cov_rand': cov_rand,
                    'nA_opt': int(v_nA[i]),
                    'gap': gap
                }

print("\n" + "=" * 60)
print("BEST CONFIGURATION FOUND")
print("=" * 60)
if best_config:
    print(f"Concentration A (sA): {best_config['sA']:.1f}")
    print(f"Concentration B (sB): {best_config['sB']:.1f}")
    print(f"Participants (N):      {best_config['N']}")
    print(f"Alpha Robust:          {best_config['alpha']:.3f}")
    print(f"Prior Gamma A:         {best_config['gammaA']:.1f}")
    print(f"Prior Gamma B:         {best_config['gammaB']:.1f}")
    print("-" * 30)
    print(f"Optimal Coverage (50%):  {best_config['cov_opt']:.4f}")
    print(f"Random Coverage (50%):   {best_config['cov_rand']:.4f}")
    print(f"Safety Gap:              {best_config['gap']:.4f}")
    print(f"Optimal n_A:             {best_config['nA_opt']}")
else:
    print("No configuration met the criteria (both > 0.50 and Opt > Rand).")
print("=" * 60)
