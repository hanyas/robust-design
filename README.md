# Robust Bayesian Experimental Design

Implementation of robust Bayesian experimental design methods based on Sibson's $\alpha$-mutual information.

## Installation

Install [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) for your available hardware. Then run:

```bash
pip install -e .
```

for an editable install.

## Features

- **Robust Information Measures**: Implementation of Sibson's $\alpha$-Mutual Information and Rényi Information Gain.
- **Policy Optimization**: PAC-Bayesian policy optimization using Relative Entropy Policy Search (REPS).
- **Tractable Benchmarks**: Experimental design for conjugate linear regression and A/B testing

## Examples

We provide several experiments to demonstrate the behavior of robust designs:

### Linear Regression
Located in `examples/linear_regression/`:
- `linreg_design_plot.py`: Visualize the information surfaces for different values of $\alpha$.
- `linreg_gain_studentt.py`: Evaluate the information gain under Student-t misspecification.
- `linreg_mse_studentt.py`: Evaluate the posterior mean MSE under Student-t misspecification.
- `linreg_coverage_studentt.py`: Analyze credible-interval coverage under Student-t misspecification.
- `linreg_opt_comp.py`: Compare stochastic policy optimization to gradient descent.
- `linreg_opt_sweep_alpha.py`: Sweep over $\alpha$ comparing GD and REPS.
- `linreg_opt_sweep_lambda.py`: Sensitivity sweep over the PAC-Bayes precision $\lambda$.
- `linreg_opt_sweep_samples.py`: Sweep over sample sizes comparing GD and REPS.

### A/B Testing
Located in `examples/ab_testing/`:
- `abtesting_design_plot.py`: Visualize the information surfaces for different values of $\alpha$.
- `abtesting_gain_mixture.py`: Evaluate the information gain under a Beta-mixture Binomial likelihood.
- `abtesting_mse_mixture.py`: Evaluate the posterior mean MSE under a Beta-mixture Binomial likelihood.
- `abtesting_mse_mixture_geig.py`: Same as above using the generalized eigenvalue formulation.
- `abtesting_coverage_mixture.py`: Analyze credible-interval coverage under a Beta-mixture Binomial likelihood.
- `abtesting_opt_comp.py`: Compare stochastic policy optimization to exhaustive enumeration.


To run an example, use:
```bash
python examples/ab_testing/abtesting_design_plot.py
```