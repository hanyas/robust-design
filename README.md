# Robust Bayesian Experimental Design

Implementation of robust Bayesian experimental design methods based on Sibson's $\alpha$-mutual information.

## Installation

Install [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) for your available hardware. Then run:

```bash
pip install -e .
```

for an editable install.

## Core Features

- **Robust Information Measures**: Implementation of Sibson's $\alpha$-Mutual Information and Rényi Information Gain.
- **Policy Optimization**: PAC-Bayesian policy optimization using Relative Entropy Policy Search (REPS).
- **Tractable Benchmarks**: Experimental design for conjugate linear regression and A/B testing 

## Examples

We provide several experiments to demonstrate the behavior of robust designs:

### Linear Regression
Located in `examples/linear_regression/`:
- `linreg_design.py`: Visualize the information surfaces for different values of $\alpha$.
- `linreg_infogain.py`: Evaluate the information gain under misspecification.
- `linreg_coverage.py`: Analyze the coverage of Bayesian credible intervals.
- `linreg_opt_reps.py`: Optimize designs using Relative Entropy Policy Search.
- `linreg_opt_comp.py`: Compare stochastic policy optimization to gradient descent.

### A/B Testing
Located in `examples/ab_testing/`:
- `abtesting_design.py`: Visualize the information surfaces for different values of $\alpha$.
- `abtesting_infogain.py`: Evaluate the information gain under misspecification.
- `abtesting_coverage.py`: Evaluate coverage performance for robust A/B tests.
- `abtesting_opt_comp.py`: Compare stochastic policy optimization to exhaustive enumeration.


To run an example, use:
```bash
python examples/ab_testing/abtesting_design.py
```