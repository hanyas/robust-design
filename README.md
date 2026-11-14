# Maximin Robust Bayesian Experimental Design

A JAX implementation of robust Bayesian experimental design under model misspecification. This repository accompanies the NeurIPS 2026 paper [Maximin Robust Bayesian Experimental Design](https://arxiv.org/abs/2603.14094).

The experimenter and an adversarial nature play a max–min game in which nature perturbs the data-generating process within a Kullback–Leibler neighborhood of the nominal model. The resulting robust objective is Sibson's $\alpha$-mutual information, the robust belief update is the $\alpha$-tilted posterior, and the conditional information gain is a Rényi divergence. Since Sibson's $\alpha$-mutual information must in general be estimated with a biased and noisy nested Monte Carlo estimator, designs are selected by optimizing a PAC-Bayes lower bound over stochastic design policies.

## Installation

Install [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) for your hardware, then run from the repository root:

```bash
pip install -e .
```

## Package Overview

The `rdesign` package provides:

- **Conjugate benchmark models** with closed-form robust quantities: linear-Gaussian regression (`rdesign/linear_regression.py`) and Beta-Binomial A/B testing (`rdesign/ab_testing.py`). Both expose the Bayes and $\alpha$-tilted posteriors, the $\alpha$-tilted marginal, Sibson's $\alpha$-mutual information, the Rényi information gain, and the power-likelihood Gibbs expected information gain used for comparison.
- **Nested Monte Carlo estimation** of Sibson's $\alpha$-mutual information in the log domain (`rdesign/utils.py`), with and without access to the marginal likelihood.
- **PAC-Bayesian design policies** (`rdesign/policy.py`) fitted with relative entropy policy search (`rdesign/reps.py`): a Gaussian policy with a tanh bijector for box-constrained continuous designs, and a categorical policy for discrete designs.

## Examples

The `examples` directory reproduces the numerical evaluation of the paper on two benchmark problems:

- `examples/linear_regression`: a continuous regression design problem evaluated under Student-t misspecification, with a ten-dimensional variant for design optimization.
- `examples/ab_testing`: a discrete allocation problem between two variants evaluated under a Beta-mixture Binomial process.

For each problem there are scripts that plot the robust information surface across $\alpha$, evaluate the realized information gain, posterior error, and credible-interval coverage under misspecification, and compare naive optimization of the Monte Carlo estimator against PAC-Bayes policies. Each script writes its results as a CSV file to the `data` folder next to it. Run them from the repository root, for example:

```bash
python examples/linear_regression/linreg_design_plot.py
```

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{abdulsamad2026maximin,
  title     = {Maximin Robust {B}ayesian Experimental Design},
  author    = {Abdulsamad, Hany and Iqbal, Sahel and Naesseth, Christian A. and Matsubara, Takuo and Corenflos, Adrien},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
