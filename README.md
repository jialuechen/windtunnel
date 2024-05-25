# shogunfolio : Portfolio Optimization with Machine Learning
**shogunfolio** is a Python library for portfolio optimization built on top of shogun machine learning library.
It offers a unified interface and tools compatible with shogun to build,fine-tune, and cross-validate portfolio models.

## Installation

```
pip install --upgrade shogunfolio
```
## Available models

* Portfolio Optimization:
    * Naive:
        * Equal-Weighted
        * Random (Dirichlet)
    * Convex:
        * Mean-Risk
        * Distributionally Robust CVaR
    * Clustering:
        * Hierarchical Risk Parity
        * Hierarchical Equal Risk Contribution
        * Nested Clusters Optimization

* Expected Returns Estimator:
    * Empirical
    * Equilibrium
    * Shrinkage

* Distance Estimator:
    * Pearson Distance
    * Kendall Distance
    * Variation of Information

* Pre-Selection Transformer:
    * Non-Dominated Selection
    * Select K Extremes (Best or Worst)
    * Drop Highly Correlated Assets

* Risk Measures:
    * Variance
    * Semi-Variance
    * Mean Absolute Deviation
    * Skew
    * Kurtosis

* Cross-Validation and Model Selection:
    * Walk Forward
    * Combinatorial Purged Cross-Validation

* Optimization Features:
    * Minimize Risk
    * Transaction Costs
    * L1 and L2 Regularization
    * Weight Constraints
    * Tracking Error Constraints
    * Turnover Constraints