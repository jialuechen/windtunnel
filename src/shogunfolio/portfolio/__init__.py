"""Portfolio module.
`Portfolio` and `MultiPeriodPortfolio` objects are returned by the `predict` method of
Optimization estimators.
They need to be homogenous to the convex optimization problems meaning that `Portfolio`
is the dot product of the assets weights with the assets returns and
`MultiPeriodPortfolio` is a list of `Portfolio`.
"""

from shogunfolio.portfolio._base import BasePortfolio
from shogunfolio.portfolio._multi_period_portfolio import MultiPeriodPortfolio
from shogunfolio.portfolio._portfolio import Portfolio

__all__ = ["BasePortfolio", "Portfolio", "MultiPeriodPortfolio"]
