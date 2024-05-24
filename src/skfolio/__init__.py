"""shogunfolio package"""

# Author: Hugo Delatte <jialuechen@outlook.com>
# License: BSD 3 clause
import importlib.metadata

from shogunfolio.measures import (
    BaseMeasure,
    ExtraRiskMeasure,
    PerfMeasure,
    RatioMeasure,
    RiskMeasure,
)
from shogunfolio.population import Population
from shogunfolio.portfolio import BasePortfolio, MultiPeriodPortfolio, Portfolio

__version__ = importlib.metadata.version("shogunfolio")

__all__ = [
    "BaseMeasure",
    "PerfMeasure",
    "RiskMeasure",
    "ExtraRiskMeasure",
    "RatioMeasure",
    "BasePortfolio",
    "Portfolio",
    "MultiPeriodPortfolio",
    "Population",
]
