"""deepfolio package"""

# Author: Hugo Delatte <jialuechen@outlook.com>
# License: BSD 3 clause
import importlib.metadata

from deepfolio.measures import (
    BaseMeasure,
    ExtraRiskMeasure,
    PerfMeasure,
    RatioMeasure,
    RiskMeasure,
)
from deepfolio.population import Population
from deepfolio.portfolio import BasePortfolio, MultiPeriodPortfolio, Portfolio

__version__ = importlib.metadata.version("deepfolio")

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
