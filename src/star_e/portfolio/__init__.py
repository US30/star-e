"""Portfolio optimization and risk management module."""

from star_e.portfolio.optimizer import PortfolioOptimizer
from star_e.portfolio.cointegration import (
    johansen_test,
    find_cointegrated_pairs,
    calculate_spread,
)
from star_e.portfolio.risk import (
    calculate_var,
    calculate_cvar,
    max_drawdown,
)
from star_e.portfolio.metrics import (
    sharpe_ratio,
    sortino_ratio,
    information_ratio,
    calmar_ratio,
)

__all__ = [
    "PortfolioOptimizer",
    "johansen_test",
    "find_cointegrated_pairs",
    "calculate_spread",
    "calculate_var",
    "calculate_cvar",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "information_ratio",
    "calmar_ratio",
]
