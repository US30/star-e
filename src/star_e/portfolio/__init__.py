"""Portfolio optimization, risk management, and graph neural networks module."""

from star_e.portfolio.optimizer import PortfolioOptimizer
from star_e.portfolio.cointegration import (
    johansen_test,
    find_cointegrated_pairs,
    calculate_spread,
    engle_granger_test,
    find_cointegrated_pairs_eg,
    granger_causality_test,
    bidirectional_granger,
    build_causality_network,
    cointegration_summary,
)
from star_e.portfolio.risk import (
    calculate_var,
    calculate_cvar,
    max_drawdown,
    monte_carlo_var,
    monte_carlo_paths,
    stress_test_var,
    incremental_var,
    component_var,
    expected_shortfall_decomposition,
)
from star_e.portfolio.metrics import (
    sharpe_ratio,
    sortino_ratio,
    information_ratio,
    calmar_ratio,
)
from star_e.portfolio.gat import (
    AssetGATEncoder,
    AssetCorrelationGraph,
    GATClusterer,
    cluster_portfolio_with_gat,
)

__all__ = [
    # Portfolio optimization
    "PortfolioOptimizer",
    # Cointegration - Johansen
    "johansen_test",
    "find_cointegrated_pairs",
    "calculate_spread",
    # Cointegration - Engle-Granger
    "engle_granger_test",
    "find_cointegrated_pairs_eg",
    # Granger Causality
    "granger_causality_test",
    "bidirectional_granger",
    "build_causality_network",
    "cointegration_summary",
    # Risk metrics - Analytical
    "calculate_var",
    "calculate_cvar",
    "max_drawdown",
    # Risk metrics - Monte Carlo
    "monte_carlo_var",
    "monte_carlo_paths",
    "stress_test_var",
    "incremental_var",
    "component_var",
    "expected_shortfall_decomposition",
    # Performance metrics
    "sharpe_ratio",
    "sortino_ratio",
    "information_ratio",
    "calmar_ratio",
    # Graph Attention Networks
    "AssetGATEncoder",
    "AssetCorrelationGraph",
    "GATClusterer",
    "cluster_portfolio_with_gat",
]
