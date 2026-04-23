"""Backtesting engine for strategy evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

import numpy as np
import pandas as pd
import polars as pl

from star_e.portfolio.metrics import calculate_all_metrics
from star_e.portfolio.risk import calculate_portfolio_risk, max_drawdown


@dataclass
class BacktestResult:
    """Container for backtest results."""

    # Returns
    portfolio_returns: np.ndarray
    cumulative_returns: np.ndarray
    benchmark_returns: Optional[np.ndarray] = None

    # Weights history
    weights_history: Optional[np.ndarray] = None
    rebalance_dates: Optional[list] = None

    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Additional info
    n_trades: int = 0
    turnover: float = 0.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # All metrics
    metrics: dict = field(default_factory=dict)


class BacktestEngine:
    """
    Backtesting engine for portfolio strategies.

    Supports:
    - Walk-forward testing
    - Transaction costs
    - Rebalancing schedules
    - Benchmark comparison
    - Comprehensive metrics calculation
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        transaction_cost: float = 0.001,
        rebalance_frequency: str = "monthly",
        risk_free_rate: float = 0.04,
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            rebalance_frequency: "daily", "weekly", "monthly", or "quarterly"
            risk_free_rate: Annual risk-free rate
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate

    def _get_rebalance_mask(
        self,
        dates: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Generate boolean mask for rebalance dates."""
        mask = np.zeros(len(dates), dtype=bool)
        mask[0] = True  # Always rebalance on first day

        if self.rebalance_frequency == "daily":
            mask[:] = True
        elif self.rebalance_frequency == "weekly":
            for i in range(1, len(dates)):
                if dates[i].weekday() < dates[i - 1].weekday():
                    mask[i] = True
        elif self.rebalance_frequency == "monthly":
            for i in range(1, len(dates)):
                if dates[i].month != dates[i - 1].month:
                    mask[i] = True
        elif self.rebalance_frequency == "quarterly":
            for i in range(1, len(dates)):
                if dates[i].quarter != dates[i - 1].quarter:
                    mask[i] = True

        return mask

    def _calculate_turnover(
        self,
        weights_history: np.ndarray,
    ) -> float:
        """Calculate average portfolio turnover."""
        if weights_history is None or len(weights_history) < 2:
            return 0.0

        turnovers = []
        for i in range(1, len(weights_history)):
            turnover = np.sum(np.abs(weights_history[i] - weights_history[i - 1]))
            turnovers.append(turnover)

        return float(np.mean(turnovers))

    def run(
        self,
        returns: np.ndarray,
        weight_function: Callable[[np.ndarray, int], np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None,
        benchmark_returns: Optional[np.ndarray] = None,
        lookback: int = 252,
    ) -> BacktestResult:
        """
        Run backtest with a weight generation function.

        Args:
            returns: (n_samples, n_assets) return matrix
            weight_function: Function that takes (returns[:t], t) and returns weights
            dates: Date index for the returns
            benchmark_returns: Optional benchmark for comparison
            lookback: Minimum lookback before starting

        Returns:
            BacktestResult with all metrics
        """
        n_samples, n_assets = returns.shape

        if dates is None:
            dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="B")

        rebalance_mask = self._get_rebalance_mask(dates)

        # Initialize
        portfolio_returns = np.zeros(n_samples - lookback)
        weights_history = []
        rebalance_dates = []
        current_weights = np.ones(n_assets) / n_assets

        for t in range(lookback, n_samples):
            idx = t - lookback

            # Rebalance if needed
            if rebalance_mask[t]:
                new_weights = weight_function(returns[:t], t)

                # Apply transaction costs
                turnover = np.sum(np.abs(new_weights - current_weights))
                cost = turnover * self.transaction_cost

                current_weights = new_weights
                weights_history.append(current_weights.copy())
                rebalance_dates.append(dates[t])
            else:
                cost = 0

            # Calculate portfolio return
            period_return = np.dot(current_weights, returns[t])
            portfolio_returns[idx] = period_return - cost

            # Update weights for drift
            if t < n_samples - 1:
                asset_values = current_weights * (1 + returns[t])
                current_weights = asset_values / np.sum(asset_values)

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns)

        # Calculate metrics
        metrics = calculate_all_metrics(
            portfolio_returns,
            benchmark_returns[lookback:] if benchmark_returns is not None else None,
            self.risk_free_rate,
        )

        risk_metrics = calculate_portfolio_risk(
            np.ones(1),  # Dummy weights for portfolio returns
            portfolio_returns.reshape(-1, 1),
        )

        dd_info = max_drawdown(portfolio_returns)

        return BacktestResult(
            portfolio_returns=portfolio_returns,
            cumulative_returns=cumulative_returns,
            benchmark_returns=benchmark_returns[lookback:] if benchmark_returns is not None else None,
            weights_history=np.array(weights_history) if weights_history else None,
            rebalance_dates=rebalance_dates,
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            volatility=metrics["volatility"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown=dd_info["max_drawdown"],
            calmar_ratio=metrics["calmar_ratio"],
            var_95=risk_metrics["var_95"],
            cvar_95=risk_metrics["cvar_95"],
            n_trades=len(rebalance_dates),
            turnover=self._calculate_turnover(np.array(weights_history) if weights_history else None),
            start_date=dates[lookback],
            end_date=dates[-1],
            metrics=metrics,
        )

    def run_simple(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run backtest with static weights.

        Args:
            returns: (n_samples, n_assets) return matrix
            weights: Static portfolio weights
            dates: Date index
            benchmark_returns: Optional benchmark

        Returns:
            BacktestResult
        """
        def static_weights(returns_history, t):
            return weights

        return self.run(
            returns,
            static_weights,
            dates,
            benchmark_returns,
            lookback=0,
        )

    def compare_strategies(
        self,
        returns: np.ndarray,
        strategies: dict[str, Callable],
        dates: Optional[pd.DatetimeIndex] = None,
        benchmark_returns: Optional[np.ndarray] = None,
        lookback: int = 252,
    ) -> dict[str, BacktestResult]:
        """
        Compare multiple strategies.

        Args:
            returns: Return matrix
            strategies: Dict mapping strategy names to weight functions
            dates: Date index
            benchmark_returns: Optional benchmark
            lookback: Minimum lookback

        Returns:
            Dict mapping strategy names to BacktestResults
        """
        results = {}

        for name, weight_func in strategies.items():
            results[name] = self.run(
                returns,
                weight_func,
                dates,
                benchmark_returns,
                lookback,
            )

        return results


def print_backtest_report(result: BacktestResult, name: str = "Strategy") -> None:
    """Print formatted backtest report."""
    print(f"\n{'='*60}")
    print(f"  {name} Backtest Report")
    print(f"{'='*60}")

    if result.start_date and result.end_date:
        print(f"  Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")

    print(f"\n  Performance Metrics:")
    print(f"  {'-'*40}")
    print(f"  Total Return:      {result.total_return:>10.2%}")
    print(f"  Annualized Return: {result.annualized_return:>10.2%}")
    print(f"  Volatility:        {result.volatility:>10.2%}")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:     {result.sortino_ratio:>10.2f}")
    print(f"  Calmar Ratio:      {result.calmar_ratio:>10.2f}")

    print(f"\n  Risk Metrics:")
    print(f"  {'-'*40}")
    print(f"  Max Drawdown:      {result.max_drawdown:>10.2%}")
    print(f"  VaR (95%):         {result.var_95:>10.2%}")
    print(f"  CVaR (95%):        {result.cvar_95:>10.2%}")

    print(f"\n  Trading Statistics:")
    print(f"  {'-'*40}")
    print(f"  Number of Trades:  {result.n_trades:>10}")
    print(f"  Avg Turnover:      {result.turnover:>10.2%}")

    print(f"{'='*60}\n")
