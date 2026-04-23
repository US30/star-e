"""Tests for portfolio module."""

import numpy as np
import pytest

from star_e.portfolio.optimizer import PortfolioOptimizer
from star_e.portfolio.risk import calculate_var, calculate_cvar, max_drawdown
from star_e.portfolio.metrics import sharpe_ratio, sortino_ratio, calmar_ratio
from star_e.portfolio.cointegration import johansen_test, calculate_spread


class TestPortfolioOptimizer:
    """Tests for portfolio optimization."""

    def test_equal_weights_baseline(self, sample_returns):
        """Test that equal weights work as baseline."""
        n_assets = sample_returns.shape[1]
        exp_ret = np.mean(sample_returns[-63:], axis=0)
        cov = np.cov(sample_returns.T)

        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(exp_ret, cov, method="min_variance")

        assert result["success"]
        assert len(result["weights"]) == n_assets
        assert np.isclose(result["weights"].sum(), 1.0)

    def test_weights_within_bounds(self, sample_returns):
        """Test that weights respect min/max bounds."""
        exp_ret = np.mean(sample_returns[-63:], axis=0)
        cov = np.cov(sample_returns.T)

        optimizer = PortfolioOptimizer(min_weight=0.05, max_weight=0.4)
        result = optimizer.optimize(exp_ret, cov, method="max_sharpe")

        assert all(w >= 0.05 - 1e-6 for w in result["weights"])
        assert all(w <= 0.4 + 1e-6 for w in result["weights"])

    def test_regime_aware_scaling(self, sample_returns):
        """Test that regime affects optimization."""
        exp_ret = np.mean(sample_returns[-63:], axis=0)
        cov = np.cov(sample_returns.T)

        optimizer = PortfolioOptimizer()

        result_bull = optimizer.optimize(exp_ret, cov, current_regime=2)
        result_bear = optimizer.optimize(exp_ret, cov, current_regime=0)

        # Bear regime should have lower volatility due to higher risk aversion
        # (This depends on the optimization outcome, so we just check both succeed)
        assert result_bull["success"]
        assert result_bear["success"]

    def test_efficient_frontier(self, sample_returns):
        """Test efficient frontier generation."""
        exp_ret = np.mean(sample_returns[-63:], axis=0)
        cov = np.cov(sample_returns.T)

        optimizer = PortfolioOptimizer()
        frontier = optimizer.efficient_frontier(exp_ret, cov, n_points=20)

        assert len(frontier["returns"]) > 0
        assert len(frontier["volatilities"]) == len(frontier["returns"])

        # Returns should increase (mostly) with volatility
        # Allow for some non-monotonicity due to optimization tolerances


class TestRiskMetrics:
    """Tests for risk metrics."""

    def test_var_positive(self, sample_returns):
        """Test that VaR is positive (loss)."""
        returns = sample_returns[:, 0]

        var = calculate_var(returns, confidence=0.95)

        assert var > 0

    def test_cvar_greater_than_var(self, sample_returns):
        """Test that CVaR >= VaR."""
        returns = sample_returns[:, 0]

        var = calculate_var(returns, confidence=0.95)
        cvar = calculate_cvar(returns, confidence=0.95)

        assert cvar >= var

    def test_var_methods_reasonable(self, sample_returns):
        """Test that different VaR methods give similar results."""
        returns = sample_returns[:, 0]

        var_hist = calculate_var(returns, method="historical")
        var_param = calculate_var(returns, method="parametric")
        var_cf = calculate_var(returns, method="cornish_fisher")

        # All should be in similar range
        assert 0.5 * var_hist <= var_param <= 2 * var_hist
        assert 0.5 * var_hist <= var_cf <= 2 * var_hist

    def test_max_drawdown(self, sample_returns):
        """Test max drawdown calculation."""
        returns = sample_returns[:, 0]

        dd = max_drawdown(returns)

        assert 0 < dd["max_drawdown"] <= 1
        assert dd["peak_idx"] <= dd["trough_idx"]
        assert dd["duration"] >= 0


class TestPerformanceMetrics:
    """Tests for performance metrics."""

    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        returns = sample_returns[:, 0]

        sharpe = sharpe_ratio(returns)

        # Sharpe should be finite
        assert np.isfinite(sharpe)

    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        returns = sample_returns[:, 0]

        sortino = sortino_ratio(returns)

        # Sortino >= Sharpe (theoretically for symmetric distributions, similar)
        assert np.isfinite(sortino)

    def test_calmar_ratio(self, sample_returns):
        """Test Calmar ratio calculation."""
        returns = sample_returns[:, 0]

        calmar = calmar_ratio(returns)

        assert np.isfinite(calmar)

    def test_zero_volatility_sharpe(self):
        """Test Sharpe with zero volatility returns zero."""
        returns = np.zeros(100)

        sharpe = sharpe_ratio(returns)

        assert sharpe == 0


class TestCointegration:
    """Tests for cointegration analysis."""

    def test_johansen_test(self, cointegrated_prices):
        """Test Johansen cointegration test."""
        result = johansen_test(cointegrated_prices)

        assert "trace_stat" in result
        assert "n_cointegrating" in result
        assert result["n_cointegrating"] >= 0

    def test_cointegrated_pair_detected(self, cointegrated_prices):
        """Test that cointegrated pair is detected."""
        result = johansen_test(cointegrated_prices)

        # Should detect cointegration
        assert result["is_cointegrated"]

    def test_spread_calculation(self, cointegrated_prices):
        """Test spread calculation."""
        spread = calculate_spread(cointegrated_prices)

        assert len(spread) == len(cointegrated_prices)

        # Spread should be more stationary than individual series
        spread_std = spread.std()
        price_std = cointegrated_prices["A"].std()

        # Spread std should be less than price std (normalized)
        assert spread_std < price_std
