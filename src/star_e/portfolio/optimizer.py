"""Mean-variance portfolio optimizer with regime awareness."""

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from star_e.config import settings


class PortfolioOptimizer:
    """
    Mean-Variance portfolio optimizer with regime-aware adjustments.

    Constructs portfolios on the efficient frontier with optional
    regime-based risk aversion scaling. In bear markets, risk aversion
    is increased; in bull markets, it's reduced.

    The optimization problem:
        max: E[R] - λ/2 * σ²  (risk-adjusted return)
        s.t.: sum(w) = 1
              w_min <= w_i <= w_max

    Attributes:
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        min_weight: Minimum allocation per asset
        max_weight: Maximum allocation per asset
        regime_risk_scaling: Multiplier for risk aversion by regime
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        min_weight: float = 0.0,
        max_weight: float = 0.3,
        regime_risk_scaling: Optional[dict[int, float]] = None,
    ):
        """
        Initialize portfolio optimizer.

        Args:
            risk_free_rate: Annual risk-free rate
            min_weight: Minimum allocation per asset (0 = no short selling)
            max_weight: Maximum allocation per asset (1 = no constraint)
            regime_risk_scaling: Risk aversion multiplier by regime
        """
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Higher risk aversion in Bear markets, lower in Bull
        self.regime_risk_scaling = regime_risk_scaling or {
            0: 2.0,   # Bear: double risk aversion
            1: 1.0,   # Sideways: normal
            2: 0.7,   # Bull: reduce risk aversion
        }

    def _portfolio_return(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
    ) -> float:
        """Calculate portfolio expected return."""
        return float(np.dot(weights, expected_returns))

    def _portfolio_volatility(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """Calculate portfolio volatility (standard deviation)."""
        return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

    def _portfolio_variance(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """Calculate portfolio variance."""
        return float(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def _sharpe_ratio(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate Sharpe ratio."""
        ret = self._portfolio_return(weights, expected_returns)
        vol = self._portfolio_volatility(weights, cov_matrix)

        if vol == 0:
            return 0

        daily_rf = self.risk_free_rate / periods_per_year
        return (ret - daily_rf) / vol

    def _sortino_ratio(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        returns_history: np.ndarray,
        target_return: float = 0,
    ) -> float:
        """
        Calculate Sortino ratio.

        Uses downside deviation instead of total standard deviation.
        """
        port_return = self._portfolio_return(weights, expected_returns)

        # Calculate downside deviation
        portfolio_returns = returns_history @ weights
        downside_returns = np.minimum(portfolio_returns - target_return, 0)
        downside_std = np.sqrt(np.mean(downside_returns**2))

        if downside_std == 0:
            return float("inf") if port_return > 0 else 0

        daily_rf = self.risk_free_rate / 252
        return (port_return - daily_rf) / downside_std

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_regime: int = 1,
        method: str = "max_sharpe",
        returns_history: Optional[np.ndarray] = None,
        target_return: Optional[float] = None,
    ) -> dict:
        """
        Optimize portfolio weights.

        Args:
            expected_returns: (n_assets,) expected daily returns
            cov_matrix: (n_assets, n_assets) covariance matrix
            current_regime: Current HMM state (0=Bear, 1=Sideways, 2=Bull)
            method: Optimization method:
                - "max_sharpe": Maximize Sharpe ratio
                - "min_variance": Minimize variance
                - "max_sortino": Maximize Sortino ratio
                - "target_return": Minimize variance for target return
            returns_history: Historical returns for Sortino calculation
            target_return: Target return for "target_return" method

        Returns:
            Dictionary with:
                - weights: Optimal weights
                - expected_return: Portfolio expected return
                - volatility: Portfolio volatility
                - sharpe: Sharpe ratio
                - regime: Current regime
                - success: Optimization success flag
        """
        n_assets = len(expected_returns)

        # Initial guess: equal weights
        init_weights = np.array([1 / n_assets] * n_assets)

        # Constraints: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # Bounds: min/max per asset
        bounds = tuple(
            (self.min_weight, self.max_weight) for _ in range(n_assets)
        )

        # Regime-adjusted risk aversion
        risk_scale = self.regime_risk_scaling.get(current_regime, 1.0)

        if method == "max_sharpe":
            def neg_sharpe(w):
                ret = self._portfolio_return(w, expected_returns)
                vol = self._portfolio_volatility(w, cov_matrix) * np.sqrt(risk_scale)
                daily_rf = self.risk_free_rate / 252
                return -(ret - daily_rf) / vol if vol > 0 else 0

            objective = neg_sharpe

        elif method == "min_variance":
            def variance(w):
                return self._portfolio_variance(w, cov_matrix) * risk_scale

            objective = variance

        elif method == "max_sortino":
            if returns_history is None:
                raise ValueError("returns_history required for Sortino optimization")

            def neg_sortino(w):
                return -self._sortino_ratio(w, expected_returns, returns_history)

            objective = neg_sortino

        elif method == "target_return":
            if target_return is None:
                raise ValueError("target_return required for target_return method")

            def variance(w):
                return self._portfolio_variance(w, cov_matrix)

            constraints.append({
                "type": "eq",
                "fun": lambda w: self._portfolio_return(w, expected_returns) - target_return
            })
            objective = variance

        else:
            raise ValueError(f"Unknown method: {method}")

        result = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        optimal_weights = result.x
        port_return = self._portfolio_return(optimal_weights, expected_returns)
        port_vol = self._portfolio_volatility(optimal_weights, cov_matrix)
        sharpe = self._sharpe_ratio(optimal_weights, expected_returns, cov_matrix)

        return {
            "weights": optimal_weights,
            "expected_return": port_return,
            "volatility": port_vol,
            "sharpe": sharpe,
            "regime": current_regime,
            "method": method,
            "success": result.success,
        }

    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_points: int = 50,
    ) -> dict:
        """
        Generate the efficient frontier.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            n_points: Number of points on the frontier

        Returns:
            Dictionary with 'returns', 'volatilities', 'weights' arrays
        """
        # Find min and max return portfolios
        n_assets = len(expected_returns)

        # Min variance portfolio
        min_var = self.optimize(
            expected_returns, cov_matrix, method="min_variance"
        )
        min_return = min_var["expected_return"]

        # Max return portfolio (100% in highest return asset, subject to constraints)
        max_idx = np.argmax(expected_returns)
        max_return = min(expected_returns[max_idx], self.max_weight * expected_returns.max())

        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)

        frontier_returns = []
        frontier_vols = []
        frontier_weights = []

        for target in target_returns:
            try:
                result = self.optimize(
                    expected_returns,
                    cov_matrix,
                    method="target_return",
                    target_return=target,
                )
                if result["success"]:
                    frontier_returns.append(result["expected_return"])
                    frontier_vols.append(result["volatility"])
                    frontier_weights.append(result["weights"])
            except Exception:
                continue

        return {
            "returns": np.array(frontier_returns),
            "volatilities": np.array(frontier_vols),
            "weights": np.array(frontier_weights),
        }

    def regime_aware_allocation(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        regime_probabilities: np.ndarray,
    ) -> dict:
        """
        Calculate allocation considering regime uncertainty.

        Weights the optimal portfolio for each regime by its probability.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            regime_probabilities: (3,) array of regime probabilities

        Returns:
            Blended portfolio allocation
        """
        if len(regime_probabilities) != 3:
            raise ValueError("regime_probabilities must have 3 elements")

        blended_weights = np.zeros(len(expected_returns))

        for regime in range(3):
            result = self.optimize(
                expected_returns,
                cov_matrix,
                current_regime=regime,
                method="max_sharpe",
            )
            blended_weights += regime_probabilities[regime] * result["weights"]

        # Renormalize
        blended_weights = blended_weights / blended_weights.sum()

        port_return = self._portfolio_return(blended_weights, expected_returns)
        port_vol = self._portfolio_volatility(blended_weights, cov_matrix)

        return {
            "weights": blended_weights,
            "expected_return": port_return,
            "volatility": port_vol,
            "sharpe": self._sharpe_ratio(blended_weights, expected_returns, cov_matrix),
            "regime_probabilities": regime_probabilities.tolist(),
        }
