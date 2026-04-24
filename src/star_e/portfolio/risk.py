"""
Risk metrics: VaR, CVaR, Monte Carlo simulation, drawdown analysis.

Implements multiple VaR methodologies including Monte Carlo simulation
with correlated asset returns.
"""

from typing import Literal, Optional, Dict, Tuple

import numpy as np
from scipy import stats
from scipy.linalg import cholesky
import pandas as pd


def calculate_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    method: Literal["historical", "parametric", "cornish_fisher"] = "historical",
) -> float:
    """
    Calculate Value at Risk (VaR).

    VaR represents the maximum expected loss over a given time period
    at a specified confidence level.

    Args:
        returns: Array of returns (as decimals, e.g., 0.02 for 2%)
        confidence: Confidence level (0.95 = 95%)
        method:
            - "historical": Non-parametric, uses empirical quantile
            - "parametric": Assumes normal distribution
            - "cornish_fisher": Adjusts for skewness and kurtosis

    Returns:
        VaR as a positive number (potential loss)
    """
    alpha = 1 - confidence
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        raise ValueError("No valid returns provided")

    if method == "historical":
        var = np.percentile(returns, alpha * 100)

    elif method == "parametric":
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        var = mu + sigma * stats.norm.ppf(alpha)

    elif method == "cornish_fisher":
        # Cornish-Fisher expansion for non-normal distributions
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis

        z = stats.norm.ppf(alpha)
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36
        )

        var = mu + sigma * z_cf

    else:
        raise ValueError(f"Unknown method: {method}")

    return -var  # Return positive VaR


def calculate_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR.
    It is a coherent risk measure (unlike VaR) and better captures
    tail risk.

    CVaR = E[Loss | Loss > VaR]

    Args:
        returns: Array of returns
        confidence: Confidence level

    Returns:
        CVaR as a positive number
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        raise ValueError("No valid returns provided")

    var = calculate_var(returns, confidence, method="historical")
    tail_returns = returns[returns <= -var]

    if len(tail_returns) == 0:
        return var

    return -np.mean(tail_returns)


def max_drawdown(
    returns: np.ndarray,
    return_details: bool = False,
) -> dict:
    """
    Calculate maximum drawdown and related statistics.

    Maximum drawdown is the largest peak-to-trough decline in
    portfolio value.

    Args:
        returns: Array of returns (not cumulative)
        return_details: Whether to return detailed drawdown info

    Returns:
        Dictionary with:
            - max_drawdown: Maximum drawdown (positive number)
            - peak_idx: Index of peak before drawdown
            - trough_idx: Index of trough
            - recovery_idx: Index of recovery (if recovered)
            - duration: Drawdown duration in periods
            - recovery_duration: Time to recover
    """
    returns = np.asarray(returns)
    cumulative = np.cumprod(1 + returns)

    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max

    max_dd = np.min(drawdowns)
    trough_idx = int(np.argmin(drawdowns))

    if trough_idx == 0:
        peak_idx = 0
    else:
        peak_idx = int(np.argmax(cumulative[:trough_idx + 1]))

    # Find recovery (if any)
    recovery_idx = trough_idx
    peak_value = cumulative[peak_idx]

    for i in range(trough_idx, len(cumulative)):
        if cumulative[i] >= peak_value:
            recovery_idx = i
            break

    result = {
        "max_drawdown": abs(max_dd),
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "recovery_idx": recovery_idx,
        "duration": trough_idx - peak_idx,
        "recovery_duration": recovery_idx - trough_idx,
        "recovered": recovery_idx > trough_idx,
    }

    if return_details:
        result["drawdown_series"] = drawdowns
        result["cumulative_returns"] = cumulative

    return result


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """
    Calculate the drawdown at each point in time.

    Args:
        returns: Array of returns

    Returns:
        Array of drawdown values (negative or zero)
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    return (cumulative - running_max) / running_max


def ulcer_index(returns: np.ndarray) -> float:
    """
    Calculate the Ulcer Index.

    The Ulcer Index measures downside risk by considering both
    the depth and duration of drawdowns. Unlike standard deviation,
    it only penalizes negative returns.

    UI = sqrt(mean(drawdown^2))

    Args:
        returns: Array of returns

    Returns:
        Ulcer Index value
    """
    dd = drawdown_series(returns)
    return np.sqrt(np.mean(dd**2))


def pain_index(returns: np.ndarray) -> float:
    """
    Calculate the Pain Index.

    Similar to Ulcer Index but uses mean absolute drawdown
    instead of RMS drawdown.

    Args:
        returns: Array of returns

    Returns:
        Pain Index value
    """
    dd = drawdown_series(returns)
    return np.mean(np.abs(dd))


def calculate_portfolio_risk(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """
    Calculate comprehensive risk metrics for a portfolio.

    Args:
        weights: Portfolio weights
        returns: (n_samples, n_assets) return matrix
        confidence: Confidence level for VaR/CVaR

    Returns:
        Dictionary with all risk metrics
    """
    portfolio_returns = returns @ weights

    return {
        "volatility": float(np.std(portfolio_returns, ddof=1)),
        "annualized_volatility": float(np.std(portfolio_returns, ddof=1) * np.sqrt(252)),
        "var_95": calculate_var(portfolio_returns, 0.95),
        "var_99": calculate_var(portfolio_returns, 0.99),
        "cvar_95": calculate_cvar(portfolio_returns, 0.95),
        "cvar_99": calculate_cvar(portfolio_returns, 0.99),
        "max_drawdown": max_drawdown(portfolio_returns)["max_drawdown"],
        "ulcer_index": ulcer_index(portfolio_returns),
        "skewness": float(stats.skew(portfolio_returns)),
        "kurtosis": float(stats.kurtosis(portfolio_returns)),
    }


def monte_carlo_var(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.95,
    n_simulations: int = 10000,
    horizon: int = 1,
    distribution: Literal["normal", "t", "historical"] = "normal",
    random_state: Optional[int] = 42,
) -> Dict:
    """
    Calculate VaR using Monte Carlo simulation.

    Simulates future portfolio returns by sampling from a fitted
    distribution while preserving the correlation structure.

    Args:
        weights: Portfolio weights (n_assets,)
        returns: Historical returns (n_samples, n_assets)
        confidence: Confidence level (0.95 = 95%)
        n_simulations: Number of simulation paths
        horizon: Forecast horizon in days
        distribution: Distribution to sample from
            "normal": Multivariate normal
            "t": Multivariate Student's t
            "historical": Bootstrap from historical returns
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with VaR, CVaR, and simulation details
    """
    np.random.seed(random_state)
    n_assets = len(weights)

    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns.T)

    if distribution == "normal":
        simulated_returns = np.random.multivariate_normal(
            mean_returns * horizon,
            cov_matrix * horizon,
            size=n_simulations,
        )

    elif distribution == "t":
        df = 5
        L = cholesky(cov_matrix * horizon, lower=True)
        z = stats.t.rvs(df, size=(n_simulations, n_assets))
        simulated_returns = mean_returns * horizon + z @ L.T

    elif distribution == "historical":
        n_samples = len(returns)
        indices = np.random.randint(0, n_samples, size=(n_simulations, horizon))

        simulated_returns = np.zeros((n_simulations, n_assets))
        for i in range(n_simulations):
            path_returns = returns[indices[i]]
            simulated_returns[i] = np.sum(path_returns, axis=0)

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    portfolio_returns = simulated_returns @ weights

    var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
    cvar = -np.mean(portfolio_returns[portfolio_returns <= -var])

    return {
        "var": float(var),
        "cvar": float(cvar),
        "mean_return": float(np.mean(portfolio_returns)),
        "std_return": float(np.std(portfolio_returns)),
        "min_return": float(np.min(portfolio_returns)),
        "max_return": float(np.max(portfolio_returns)),
        "n_simulations": n_simulations,
        "horizon": horizon,
        "distribution": distribution,
        "simulated_returns": portfolio_returns,
    }


def monte_carlo_paths(
    initial_value: float,
    weights: np.ndarray,
    returns: np.ndarray,
    n_simulations: int = 1000,
    horizon: int = 252,
    distribution: Literal["normal", "gbm"] = "gbm",
    random_state: Optional[int] = 42,
) -> Dict:
    """
    Simulate portfolio value paths using Monte Carlo.

    Args:
        initial_value: Initial portfolio value
        weights: Portfolio weights
        returns: Historical returns
        n_simulations: Number of paths
        horizon: Number of time steps
        distribution: "normal" or "gbm" (Geometric Brownian Motion)
        random_state: Random seed

    Returns:
        Dictionary with paths and statistics
    """
    np.random.seed(random_state)

    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns.T)

    paths = np.zeros((n_simulations, horizon + 1))
    paths[:, 0] = initial_value

    if distribution == "normal":
        for t in range(1, horizon + 1):
            daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_simulations)
            portfolio_returns = daily_returns @ weights
            paths[:, t] = paths[:, t-1] * (1 + portfolio_returns)

    elif distribution == "gbm":
        portfolio_mu = mean_returns @ weights
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_sigma = np.sqrt(portfolio_var)

        drift = (portfolio_mu - 0.5 * portfolio_var)
        diffusion = portfolio_sigma

        for t in range(1, horizon + 1):
            z = np.random.standard_normal(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * z)

    final_values = paths[:, -1]

    return {
        "paths": paths,
        "final_values": final_values,
        "mean_final": float(np.mean(final_values)),
        "median_final": float(np.median(final_values)),
        "std_final": float(np.std(final_values)),
        "percentile_5": float(np.percentile(final_values, 5)),
        "percentile_95": float(np.percentile(final_values, 95)),
        "prob_loss": float(np.mean(final_values < initial_value)),
    }


def stress_test_var(
    weights: np.ndarray,
    returns: np.ndarray,
    stress_scenarios: Dict[str, Dict],
    confidence: float = 0.95,
    n_simulations: int = 10000,
) -> Dict:
    """
    Perform stress testing on VaR estimates.

    Args:
        weights: Portfolio weights
        returns: Historical returns
        stress_scenarios: Dictionary of scenarios, each with:
            - "vol_multiplier": Volatility shock multiplier
            - "correlation_shock": Correlation adjustment
            - "mean_shift": Mean return shift
        confidence: Confidence level
        n_simulations: Number of simulations per scenario

    Returns:
        VaR under each stress scenario
    """
    base_var = monte_carlo_var(weights, returns, confidence, n_simulations)

    results = {
        "base": base_var["var"],
    }

    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns.T)

    for name, scenario in stress_scenarios.items():
        stressed_mean = mean_returns + scenario.get("mean_shift", 0)

        vol_mult = scenario.get("vol_multiplier", 1.0)
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

        stressed_std = std_devs * vol_mult

        corr_shock = scenario.get("correlation_shock", 0)
        stressed_corr = corr_matrix * (1 - corr_shock) + corr_shock * np.ones_like(corr_matrix)
        np.fill_diagonal(stressed_corr, 1.0)

        stressed_cov = np.outer(stressed_std, stressed_std) * stressed_corr

        try:
            stressed_returns = np.random.multivariate_normal(
                stressed_mean, stressed_cov, n_simulations
            )
            portfolio_returns = stressed_returns @ weights
            results[name] = float(-np.percentile(portfolio_returns, (1 - confidence) * 100))
        except Exception:
            results[name] = None

    return results


def incremental_var(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.95,
    n_simulations: int = 10000,
) -> np.ndarray:
    """
    Calculate incremental VaR contribution for each asset.

    Measures how VaR changes when each asset's weight increases marginally.

    Args:
        weights: Portfolio weights
        returns: Historical returns
        confidence: Confidence level
        n_simulations: Number of simulations

    Returns:
        Array of incremental VaR for each asset
    """
    base_var = monte_carlo_var(weights, returns, confidence, n_simulations)["var"]

    n_assets = len(weights)
    incr_var = np.zeros(n_assets)

    delta = 0.01

    for i in range(n_assets):
        shocked_weights = weights.copy()
        shocked_weights[i] += delta
        shocked_weights /= shocked_weights.sum()

        shocked_var = monte_carlo_var(shocked_weights, returns, confidence, n_simulations)["var"]

        incr_var[i] = (shocked_var - base_var) / delta

    return incr_var


def component_var(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.95,
) -> Dict:
    """
    Calculate component VaR (marginal VaR * weight).

    Decomposes total VaR into contributions from each asset.
    Sum of component VaRs equals total VaR.

    Args:
        weights: Portfolio weights
        returns: Historical returns
        confidence: Confidence level

    Returns:
        Dictionary with component VaR details
    """
    portfolio_returns = returns @ weights
    portfolio_var = calculate_var(portfolio_returns, confidence)
    portfolio_vol = np.std(portfolio_returns)

    cov_matrix = np.cov(returns.T)
    marginal_var = (cov_matrix @ weights) / portfolio_vol * stats.norm.ppf(confidence)

    component_var = weights * marginal_var

    return {
        "total_var": float(portfolio_var),
        "component_var": component_var,
        "pct_contribution": component_var / portfolio_var * 100,
        "marginal_var": marginal_var,
    }


def expected_shortfall_decomposition(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.95,
    n_simulations: int = 10000,
) -> Dict:
    """
    Decompose Expected Shortfall (CVaR) by asset.

    Args:
        weights: Portfolio weights
        returns: Historical returns
        confidence: Confidence level
        n_simulations: Number of simulations

    Returns:
        ES contributions by asset
    """
    mc_result = monte_carlo_var(weights, returns, confidence, n_simulations)
    sim_returns = mc_result["simulated_returns"]
    var_threshold = -mc_result["var"]

    tail_mask = sim_returns <= var_threshold
    n_tail = tail_mask.sum()

    if n_tail == 0:
        return {"total_es": mc_result["cvar"], "contributions": np.zeros(len(weights))}

    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns.T)

    contributions = np.zeros(len(weights))
    for i in range(len(weights)):
        contributions[i] = weights[i] * mean_returns[i]

    total_es = mc_result["cvar"]
    contributions = contributions / contributions.sum() * total_es

    return {
        "total_es": float(total_es),
        "contributions": contributions,
        "pct_contributions": contributions / total_es * 100,
    }
