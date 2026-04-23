"""Risk metrics: VaR, CVaR, drawdown analysis."""

from typing import Literal

import numpy as np
from scipy import stats


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
