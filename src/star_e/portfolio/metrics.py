"""Performance metrics for portfolio evaluation."""

import numpy as np
from scipy import stats


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Sharpe ratio measures risk-adjusted return:
        Sharpe = (E[R] - Rf) / σ(R) * √(periods_per_year)

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sharpe ratio
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    return float(np.mean(excess_returns) / np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    target_return: float = 0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sortino ratio.

    Sortino ratio uses downside deviation instead of total standard deviation,
    making it more appropriate for asymmetric return distributions.

    Sortino = (E[R] - Rf) / σ_down(R) * √(periods_per_year)

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        target_return: Minimum acceptable return
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year

    # Calculate downside deviation
    downside_returns = np.minimum(returns - target_return, 0)
    downside_std = np.sqrt(np.mean(downside_returns**2))

    if downside_std == 0:
        return float("inf") if np.mean(excess_returns) > 0 else 0.0

    return float(np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year))


def information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Information Ratio.

    IR measures the portfolio's ability to generate excess returns
    relative to a benchmark, adjusted for tracking error.

    IR = (Active Return) / (Tracking Error)

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        Annualized Information Ratio
    """
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)

    # Align arrays
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]

    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns, ddof=1)

    if tracking_error == 0:
        return 0.0

    return float(np.mean(active_returns) / tracking_error * np.sqrt(periods_per_year))


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar Ratio.

    Calmar ratio measures the relationship between annualized return
    and maximum drawdown.

    Calmar = Annualized Return / Max Drawdown

    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year

    Returns:
        Calmar Ratio
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = abs(np.min(drawdowns))

    if max_dd == 0:
        return float("inf") if cumulative[-1] > 1 else 0.0

    # Annualized return
    total_return = cumulative[-1] - 1
    n_years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    return float(annualized_return / max_dd)


def omega_ratio(
    returns: np.ndarray,
    threshold: float = 0,
) -> float:
    """
    Calculate Omega Ratio.

    Omega ratio is the probability-weighted ratio of gains to losses
    relative to a threshold return.

    Omega = ∫(threshold to ∞) [1-F(r)] dr / ∫(-∞ to threshold) F(r) dr

    In practice: sum of returns above threshold / sum of returns below threshold

    Args:
        returns: Array of returns
        threshold: Minimum acceptable return

    Returns:
        Omega Ratio
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]

    gains = np.sum(returns[returns > threshold] - threshold)
    losses = np.sum(threshold - returns[returns <= threshold])

    if losses == 0:
        return float("inf") if gains > 0 else 1.0

    return float(gains / losses)


def treynor_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Treynor Ratio.

    Treynor ratio measures excess return per unit of systematic risk (beta).

    Treynor = (E[R] - Rf) / β

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns (market)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Treynor Ratio
    """
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)

    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]

    # Calculate beta
    cov = np.cov(returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns, ddof=1)

    if var == 0:
        return 0.0

    beta = cov / var

    if beta == 0:
        return 0.0

    excess_return = np.mean(returns) - risk_free_rate / periods_per_year

    return float(excess_return / beta * periods_per_year)


def calculate_beta(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """
    Calculate portfolio beta relative to benchmark.

    Beta measures systematic risk (sensitivity to market movements).

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Portfolio beta
    """
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)

    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]

    cov = np.cov(returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns, ddof=1)

    if var == 0:
        return 0.0

    return float(cov / var)


def calculate_alpha(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Jensen's Alpha.

    Alpha measures excess return beyond what CAPM predicts.

    α = R_p - [Rf + β * (R_m - Rf)]

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized alpha
    """
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)

    beta = calculate_beta(returns, benchmark_returns)
    rf_daily = risk_free_rate / periods_per_year

    expected_return = rf_daily + beta * (np.mean(benchmark_returns) - rf_daily)
    alpha = np.mean(returns) - expected_return

    return float(alpha * periods_per_year)


def calculate_all_metrics(
    returns: np.ndarray,
    benchmark_returns: np.ndarray = None,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> dict:
    """
    Calculate all performance metrics.

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns (optional)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Dictionary with all metrics
    """
    returns = np.asarray(returns)
    cumulative = np.cumprod(1 + returns)

    metrics = {
        "total_return": float(cumulative[-1] - 1),
        "annualized_return": float((cumulative[-1] ** (periods_per_year / len(returns))) - 1),
        "volatility": float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, 0, periods_per_year),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "omega_ratio": omega_ratio(returns, 0),
        "skewness": float(stats.skew(returns)),
        "kurtosis": float(stats.kurtosis(returns)),
        "best_day": float(np.max(returns)),
        "worst_day": float(np.min(returns)),
        "positive_days": int(np.sum(returns > 0)),
        "negative_days": int(np.sum(returns < 0)),
        "win_rate": float(np.sum(returns > 0) / len(returns)),
    }

    if benchmark_returns is not None:
        benchmark_returns = np.asarray(benchmark_returns)
        metrics["information_ratio"] = information_ratio(
            returns, benchmark_returns, periods_per_year
        )
        metrics["treynor_ratio"] = treynor_ratio(
            returns, benchmark_returns, risk_free_rate, periods_per_year
        )
        metrics["beta"] = calculate_beta(returns, benchmark_returns)
        metrics["alpha"] = calculate_alpha(
            returns, benchmark_returns, risk_free_rate, periods_per_year
        )

    return metrics
