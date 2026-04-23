"""Cointegration analysis for statistical arbitrage."""

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen_test(
    prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """
    Perform Johansen cointegration test.

    The Johansen test identifies cointegrating relationships between multiple
    time series. Unlike the Engle-Granger test, it can find multiple
    cointegrating vectors in an N-dimensional system.

    Args:
        prices: DataFrame with price series as columns
        det_order: Deterministic trend order:
            -1 = no constant, no trend
             0 = constant only (most common)
             1 = constant and trend
        k_ar_diff: Number of lagged differences in the VAR model

    Returns:
        Dictionary with:
            - trace_stat: Trace test statistics
            - trace_crit_95: 95% critical values for trace test
            - max_eigen_stat: Max eigenvalue statistics
            - max_eigen_crit_95: 95% critical values for max eigenvalue
            - eigenvectors: Cointegrating vectors
            - eigenvalues: Eigenvalues
            - n_cointegrating: Number of cointegrating relationships
    """
    if prices.shape[1] < 2:
        raise ValueError("Need at least 2 price series for cointegration test")

    result = coint_johansen(
        prices.values,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
    )

    # Count cointegrating relationships (trace test)
    n_coint = sum(result.lr1 > result.cvt[:, 1])

    return {
        "trace_stat": result.lr1.tolist(),
        "trace_crit_95": result.cvt[:, 1].tolist(),
        "max_eigen_stat": result.lr2.tolist(),
        "max_eigen_crit_95": result.cvm[:, 1].tolist(),
        "eigenvectors": result.evec.tolist(),
        "eigenvalues": result.eig.tolist(),
        "n_cointegrating": n_coint,
        "is_cointegrated": n_coint > 0,
    }


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    significance: float = 0.05,
    min_observations: int = 100,
) -> list[tuple[str, str, dict]]:
    """
    Find all cointegrated pairs in a universe of assets.

    Tests all pairwise combinations for cointegration using the
    Johansen test. Returns pairs where the trace test rejects
    the null hypothesis of no cointegration.

    Args:
        prices: DataFrame with price series as columns
        significance: Significance level for test (0.05 = 95% confidence)
        min_observations: Minimum observations required for test

    Returns:
        List of (ticker1, ticker2, test_results) tuples for cointegrated pairs
    """
    tickers = prices.columns.tolist()
    cointegrated = []

    for t1, t2 in combinations(tickers, 2):
        pair_prices = prices[[t1, t2]].dropna()

        if len(pair_prices) < min_observations:
            continue

        try:
            result = johansen_test(pair_prices)

            # Check if at least one cointegrating relationship exists
            if result["is_cointegrated"]:
                # Add hedge ratio (first eigenvector, normalized)
                evec = np.array(result["eigenvectors"])[:, 0]
                hedge_ratio = -evec[1] / evec[0] if evec[0] != 0 else np.nan

                result["hedge_ratio"] = float(hedge_ratio)
                result["pair"] = (t1, t2)

                cointegrated.append((t1, t2, result))

        except Exception as e:
            print(f"Warning: Could not test {t1}-{t2}: {e}")
            continue

    # Sort by trace statistic strength
    cointegrated.sort(
        key=lambda x: x[2]["trace_stat"][0] - x[2]["trace_crit_95"][0],
        reverse=True,
    )

    return cointegrated


def calculate_spread(
    prices: pd.DataFrame,
    hedge_ratio: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Calculate cointegration spread (error correction term).

    The spread represents the deviation from the long-term equilibrium
    relationship. It should be stationary and mean-reverting.

    Args:
        prices: Price DataFrame with exactly 2 columns
        hedge_ratio: Cointegrating vector. If None, calculated from Johansen test.

    Returns:
        Spread series
    """
    if prices.shape[1] != 2:
        raise ValueError("Spread calculation requires exactly 2 price series")

    if hedge_ratio is None:
        result = johansen_test(prices)
        hedge_ratio = np.array(result["eigenvectors"])[:, 0]

    spread = (prices.values @ hedge_ratio).flatten()

    return pd.Series(spread, index=prices.index, name="spread")


def zscore_spread(spread: pd.Series, lookback: int = 21) -> pd.Series:
    """
    Calculate z-score of spread for trading signals.

    Args:
        spread: Spread series
        lookback: Rolling window for z-score calculation

    Returns:
        Z-score series
    """
    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std()

    zscore = (spread - rolling_mean) / rolling_std

    return zscore


def half_life(spread: pd.Series) -> float:
    """
    Estimate half-life of mean reversion for the spread.

    Uses OLS regression of spread changes on lagged spread:
        Δspread_t = α + β * spread_{t-1} + ε

    Half-life = -ln(2) / ln(1 + β)

    Args:
        spread: Spread series

    Returns:
        Half-life in periods
    """
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag

    # Remove NaN
    valid = ~(spread_lag.isna() | spread_diff.isna())
    y = spread_diff[valid].values
    X = spread_lag[valid].values

    # OLS regression
    beta = np.cov(X, y)[0, 1] / np.var(X)

    if beta >= 0:
        return float("inf")  # Not mean-reverting

    half_life = -np.log(2) / np.log(1 + beta)

    return max(half_life, 0)


def generate_pairs_signals(
    prices: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    lookback: int = 21,
) -> pd.DataFrame:
    """
    Generate trading signals for a cointegrated pair.

    Long spread when z-score < -entry_z, short when z-score > entry_z.
    Exit when |z-score| < exit_z.

    Args:
        prices: Price DataFrame with 2 columns
        entry_z: Z-score threshold for entry
        exit_z: Z-score threshold for exit
        lookback: Rolling window for z-score

    Returns:
        DataFrame with spread, zscore, and position columns
    """
    spread = calculate_spread(prices)
    z = zscore_spread(spread, lookback)

    signals = pd.DataFrame(index=prices.index)
    signals["spread"] = spread
    signals["zscore"] = z

    # Generate positions: 1 = long spread, -1 = short spread, 0 = flat
    position = np.zeros(len(z))
    current_pos = 0

    for i in range(len(z)):
        if np.isnan(z.iloc[i]):
            position[i] = 0
            continue

        if current_pos == 0:
            # Entry conditions
            if z.iloc[i] < -entry_z:
                current_pos = 1  # Long spread
            elif z.iloc[i] > entry_z:
                current_pos = -1  # Short spread
        else:
            # Exit conditions
            if abs(z.iloc[i]) < exit_z:
                current_pos = 0
            # Stop-loss: exit if z-score moves against position
            elif current_pos == 1 and z.iloc[i] > entry_z:
                current_pos = 0
            elif current_pos == -1 and z.iloc[i] < -entry_z:
                current_pos = 0

        position[i] = current_pos

    signals["position"] = position

    return signals
