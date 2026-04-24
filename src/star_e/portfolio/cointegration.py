"""
Cointegration and Granger Causality analysis for statistical arbitrage.

Implements both Johansen and Engle-Granger cointegration tests, plus
Granger Causality for lead-lag relationship detection.
"""

from itertools import combinations
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint, adfuller, grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings


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


def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
    trend: str = "c",
    autolag: str = "aic",
) -> Dict:
    """
    Perform Engle-Granger two-step cointegration test.

    Step 1: Regress y on x to get residuals
    Step 2: Test residuals for stationarity using ADF

    Args:
        y: Dependent series (prices)
        x: Independent series (prices)
        trend: Trend specification for ADF test
            "c" = constant only
            "ct" = constant and trend
            "nc" = no constant
        autolag: Method for lag selection in ADF

    Returns:
        Dictionary with test results
    """
    y_arr = y.values if isinstance(y, pd.Series) else y
    x_arr = x.values if isinstance(x, pd.Series) else x

    x_const = add_constant(x_arr)
    model = OLS(y_arr, x_const).fit()
    residuals = model.resid

    adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(
        residuals, regression=trend, autolag=autolag
    )

    coint_stat, coint_pvalue, _ = coint(y_arr, x_arr, trend=trend, autolag=autolag)

    beta = model.params[1]
    alpha = model.params[0]

    return {
        "cointegration_stat": float(coint_stat),
        "cointegration_pvalue": float(coint_pvalue),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "critical_values": {k: float(v) for k, v in critical_values.items()},
        "is_cointegrated": coint_pvalue < 0.05,
        "hedge_ratio": float(beta),
        "intercept": float(alpha),
        "residuals": residuals,
        "r_squared": model.rsquared,
    }


def find_cointegrated_pairs_eg(
    prices: pd.DataFrame,
    significance: float = 0.05,
    min_observations: int = 100,
) -> List[Tuple[str, str, Dict]]:
    """
    Find cointegrated pairs using Engle-Granger test.

    Tests all pairwise combinations in both directions since
    Engle-Granger is not symmetric.

    Args:
        prices: DataFrame with price series as columns
        significance: Significance level
        min_observations: Minimum required observations

    Returns:
        List of (ticker_y, ticker_x, results) tuples
    """
    tickers = prices.columns.tolist()
    cointegrated = []

    for t1, t2 in combinations(tickers, 2):
        pair_prices = prices[[t1, t2]].dropna()

        if len(pair_prices) < min_observations:
            continue

        try:
            result1 = engle_granger_test(pair_prices[t1], pair_prices[t2])
            if result1["cointegration_pvalue"] < significance:
                result1["pair"] = (t1, t2)
                result1["direction"] = f"{t1} ~ {t2}"
                cointegrated.append((t1, t2, result1))

            result2 = engle_granger_test(pair_prices[t2], pair_prices[t1])
            if result2["cointegration_pvalue"] < significance:
                result2["pair"] = (t2, t1)
                result2["direction"] = f"{t2} ~ {t1}"
                cointegrated.append((t2, t1, result2))

        except Exception as e:
            continue

    cointegrated.sort(key=lambda x: x[2]["cointegration_pvalue"])

    return cointegrated


def granger_causality_test(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 10,
    significance: float = 0.05,
) -> Dict:
    """
    Test for Granger causality between two series.

    Tests whether x Granger-causes y (x contains information
    useful for predicting y beyond y's own history).

    Args:
        x: Potential cause series
        y: Potential effect series
        max_lag: Maximum lag to test
        significance: Significance level

    Returns:
        Dictionary with test results for each lag
    """
    data = pd.DataFrame({"y": y, "x": x}).dropna()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = grangercausalitytests(data[["y", "x"]], maxlag=max_lag, verbose=False)

    lag_results = {}
    significant_lags = []

    for lag in range(1, max_lag + 1):
        test_result = results[lag][0]
        f_stat = test_result["ssr_ftest"][0]
        f_pvalue = test_result["ssr_ftest"][1]

        lag_results[lag] = {
            "f_statistic": float(f_stat),
            "p_value": float(f_pvalue),
            "is_significant": f_pvalue < significance,
        }

        if f_pvalue < significance:
            significant_lags.append(lag)

    optimal_lag = significant_lags[0] if significant_lags else None

    return {
        "lag_results": lag_results,
        "significant_lags": significant_lags,
        "optimal_lag": optimal_lag,
        "granger_causes": len(significant_lags) > 0,
    }


def bidirectional_granger(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 10,
) -> Dict:
    """
    Test bidirectional Granger causality.

    Determines:
    - x → y (x Granger-causes y)
    - y → x (y Granger-causes x)
    - Bidirectional (feedback)
    - No causality

    Args:
        x: First series
        y: Second series
        max_lag: Maximum lag

    Returns:
        Dictionary with causality direction
    """
    x_causes_y = granger_causality_test(x, y, max_lag)
    y_causes_x = granger_causality_test(y, x, max_lag)

    x_to_y = x_causes_y["granger_causes"]
    y_to_x = y_causes_x["granger_causes"]

    if x_to_y and y_to_x:
        direction = "bidirectional"
    elif x_to_y:
        direction = "x_causes_y"
    elif y_to_x:
        direction = "y_causes_x"
    else:
        direction = "no_causality"

    return {
        "direction": direction,
        "x_causes_y": x_causes_y,
        "y_causes_x": y_causes_x,
    }


def build_causality_network(
    returns: pd.DataFrame,
    max_lag: int = 5,
    significance: float = 0.05,
) -> Dict:
    """
    Build a Granger causality network for all assets.

    Creates a directed graph where edges represent
    significant Granger causality relationships.

    Args:
        returns: DataFrame with asset returns
        max_lag: Maximum lag for causality tests
        significance: Significance threshold

    Returns:
        Dictionary with adjacency matrix and edge list
    """
    tickers = returns.columns.tolist()
    n = len(tickers)

    adj_matrix = np.zeros((n, n))
    edges = []

    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i == j:
                continue

            result = granger_causality_test(
                returns[t1], returns[t2], max_lag, significance
            )

            if result["granger_causes"]:
                adj_matrix[i, j] = 1
                optimal_lag = result["optimal_lag"]
                edges.append({
                    "source": t1,
                    "target": t2,
                    "lag": optimal_lag,
                    "p_value": result["lag_results"][optimal_lag]["p_value"],
                })

    return {
        "adjacency_matrix": adj_matrix,
        "edges": edges,
        "tickers": tickers,
        "n_edges": len(edges),
    }


def cointegration_summary(
    prices: pd.DataFrame,
    method: str = "both",
) -> pd.DataFrame:
    """
    Generate summary of all cointegration relationships.

    Args:
        prices: Price DataFrame
        method: "johansen", "engle_granger", or "both"

    Returns:
        Summary DataFrame
    """
    results = []

    if method in ["johansen", "both"]:
        johansen_pairs = find_cointegrated_pairs(prices)
        for t1, t2, res in johansen_pairs:
            results.append({
                "pair": f"{t1}-{t2}",
                "method": "Johansen",
                "is_cointegrated": True,
                "trace_stat": res["trace_stat"][0],
                "hedge_ratio": res.get("hedge_ratio"),
                "n_cointegrating": res["n_cointegrating"],
            })

    if method in ["engle_granger", "both"]:
        eg_pairs = find_cointegrated_pairs_eg(prices)
        for t1, t2, res in eg_pairs:
            results.append({
                "pair": f"{t1}-{t2}",
                "method": "Engle-Granger",
                "is_cointegrated": True,
                "coint_stat": res["cointegration_stat"],
                "p_value": res["cointegration_pvalue"],
                "hedge_ratio": res["hedge_ratio"],
                "r_squared": res["r_squared"],
            })

    return pd.DataFrame(results)
