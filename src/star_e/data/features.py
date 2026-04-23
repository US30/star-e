"""Feature engineering for time series data."""

from typing import Optional

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller, kpss


def compute_returns(
    df: pl.DataFrame,
    price_col: str = "adj_close",
    periods: list[int] = [1, 5, 21],
) -> pl.DataFrame:
    """
    Compute log returns over multiple periods.

    Args:
        df: DataFrame with price data
        price_col: Column name for prices
        periods: List of return periods (1 = daily, 5 = weekly, 21 = monthly)

    Returns:
        DataFrame with return columns added
    """
    result = df.clone()
    for period in periods:
        result = result.with_columns(
            (pl.col(price_col).log() - pl.col(price_col).shift(period).log())
            .over("ticker")
            .alias(f"return_{period}d")
        )
    return result


def compute_rolling_stats(
    df: pl.DataFrame,
    return_col: str = "return_1d",
    windows: list[int] = [5, 10, 21, 63],
) -> pl.DataFrame:
    """
    Compute rolling statistics: mean, std, skewness, kurtosis.

    Args:
        df: DataFrame with return data
        return_col: Column name for returns
        windows: List of rolling window sizes

    Returns:
        DataFrame with rolling statistic columns added
    """
    result = df.clone()
    for window in windows:
        result = result.with_columns(
            [
                pl.col(return_col)
                .rolling_mean(window)
                .over("ticker")
                .alias(f"mean_{window}d"),
                pl.col(return_col)
                .rolling_std(window)
                .over("ticker")
                .alias(f"vol_{window}d"),
                pl.col(return_col)
                .rolling_skew(window)
                .over("ticker")
                .alias(f"skew_{window}d"),
            ]
        )
    return result


def compute_technical_indicators(
    df: pl.DataFrame,
    price_col: str = "adj_close",
    volume_col: str = "volume",
) -> pl.DataFrame:
    """
    Compute common technical indicators.

    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name
        volume_col: Volume column name

    Returns:
        DataFrame with technical indicator columns
    """
    result = df.clone()

    result = result.with_columns(
        [
            # RSI components
            (pl.col(price_col) - pl.col(price_col).shift(1))
            .over("ticker")
            .alias("price_change"),
            # Moving averages
            pl.col(price_col).rolling_mean(10).over("ticker").alias("sma_10"),
            pl.col(price_col).rolling_mean(20).over("ticker").alias("sma_20"),
            pl.col(price_col).rolling_mean(50).over("ticker").alias("sma_50"),
            # Bollinger Band components
            pl.col(price_col).rolling_mean(20).over("ticker").alias("bb_middle"),
            pl.col(price_col).rolling_std(20).over("ticker").alias("bb_std"),
            # Volume indicators
            pl.col(volume_col).rolling_mean(20).over("ticker").alias("volume_sma_20"),
        ]
    )

    # Bollinger Bands
    result = result.with_columns(
        [
            (pl.col("bb_middle") + 2 * pl.col("bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - 2 * pl.col("bb_std")).alias("bb_lower"),
            (
                (pl.col(price_col) - pl.col("bb_lower"))
                / (pl.col("bb_upper") - pl.col("bb_lower"))
            ).alias("bb_position"),
        ]
    )

    return result


def fractional_diff(
    series: np.ndarray,
    d: float = 0.4,
    threshold: float = 1e-5,
) -> np.ndarray:
    """
    Apply fractional differentiation to preserve memory while achieving stationarity.

    Implements Marcos Lopez de Prado's method from "Advances in Financial Machine Learning".
    Fractional differentiation with d < 1 removes enough non-stationarity to allow
    statistical modeling while preserving predictive information.

    Args:
        series: Time series to differentiate
        d: Differentiation order (0 < d < 1). Lower values preserve more memory.
        threshold: Weight threshold for truncation

    Returns:
        Fractionally differentiated series
    """

    def get_weights(d: float, size: int, threshold: float) -> np.ndarray:
        weights = [1.0]
        for k in range(1, size):
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < threshold:
                break
            weights.append(w)
        return np.array(weights[::-1])

    weights = get_weights(d, len(series), threshold)
    width = len(weights)

    result = np.full(len(series), np.nan)
    for i in range(width - 1, len(series)):
        result[i] = np.dot(weights, series[i - width + 1 : i + 1])

    return result


def test_stationarity(
    series: np.ndarray,
    significance: float = 0.05,
) -> dict:
    """
    Run ADF and KPSS tests for stationarity.

    ADF tests H0: unit root exists (non-stationary)
    KPSS tests H0: series is stationary

    For a truly stationary series: ADF rejects, KPSS fails to reject.

    Args:
        series: Time series to test
        significance: Significance level for tests

    Returns:
        Dictionary with test statistics and stationarity conclusions
    """
    clean_series = series[~np.isnan(series)]

    if len(clean_series) < 20:
        raise ValueError("Series too short for stationarity tests")

    adf_result = adfuller(clean_series, autolag="AIC")
    kpss_result = kpss(clean_series, regression="c", nlags="auto")

    adf_stationary = adf_result[1] < significance
    kpss_stationary = kpss_result[1] > significance

    return {
        "adf_statistic": float(adf_result[0]),
        "adf_pvalue": float(adf_result[1]),
        "adf_is_stationary": adf_stationary,
        "kpss_statistic": float(kpss_result[0]),
        "kpss_pvalue": float(kpss_result[1]),
        "kpss_is_stationary": kpss_stationary,
        "is_stationary": adf_stationary and kpss_stationary,
        "conclusion": _interpret_stationarity(adf_stationary, kpss_stationary),
    }


def _interpret_stationarity(adf_stationary: bool, kpss_stationary: bool) -> str:
    """Interpret stationarity test results."""
    if adf_stationary and kpss_stationary:
        return "stationary"
    elif not adf_stationary and not kpss_stationary:
        return "non-stationary"
    elif adf_stationary and not kpss_stationary:
        return "trend-stationary"
    else:
        return "difference-stationary"


def detect_outliers(
    df: pl.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 3.0,
) -> pl.DataFrame:
    """
    Detect and flag outliers using IQR or Z-score method.

    Args:
        df: Input DataFrame
        column: Column to check for outliers
        method: "iqr" or "zscore"
        threshold: IQR multiplier or Z-score threshold

    Returns:
        DataFrame with is_outlier column added
    """
    if method == "zscore":
        return df.with_columns(
            (
                (pl.col(column) - pl.col(column).mean()).abs() / pl.col(column).std()
                > threshold
            ).alias("is_outlier")
        )
    elif method == "iqr":
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return df.with_columns(
            ((pl.col(column) < lower) | (pl.col(column) > upper)).alias("is_outlier")
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")
