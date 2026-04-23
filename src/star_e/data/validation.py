"""Data validation utilities."""

from datetime import timedelta
from typing import Optional

import polars as pl
from pydantic import BaseModel, Field


class DataQualityReport(BaseModel):
    """Report of data quality issues."""

    total_rows: int
    missing_values: dict[str, int]
    date_gaps: list[dict]
    negative_prices: int
    zero_volume_days: int
    duplicate_rows: int
    is_valid: bool
    issues: list[str] = Field(default_factory=list)


def validate_ohlcv(df: pl.DataFrame) -> DataQualityReport:
    """
    Validate OHLCV data quality.

    Checks:
    - Missing values in required columns
    - Negative or zero prices
    - High < Low violations
    - Duplicate date-ticker combinations
    - Zero volume days

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataQualityReport with validation results
    """
    issues = []
    required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Count missing values
    missing_values = {}
    for col in required_cols:
        null_count = df[col].null_count()
        if null_count > 0:
            missing_values[col] = null_count
            issues.append(f"{null_count} missing values in {col}")

    # Check for negative prices
    price_cols = ["open", "high", "low", "close"]
    negative_prices = 0
    for col in price_cols:
        neg_count = df.filter(pl.col(col) <= 0).height
        negative_prices += neg_count
    if negative_prices > 0:
        issues.append(f"{negative_prices} negative or zero prices found")

    # Check High >= Low
    hl_violations = df.filter(pl.col("high") < pl.col("low")).height
    if hl_violations > 0:
        issues.append(f"{hl_violations} rows where High < Low")

    # Check for duplicates
    duplicate_rows = df.height - df.unique(subset=["date", "ticker"]).height
    if duplicate_rows > 0:
        issues.append(f"{duplicate_rows} duplicate date-ticker combinations")

    # Count zero volume days
    zero_volume_days = df.filter(pl.col("volume") == 0).height

    return DataQualityReport(
        total_rows=df.height,
        missing_values=missing_values,
        date_gaps=[],  # Will be filled by check_gaps
        negative_prices=negative_prices,
        zero_volume_days=zero_volume_days,
        duplicate_rows=duplicate_rows,
        is_valid=len(issues) == 0,
        issues=issues,
    )


def check_gaps(
    df: pl.DataFrame,
    max_gap_days: int = 5,
) -> list[dict]:
    """
    Check for gaps in date coverage.

    Args:
        df: DataFrame with date and ticker columns
        max_gap_days: Maximum allowed gap in trading days

    Returns:
        List of gap records with ticker, start_date, end_date, gap_days
    """
    gaps = []

    for ticker in df["ticker"].unique().to_list():
        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
        dates = ticker_df["date"].to_list()

        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i - 1]).days
            if gap > max_gap_days:
                gaps.append(
                    {
                        "ticker": ticker,
                        "start_date": dates[i - 1],
                        "end_date": dates[i],
                        "gap_days": gap,
                    }
                )

    return gaps


def clean_data(
    df: pl.DataFrame,
    fill_method: str = "forward",
    remove_outliers: bool = True,
    outlier_threshold: float = 5.0,
) -> pl.DataFrame:
    """
    Clean OHLCV data by handling missing values and outliers.

    Args:
        df: Raw DataFrame
        fill_method: Method for filling missing values ("forward", "linear")
        remove_outliers: Whether to remove extreme outliers
        outlier_threshold: Z-score threshold for outlier removal

    Returns:
        Cleaned DataFrame
    """
    result = df.clone()

    # Forward fill missing prices within each ticker
    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        if col in result.columns:
            if fill_method == "forward":
                result = result.with_columns(
                    pl.col(col).forward_fill().over("ticker").alias(col)
                )

    # Fill missing volume with 0
    if "volume" in result.columns:
        result = result.with_columns(pl.col("volume").fill_null(0))

    # Remove extreme return outliers if requested
    if remove_outliers and "return_1d" in result.columns:
        mean_ret = result["return_1d"].mean()
        std_ret = result["return_1d"].std()
        result = result.filter(
            (pl.col("return_1d").is_null())
            | (
                (pl.col("return_1d") - mean_ret).abs() / std_ret
                < outlier_threshold
            )
        )

    return result
