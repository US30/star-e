"""Data ingestion from yFinance."""

from datetime import date, datetime
from typing import Optional

import pandas as pd
import polars as pl
import yfinance as yf
from pydantic import BaseModel, field_validator

from star_e.config import settings


class TickerData(BaseModel):
    """Validated OHLCV data for a single ticker."""

    ticker: str
    dates: list[date]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[int]
    adj_close: list[float]

    @field_validator("close", "adj_close", "open", "high", "low")
    @classmethod
    def no_negative_prices(cls, v: list[float]) -> list[float]:
        if any(x <= 0 for x in v if x is not None):
            raise ValueError("Prices must be positive")
        return v

    @field_validator("volume")
    @classmethod
    def no_negative_volume(cls, v: list[int]) -> list[int]:
        if any(x < 0 for x in v if x is not None):
            raise ValueError("Volume must be non-negative")
        return v


def fetch_tickers(
    tickers: list[str],
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = "1d",
) -> pl.DataFrame:
    """
    Fetch OHLCV data for multiple tickers from yFinance.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        interval: Data interval (1d, 1h, etc.)

    Returns:
        Polars DataFrame with columns: date, ticker, open, high, low, close, volume, adj_close
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=True,
    )

    if data.empty:
        raise ValueError(f"No data returned for tickers: {tickers}")

    frames = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                ticker_df = data.copy()
            else:
                ticker_df = data[ticker].copy()

            ticker_df = ticker_df.reset_index()
            ticker_df["ticker"] = ticker
            ticker_df.columns = [c.lower().replace(" ", "_") for c in ticker_df.columns]

            if "adj_close" not in ticker_df.columns and "adj close" in ticker_df.columns:
                ticker_df = ticker_df.rename(columns={"adj close": "adj_close"})

            frames.append(ticker_df)
        except KeyError:
            print(f"Warning: No data for ticker {ticker}")
            continue

    if not frames:
        raise ValueError("No valid data retrieved for any ticker")

    combined = pd.concat(frames, axis=0, ignore_index=True)

    return pl.from_pandas(combined)


def update_data(
    existing_df: pl.DataFrame,
    tickers: list[str],
    interval: str = "1d",
) -> pl.DataFrame:
    """
    Update existing data with latest prices.

    Args:
        existing_df: Existing DataFrame with historical data
        tickers: List of tickers to update
        interval: Data interval

    Returns:
        Updated DataFrame with new data appended
    """
    last_date = existing_df["date"].max()
    if last_date is None:
        raise ValueError("Existing DataFrame has no date column or is empty")

    start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date >= end_date:
        return existing_df

    new_data = fetch_tickers(tickers, start_date, end_date, interval)

    return pl.concat([existing_df, new_data]).unique(subset=["date", "ticker"])
