"""Pytest fixtures for StAR-E tests."""

import numpy as np
import pandas as pd
import polars as pl
import pytest


@pytest.fixture
def sample_returns():
    """Generate sample return data."""
    np.random.seed(42)
    n_samples = 500
    n_assets = 5

    returns = np.random.randn(n_samples, n_assets) * 0.02
    returns[:, 0] += 0.0005  # Add slight positive drift

    return returns


@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    n_samples = 500
    n_assets = 5

    returns = np.random.randn(n_samples, n_assets) * 0.02
    prices = 100 * np.cumprod(1 + returns, axis=0)

    return prices


@pytest.fixture
def sample_features():
    """Generate sample features for HMM."""
    np.random.seed(42)
    n_samples = 500

    returns = np.random.randn(n_samples) * 0.02
    volatility = np.abs(returns) + np.random.randn(n_samples) * 0.005

    return np.column_stack([returns, volatility])


@pytest.fixture
def sample_polars_df():
    """Generate sample Polars DataFrame with OHLCV data."""
    np.random.seed(42)
    n_days = 252
    tickers = ["AAPL", "MSFT", "GOOGL"]

    data = []
    for ticker in tickers:
        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")
        base_price = 100 + np.random.randn() * 20

        returns = np.random.randn(n_days) * 0.02
        close = base_price * np.cumprod(1 + returns)

        for i, date in enumerate(dates):
            data.append({
                "date": date,
                "ticker": ticker,
                "open": close[i] * (1 + np.random.randn() * 0.005),
                "high": close[i] * (1 + abs(np.random.randn() * 0.01)),
                "low": close[i] * (1 - abs(np.random.randn() * 0.01)),
                "close": close[i],
                "adj_close": close[i],
                "volume": int(1_000_000 * (1 + np.random.randn() * 0.3)),
            })

    return pl.DataFrame(data)


@pytest.fixture
def cointegrated_prices():
    """Generate cointegrated price series."""
    np.random.seed(42)
    n = 500

    # Random walk
    x = np.cumsum(np.random.randn(n)) + 100

    # Cointegrated series (x + stationary noise)
    noise = np.cumsum(np.random.randn(n) * 0.1)
    noise = noise - noise.mean()  # Make mean-zero
    y = 0.5 * x + noise + 50

    return pd.DataFrame({"A": x, "B": y})
