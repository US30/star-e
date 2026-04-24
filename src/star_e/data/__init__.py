"""Data engineering module for StAR-E."""

from star_e.data.ingestion import fetch_tickers, update_data
from star_e.data.features import (
    compute_returns,
    compute_rolling_stats,
    fractional_diff,
    test_stationarity,
)
from star_e.data.validation import validate_ohlcv, check_gaps
from star_e.data.storage import save_to_duckdb, load_from_duckdb, query_duckdb
from star_e.data.binance_stream import (
    BinanceStreamManager,
    BinanceHistoricalFetcher,
    KlineData,
    stream_to_storage,
)
from star_e.data.kalman_filter import (
    PriceKalmanFilter,
    AdaptiveKalmanFilter,
    MultivariateKalmanFilter,
    UKFPriceFilter,
    denoise_prices,
)

__all__ = [
    # yFinance ingestion
    "fetch_tickers",
    "update_data",
    # Feature engineering
    "compute_returns",
    "compute_rolling_stats",
    "fractional_diff",
    "test_stationarity",
    # Validation
    "validate_ohlcv",
    "check_gaps",
    # Storage
    "save_to_duckdb",
    "load_from_duckdb",
    "query_duckdb",
    # Binance streaming
    "BinanceStreamManager",
    "BinanceHistoricalFetcher",
    "KlineData",
    "stream_to_storage",
    # Kalman filtering
    "PriceKalmanFilter",
    "AdaptiveKalmanFilter",
    "MultivariateKalmanFilter",
    "UKFPriceFilter",
    "denoise_prices",
]
