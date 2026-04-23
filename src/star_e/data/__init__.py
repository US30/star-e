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

__all__ = [
    "fetch_tickers",
    "update_data",
    "compute_returns",
    "compute_rolling_stats",
    "fractional_diff",
    "test_stationarity",
    "validate_ohlcv",
    "check_gaps",
    "save_to_duckdb",
    "load_from_duckdb",
    "query_duckdb",
]
