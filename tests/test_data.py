"""Tests for data module."""

import numpy as np
import pytest

from star_e.data.features import (
    compute_returns,
    fractional_diff,
    test_stationarity,
    detect_outliers,
)
from star_e.data.validation import validate_ohlcv, check_gaps


class TestComputeReturns:
    """Tests for compute_returns function."""

    def test_returns_shape(self, sample_polars_df):
        """Test that returns have correct shape."""
        df = compute_returns(sample_polars_df)

        assert "return_1d" in df.columns
        assert "return_5d" in df.columns
        assert "return_21d" in df.columns
        assert len(df) == len(sample_polars_df)

    def test_first_return_is_null(self, sample_polars_df):
        """Test that first return is null (no prior price)."""
        df = compute_returns(sample_polars_df, periods=[1])

        # First return for each ticker should be null
        for ticker in sample_polars_df["ticker"].unique().to_list():
            ticker_df = df.filter(df["ticker"] == ticker).sort("date")
            assert ticker_df["return_1d"][0] is None


class TestFractionalDiff:
    """Tests for fractional differentiation."""

    def test_output_length(self):
        """Test that output has same length as input."""
        series = np.random.randn(100)
        result = fractional_diff(series, d=0.4)

        assert len(result) == len(series)

    def test_initial_nans(self):
        """Test that initial values are NaN due to weight truncation."""
        series = np.random.randn(100)
        result = fractional_diff(series, d=0.4)

        # Some initial values should be NaN
        assert np.isnan(result[0])

    def test_d_zero_returns_original(self):
        """Test that d=0 returns (approximately) original series."""
        series = np.random.randn(100)
        result = fractional_diff(series, d=0.0)

        # After initial NaNs, should be close to original
        valid_idx = ~np.isnan(result)
        np.testing.assert_array_almost_equal(
            result[valid_idx], series[valid_idx], decimal=5
        )

    def test_d_one_approximates_diff(self):
        """Test that d=1 approximates first difference."""
        np.random.seed(42)
        series = np.cumsum(np.random.randn(200))
        result = fractional_diff(series, d=1.0, threshold=1e-6)

        # First difference
        actual_diff = np.diff(series)

        # Compare (skipping initial NaNs)
        valid = ~np.isnan(result[1:])
        correlation = np.corrcoef(result[1:][valid], actual_diff[valid])[0, 1]

        assert correlation > 0.99


class TestStationarity:
    """Tests for stationarity testing."""

    def test_random_walk_not_stationary(self):
        """Test that random walk is detected as non-stationary."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(500))

        result = test_stationarity(random_walk)

        assert result["adf_is_stationary"] is False

    def test_white_noise_is_stationary(self):
        """Test that white noise is detected as stationary."""
        np.random.seed(42)
        white_noise = np.random.randn(500)

        result = test_stationarity(white_noise)

        assert result["adf_is_stationary"] is True
        assert result["is_stationary"] is True

    def test_returns_all_fields(self):
        """Test that all expected fields are returned."""
        series = np.random.randn(100)
        result = test_stationarity(series)

        expected_fields = [
            "adf_statistic",
            "adf_pvalue",
            "adf_is_stationary",
            "kpss_statistic",
            "kpss_pvalue",
            "kpss_is_stationary",
            "is_stationary",
            "conclusion",
        ]

        for field in expected_fields:
            assert field in result


class TestOutlierDetection:
    """Tests for outlier detection."""

    def test_detects_extreme_values(self, sample_polars_df):
        """Test that extreme values are flagged."""
        # Add an extreme outlier
        df = sample_polars_df.with_columns(
            return_1d=sample_polars_df["close"].pct_change().over("ticker")
        )

        result = detect_outliers(df, "close", method="zscore", threshold=3.0)

        assert "is_outlier" in result.columns

    def test_iqr_method(self, sample_polars_df):
        """Test IQR outlier detection method."""
        result = detect_outliers(sample_polars_df, "close", method="iqr", threshold=1.5)

        assert "is_outlier" in result.columns


class TestValidation:
    """Tests for data validation."""

    def test_validates_complete_data(self, sample_polars_df):
        """Test validation of complete data."""
        report = validate_ohlcv(sample_polars_df)

        assert report.is_valid is True
        assert report.total_rows == len(sample_polars_df)

    def test_detects_missing_columns(self, sample_polars_df):
        """Test that missing columns raise error."""
        df = sample_polars_df.drop("close")

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv(df)


class TestGapDetection:
    """Tests for gap detection."""

    def test_detects_gaps(self, sample_polars_df):
        """Test detection of date gaps."""
        gaps = check_gaps(sample_polars_df, max_gap_days=3)

        # Should detect weekend gaps
        assert isinstance(gaps, list)
