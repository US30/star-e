"""Tests for models module."""

import numpy as np
import pytest

from star_e.models.hmm import RegimeHMM
from star_e.models.garch import GARCHModel
from star_e.models.sarima import SARIMAForecaster


class TestRegimeHMM:
    """Tests for Hidden Markov Model."""

    def test_fit_decode(self, sample_features):
        """Test that HMM can fit and decode."""
        hmm = RegimeHMM(n_states=3)
        hmm.fit(sample_features)

        states, probs = hmm.decode(sample_features)

        assert len(states) == len(sample_features)
        assert probs.shape == (len(sample_features), 3)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_state_labels_ordered(self, sample_features):
        """Test that states are ordered by mean return."""
        hmm = RegimeHMM(n_states=3)
        hmm.fit(sample_features)

        # After fitting, states should be ordered Bear < Sideways < Bull
        assert hmm.state_labels == ["Bear", "Sideways", "Bull"]

    def test_transition_matrix_valid(self, sample_features):
        """Test that transition matrix rows sum to 1."""
        hmm = RegimeHMM(n_states=3)
        hmm.fit(sample_features)

        trans = hmm.get_transition_matrix()

        assert trans.shape == (3, 3)
        assert np.allclose(trans.sum(axis=1), 1.0)
        assert (trans >= 0).all()
        assert (trans <= 1).all()

    def test_expected_duration_positive(self, sample_features):
        """Test that expected durations are positive."""
        hmm = RegimeHMM(n_states=3)
        hmm.fit(sample_features)

        durations = hmm.expected_duration()

        for label, duration in durations.items():
            assert duration > 0

    def test_select_n_states(self, sample_features):
        """Test automatic state selection."""
        hmm = RegimeHMM(n_states=3)

        optimal = hmm.select_n_states(sample_features, state_range=range(2, 5))

        assert 2 <= optimal <= 4

    def test_predict(self, sample_features):
        """Test predict method."""
        hmm = RegimeHMM(n_states=3)
        hmm.fit(sample_features)

        predictions = hmm.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert all(p in [0, 1, 2] for p in predictions)

    def test_unfitted_raises_error(self, sample_features):
        """Test that unfitted model raises error."""
        hmm = RegimeHMM(n_states=3)

        with pytest.raises(RuntimeError, match="fitted"):
            hmm.decode(sample_features)


class TestGARCHModel:
    """Tests for GARCH volatility model."""

    def test_fit_forecast(self, sample_returns):
        """Test GARCH fit and forecast."""
        returns = sample_returns[:, 0]

        garch = GARCHModel(p=1, q=1)
        garch.fit(returns)

        forecast = garch.forecast(horizon=21)

        assert "variance" in forecast
        assert "volatility" in forecast
        assert len(forecast["volatility"]) == 21

    def test_conditional_volatility(self, sample_returns):
        """Test conditional volatility calculation."""
        returns = sample_returns[:, 0]

        garch = GARCHModel(p=1, q=1)
        garch.fit(returns)

        cond_vol = garch.conditional_volatility

        assert len(cond_vol) == len(returns)
        assert (cond_vol > 0).all()

    def test_persistence(self, sample_returns):
        """Test persistence calculation."""
        returns = sample_returns[:, 0]

        garch = GARCHModel(p=1, q=1)
        garch.fit(returns)

        persistence = garch.persistence()

        # Persistence should be between 0 and 1 for stationary volatility
        assert 0 < persistence < 1

    def test_half_life(self, sample_returns):
        """Test half-life calculation."""
        returns = sample_returns[:, 0]

        garch = GARCHModel(p=1, q=1)
        garch.fit(returns)

        half_life = garch.half_life()

        assert half_life > 0


class TestSARIMAForecaster:
    """Tests for SARIMA forecaster."""

    def test_fit_forecast(self, sample_returns):
        """Test SARIMA fit and forecast."""
        series = sample_returns[:, 0]

        sarima = SARIMAForecaster(
            order=(1, 0, 1),
            seasonal_order=(0, 0, 0, 1),
        )
        sarima.fit(series)

        forecast = sarima.forecast(steps=10)

        assert "mean" in forecast
        assert "lower" in forecast
        assert "upper" in forecast
        assert len(forecast["mean"]) == 10

    def test_aic_bic(self, sample_returns):
        """Test AIC/BIC properties."""
        series = sample_returns[:, 0]

        sarima = SARIMAForecaster(order=(1, 0, 1), seasonal_order=(0, 0, 0, 1))
        sarima.fit(series)

        assert sarima.aic is not None
        assert sarima.bic is not None
        assert sarima.aic < sarima.bic  # AIC typically less than BIC

    def test_in_sample_predictions(self, sample_returns):
        """Test in-sample predictions."""
        series = sample_returns[:, 0]

        sarima = SARIMAForecaster(order=(1, 0, 1), seasonal_order=(0, 0, 0, 1))
        sarima.fit(series)

        predictions = sarima.predict_in_sample()

        assert len(predictions) == len(series)

    def test_auto_order(self, sample_returns):
        """Test automatic order selection."""
        series = sample_returns[:100, 0]  # Use smaller sample for speed

        sarima = SARIMAForecaster()
        order = sarima.auto_order(series, max_p=2, max_q=2)

        assert len(order) == 3
        assert all(isinstance(x, int) for x in order)
