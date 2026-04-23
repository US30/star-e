"""Ensemble forecaster combining multiple models."""

from typing import Any, Optional

import numpy as np

from star_e.models.sarima import SARIMAForecaster
from star_e.models.lstm import LSTMForecaster


class EnsembleForecaster:
    """
    Regime-aware ensemble of forecasters.

    Combines SARIMA and LSTM predictions with weights that vary by market regime.
    The intuition is:
    - SARIMA performs well in sideways/mean-reverting markets (linear patterns)
    - LSTM captures non-linear dynamics in trending markets

    Attributes:
        forecasters: Dictionary mapping model names to instances
        regime_weights: Weights for each model by regime state
    """

    def __init__(
        self,
        forecasters: dict[str, Any],
        regime_weights: Optional[dict[int, dict[str, float]]] = None,
    ):
        """
        Initialize ensemble forecaster.

        Args:
            forecasters: Dictionary mapping names to forecaster instances
            regime_weights: Dictionary mapping regime (0=Bear, 1=Sideways, 2=Bull)
                          to dictionaries of model weights
        """
        self.forecasters = forecasters

        # Default weights: favor SARIMA in sideways, LSTM in trends
        self.regime_weights = regime_weights or {
            0: {"sarima": 0.3, "lstm": 0.7},  # Bear: trust LSTM more (non-linear crashes)
            1: {"sarima": 0.6, "lstm": 0.4},  # Sideways: trust SARIMA more (mean reversion)
            2: {"sarima": 0.4, "lstm": 0.6},  # Bull: slight LSTM edge (momentum)
        }

        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validate that weights sum to 1 for each regime."""
        for regime, weights in self.regime_weights.items():
            total = sum(weights.values())
            if not np.isclose(total, 1.0):
                raise ValueError(
                    f"Weights for regime {regime} sum to {total}, expected 1.0"
                )

            for name in weights:
                if name not in self.forecasters:
                    raise ValueError(
                        f"Weight specified for unknown forecaster: {name}"
                    )

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        **kwargs,
    ) -> "EnsembleForecaster":
        """
        Fit all component forecasters.

        Args:
            features: (n_samples, n_features) input features
            targets: (n_samples,) target values
            **kwargs: Additional arguments passed to forecasters

        Returns:
            Self for method chaining
        """
        for name, forecaster in self.forecasters.items():
            if isinstance(forecaster, SARIMAForecaster):
                forecaster.fit(targets)
            elif isinstance(forecaster, LSTMForecaster):
                forecaster.fit(features, targets, **kwargs)
            else:
                forecaster.fit(features, targets)

        return self

    def forecast(
        self,
        features: np.ndarray,
        current_regime: int,
        steps: int = 1,
    ) -> np.ndarray:
        """
        Generate ensemble forecast based on current regime.

        Args:
            features: Input features for forecasters
            current_regime: Current HMM state (0=Bear, 1=Sideways, 2=Bull)
            steps: Forecast horizon

        Returns:
            Weighted ensemble forecast array
        """
        predictions = {}
        weights = self.regime_weights.get(current_regime, self.regime_weights[1])

        for name, forecaster in self.forecasters.items():
            if name not in weights:
                continue

            if isinstance(forecaster, SARIMAForecaster):
                pred = forecaster.forecast(steps=steps)["mean"]
            elif isinstance(forecaster, LSTMForecaster):
                pred = forecaster.forecast(features, steps=steps)
            else:
                pred = forecaster.forecast(steps=steps)

            # Ensure prediction is array
            if np.isscalar(pred):
                pred = np.array([pred])

            predictions[name] = pred

        # Weighted combination
        ensemble = sum(
            weights[name] * predictions[name]
            for name in predictions
            if name in weights
        )

        return ensemble

    def forecast_all(
        self,
        features: np.ndarray,
        steps: int = 1,
    ) -> dict[str, np.ndarray]:
        """
        Get forecasts from all individual models.

        Args:
            features: Input features
            steps: Forecast horizon

        Returns:
            Dictionary mapping model names to their forecasts
        """
        predictions = {}

        for name, forecaster in self.forecasters.items():
            if isinstance(forecaster, SARIMAForecaster):
                pred = forecaster.forecast(steps=steps)["mean"]
            elif isinstance(forecaster, LSTMForecaster):
                pred = forecaster.forecast(features, steps=steps)
            else:
                pred = forecaster.forecast(steps=steps)

            if np.isscalar(pred):
                pred = np.array([pred])

            predictions[name] = pred

        return predictions

    def update_weights(
        self,
        regime: int,
        weights: dict[str, float],
    ) -> None:
        """
        Update weights for a specific regime.

        Args:
            regime: Regime index (0, 1, or 2)
            weights: Dictionary of model weights (must sum to 1)
        """
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")

        self.regime_weights[regime] = weights

    def optimize_weights(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        regimes: np.ndarray,
        metric: str = "mse",
    ) -> dict[int, dict[str, float]]:
        """
        Optimize ensemble weights based on historical performance.

        Uses grid search to find optimal weights per regime.

        Args:
            features: Historical features
            targets: Historical targets
            regimes: Historical regime labels
            metric: Optimization metric ("mse" or "mae")

        Returns:
            Optimized weights dictionary
        """
        weight_grid = np.arange(0.1, 1.0, 0.1)
        optimized = {}

        for regime in [0, 1, 2]:
            regime_mask = regimes == regime
            if not regime_mask.any():
                optimized[regime] = self.regime_weights[regime]
                continue

            regime_features = features[regime_mask]
            regime_targets = targets[regime_mask]

            # Get individual model predictions
            predictions = {}
            for name, forecaster in self.forecasters.items():
                if isinstance(forecaster, LSTMForecaster):
                    preds = forecaster.predict_sequence(features, targets)
                    predictions[name] = preds[regime_mask[len(targets) - len(preds):]]
                else:
                    predictions[name] = forecaster.predict_in_sample()[regime_mask]

            # Grid search for optimal weights
            best_score = float("inf")
            best_weights = None
            model_names = list(self.forecasters.keys())

            if len(model_names) == 2:
                for w in weight_grid:
                    weights = {model_names[0]: w, model_names[1]: 1 - w}
                    ensemble_pred = sum(
                        weights[name] * predictions[name]
                        for name in predictions
                    )

                    if metric == "mse":
                        score = np.mean((ensemble_pred - regime_targets) ** 2)
                    else:
                        score = np.mean(np.abs(ensemble_pred - regime_targets))

                    if score < best_score:
                        best_score = score
                        best_weights = weights

                optimized[regime] = best_weights or self.regime_weights[regime]
            else:
                optimized[regime] = self.regime_weights[regime]

        return optimized
