"""Walk-forward validation for time series models."""

from dataclasses import dataclass
from typing import Callable, Optional, Any

import numpy as np


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""

    predictions: np.ndarray
    actuals: np.ndarray
    train_sizes: list[int]
    test_sizes: list[int]
    metrics_per_fold: list[dict]
    aggregate_metrics: dict


class WalkForwardValidator:
    """
    Walk-forward validation for time series.

    Unlike cross-validation, walk-forward respects temporal ordering
    by always training on past data and testing on future data.

    Modes:
    - Expanding window: Training set grows with each fold
    - Rolling window: Training set size stays fixed
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 63,
        min_train_size: int = 252,
        window_type: str = "expanding",
        gap: int = 0,
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of train/test splits
            test_size: Size of each test set (in periods)
            min_train_size: Minimum training set size
            window_type: "expanding" or "rolling"
            gap: Gap between train and test (to prevent lookahead)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.window_type = window_type
        self.gap = gap

    def split(
        self,
        n_samples: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []

        # Calculate split points
        total_test = self.n_splits * self.test_size
        if n_samples < self.min_train_size + total_test + self.gap * self.n_splits:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} splits "
                f"with min_train_size={self.min_train_size}, test_size={self.test_size}"
            )

        for i in range(self.n_splits):
            test_end = n_samples - (self.n_splits - i - 1) * self.test_size
            test_start = test_end - self.test_size

            if self.window_type == "expanding":
                train_start = 0
            else:  # rolling
                train_start = max(0, test_start - self.gap - self.min_train_size)

            train_end = test_start - self.gap

            if train_end - train_start < self.min_train_size:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

        return splits

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable[[], Any],
        fit_method: str = "fit",
        predict_method: str = "predict",
        metric_funcs: Optional[dict[str, Callable]] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array (n_samples,)
            model_factory: Function that creates a new model instance
            fit_method: Name of fit method
            predict_method: Name of predict method
            metric_funcs: Dict of metric name to function(y_true, y_pred)

        Returns:
            WalkForwardResult with predictions and metrics
        """
        if metric_funcs is None:
            metric_funcs = {
                "mse": lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
                "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
                "rmse": lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2)),
            }

        splits = self.split(len(X))

        all_predictions = []
        all_actuals = []
        train_sizes = []
        test_sizes = []
        metrics_per_fold = []

        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create and fit model
            model = model_factory()
            fit_fn = getattr(model, fit_method)
            predict_fn = getattr(model, predict_method)

            fit_fn(X_train, y_train)
            predictions = predict_fn(X_test)

            all_predictions.extend(predictions)
            all_actuals.extend(y_test)
            train_sizes.append(len(train_idx))
            test_sizes.append(len(test_idx))

            # Calculate metrics for this fold
            fold_metrics = {}
            for name, func in metric_funcs.items():
                fold_metrics[name] = func(y_test, predictions)
            metrics_per_fold.append(fold_metrics)

        # Aggregate metrics
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)

        aggregate_metrics = {}
        for name, func in metric_funcs.items():
            aggregate_metrics[name] = func(all_actuals, all_predictions)

        # Add directional accuracy
        if len(all_actuals) > 0:
            direction_correct = np.sign(all_predictions) == np.sign(all_actuals)
            aggregate_metrics["directional_accuracy"] = np.mean(direction_correct)

        return WalkForwardResult(
            predictions=all_predictions,
            actuals=all_actuals,
            train_sizes=train_sizes,
            test_sizes=test_sizes,
            metrics_per_fold=metrics_per_fold,
            aggregate_metrics=aggregate_metrics,
        )

    def validate_forecaster(
        self,
        series: np.ndarray,
        forecaster_factory: Callable[[], Any],
        horizon: int = 1,
        metric_funcs: Optional[dict[str, Callable]] = None,
    ) -> WalkForwardResult:
        """
        Validate a time series forecaster.

        Args:
            series: 1D time series
            forecaster_factory: Function that creates a forecaster
            horizon: Forecast horizon
            metric_funcs: Metric functions

        Returns:
            WalkForwardResult
        """
        if metric_funcs is None:
            metric_funcs = {
                "mse": lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),
                "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
                "mape": lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            }

        splits = self.split(len(series))

        all_predictions = []
        all_actuals = []
        train_sizes = []
        test_sizes = []
        metrics_per_fold = []

        for train_idx, test_idx in splits:
            train_series = series[train_idx]

            # Fit and forecast
            forecaster = forecaster_factory()
            forecaster.fit(train_series)

            # Generate forecasts for test period
            predictions = []
            actuals = []

            for i in range(0, len(test_idx), horizon):
                forecast = forecaster.forecast(steps=min(horizon, len(test_idx) - i))
                if isinstance(forecast, dict):
                    forecast = forecast.get("mean", forecast)

                actual_idx = test_idx[i : i + horizon]
                actual_values = series[actual_idx]

                predictions.extend(forecast[: len(actual_values)])
                actuals.extend(actual_values)

            predictions = np.array(predictions)
            actuals = np.array(actuals)

            all_predictions.extend(predictions)
            all_actuals.extend(actuals)
            train_sizes.append(len(train_idx))
            test_sizes.append(len(test_idx))

            # Calculate metrics
            fold_metrics = {}
            for name, func in metric_funcs.items():
                try:
                    fold_metrics[name] = func(actuals, predictions)
                except Exception:
                    fold_metrics[name] = np.nan
            metrics_per_fold.append(fold_metrics)

        # Aggregate
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)

        aggregate_metrics = {}
        for name, func in metric_funcs.items():
            try:
                aggregate_metrics[name] = func(all_actuals, all_predictions)
            except Exception:
                aggregate_metrics[name] = np.nan

        return WalkForwardResult(
            predictions=all_predictions,
            actuals=all_actuals,
            train_sizes=train_sizes,
            test_sizes=test_sizes,
            metrics_per_fold=metrics_per_fold,
            aggregate_metrics=aggregate_metrics,
        )
