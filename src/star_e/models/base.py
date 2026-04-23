"""Base classes for StAR-E models."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BaseForecaster(ABC):
    """Abstract base class for forecasting models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseForecaster":
        """Fit the model to training data."""
        pass

    @abstractmethod
    def forecast(self, steps: int = 1, **kwargs) -> np.ndarray:
        """Generate forecast for future steps."""
        pass

    def fit_forecast(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        steps: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Fit model and generate forecast."""
        self.fit(X, y)
        return self.forecast(steps, **kwargs)


class BaseRegimeDetector(ABC):
    """Abstract base class for regime detection models."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseRegimeDetector":
        """Fit the regime detector to data."""
        pass

    @abstractmethod
    def decode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode states from observations."""
        pass

    @abstractmethod
    def get_transition_matrix(self) -> np.ndarray:
        """Get state transition probability matrix."""
        pass
