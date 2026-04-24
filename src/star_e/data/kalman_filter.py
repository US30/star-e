"""
Kalman Filter for time series noise reduction and state estimation.

Implements univariate and multivariate Kalman filters for financial data
denoising and trend estimation.
"""

import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy import stats
from typing import Optional, Tuple


class PriceKalmanFilter:
    """
    Kalman Filter for price series denoising.

    Uses a local linear trend model to estimate the true underlying price
    and its derivative (momentum/trend).

    State: [price, trend]
    Observation: [observed_price]

    Attributes:
        process_noise: Q matrix scaling factor
        observation_noise: R matrix scaling factor
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
        initial_price: Optional[float] = None,
        initial_trend: float = 0.0,
    ):
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.initial_price = initial_price
        self.initial_trend = initial_trend

        self.kf: Optional[KalmanFilter] = None
        self._history: list[dict] = []

    def _initialize(self, initial_price: float) -> None:
        """Initialize Kalman filter with state space model."""
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        self.kf.x = np.array([[initial_price], [self.initial_trend]])

        self.kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])

        self.kf.H = np.array([[1.0, 0.0]])

        self.kf.P = np.array([[1.0, 0.0], [0.0, 1.0]])

        self.kf.Q = np.array(
            [
                [self.process_noise, 0.0],
                [0.0, self.process_noise * 0.1],
            ]
        )

        self.kf.R = np.array([[self.observation_noise]])

    def filter(self, prices: np.ndarray) -> dict:
        """
        Apply Kalman filter to price series.

        Args:
            prices: Array of observed prices

        Returns:
            Dictionary with filtered prices, trends, and uncertainties
        """
        if self.initial_price is not None:
            self._initialize(self.initial_price)
        else:
            self._initialize(prices[0])

        filtered_prices = np.zeros(len(prices))
        filtered_trends = np.zeros(len(prices))
        price_variance = np.zeros(len(prices))
        trend_variance = np.zeros(len(prices))

        for i, price in enumerate(prices):
            self.kf.predict()

            self.kf.update(np.array([[price]]))

            filtered_prices[i] = self.kf.x[0, 0]
            filtered_trends[i] = self.kf.x[1, 0]
            price_variance[i] = self.kf.P[0, 0]
            trend_variance[i] = self.kf.P[1, 1]

            self._history.append(
                {
                    "observed": price,
                    "filtered_price": filtered_prices[i],
                    "filtered_trend": filtered_trends[i],
                    "price_var": price_variance[i],
                    "trend_var": trend_variance[i],
                }
            )

        return {
            "filtered_prices": filtered_prices,
            "filtered_trends": filtered_trends,
            "price_variance": price_variance,
            "trend_variance": trend_variance,
            "price_std": np.sqrt(price_variance),
            "trend_std": np.sqrt(trend_variance),
        }

    def smooth(self, prices: np.ndarray) -> dict:
        """
        Apply Rauch-Tung-Striebel smoother for offline smoothing.

        Uses future information for better estimates (non-causal).

        Args:
            prices: Array of observed prices

        Returns:
            Dictionary with smoothed estimates
        """
        filter_result = self.filter(prices)

        xs, covs = [], []
        for i, price in enumerate(prices):
            xs.append(np.array([[filter_result["filtered_prices"][i]],
                               [filter_result["filtered_trends"][i]]]))
            covs.append(self.kf.P.copy())

        Xs, Ps, _, _ = self.kf.rts_smoother(
            np.array(xs).squeeze(), np.array(covs)
        )

        return {
            "smoothed_prices": Xs[:, 0],
            "smoothed_trends": Xs[:, 1],
            "smoothed_price_var": Ps[:, 0, 0],
            "smoothed_trend_var": Ps[:, 1, 1],
        }

    def predict_ahead(self, steps: int = 5) -> dict:
        """
        Predict future prices using current state.

        Args:
            steps: Number of steps to predict ahead

        Returns:
            Dictionary with predictions and confidence intervals
        """
        if self.kf is None:
            raise ValueError("Filter must be applied before prediction")

        predictions = np.zeros(steps)
        variances = np.zeros(steps)

        x_pred = self.kf.x.copy()
        P_pred = self.kf.P.copy()

        for i in range(steps):
            x_pred = self.kf.F @ x_pred
            P_pred = self.kf.F @ P_pred @ self.kf.F.T + self.kf.Q

            predictions[i] = x_pred[0, 0]
            variances[i] = P_pred[0, 0]

        std = np.sqrt(variances)

        return {
            "predictions": predictions,
            "variance": variances,
            "std": std,
            "upper_95": predictions + 1.96 * std,
            "lower_95": predictions - 1.96 * std,
        }


class AdaptiveKalmanFilter:
    """
    Adaptive Kalman Filter with time-varying noise parameters.

    Automatically adjusts process and observation noise based on
    innovation sequence analysis.
    """

    def __init__(
        self,
        initial_process_noise: float = 0.01,
        initial_observation_noise: float = 0.1,
        adaptation_rate: float = 0.1,
        window_size: int = 20,
    ):
        self.initial_process_noise = initial_process_noise
        self.initial_observation_noise = initial_observation_noise
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size

        self.kf: Optional[KalmanFilter] = None
        self.innovations: list[float] = []
        self.innovation_covariances: list[float] = []

    def _initialize(self, initial_price: float) -> None:
        """Initialize adaptive Kalman filter."""
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([[initial_price], [0.0]])
        self.kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.kf.H = np.array([[1.0, 0.0]])
        self.kf.P = np.eye(2)
        self.kf.Q = np.eye(2) * self.initial_process_noise
        self.kf.R = np.array([[self.initial_observation_noise]])

    def _adapt_noise(self) -> None:
        """Adapt noise parameters based on innovation sequence."""
        if len(self.innovations) < self.window_size:
            return

        recent_innovations = np.array(self.innovations[-self.window_size:])
        recent_S = np.array(self.innovation_covariances[-self.window_size:])

        empirical_variance = np.var(recent_innovations)
        expected_variance = np.mean(recent_S)

        ratio = empirical_variance / expected_variance if expected_variance > 0 else 1.0

        if ratio > 1.5:
            self.kf.Q *= 1 + self.adaptation_rate
        elif ratio < 0.5:
            self.kf.Q *= 1 - self.adaptation_rate * 0.5

        self.kf.Q = np.clip(self.kf.Q, 1e-6, 1.0)

    def filter(self, prices: np.ndarray) -> dict:
        """Apply adaptive Kalman filter."""
        self._initialize(prices[0])

        filtered_prices = np.zeros(len(prices))
        filtered_trends = np.zeros(len(prices))
        adapted_Q = np.zeros(len(prices))

        for i, price in enumerate(prices):
            self.kf.predict()

            innovation = price - (self.kf.H @ self.kf.x)[0, 0]
            S = (self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R)[0, 0]

            self.innovations.append(innovation)
            self.innovation_covariances.append(S)

            self._adapt_noise()

            self.kf.update(np.array([[price]]))

            filtered_prices[i] = self.kf.x[0, 0]
            filtered_trends[i] = self.kf.x[1, 0]
            adapted_Q[i] = self.kf.Q[0, 0]

        return {
            "filtered_prices": filtered_prices,
            "filtered_trends": filtered_trends,
            "adapted_process_noise": adapted_Q,
        }


class MultivariateKalmanFilter:
    """
    Multivariate Kalman Filter for multiple correlated assets.

    Jointly estimates the state of multiple assets while accounting
    for cross-asset correlations.
    """

    def __init__(
        self,
        n_assets: int,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
    ):
        self.n_assets = n_assets
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.kf: Optional[KalmanFilter] = None

    def _initialize(self, initial_prices: np.ndarray) -> None:
        """Initialize multivariate filter."""
        dim_x = 2 * self.n_assets
        dim_z = self.n_assets

        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

        x = np.zeros((dim_x, 1))
        for i in range(self.n_assets):
            x[2 * i, 0] = initial_prices[i]
        self.kf.x = x

        F = np.zeros((dim_x, dim_x))
        for i in range(self.n_assets):
            F[2 * i, 2 * i] = 1.0
            F[2 * i, 2 * i + 1] = 1.0
            F[2 * i + 1, 2 * i + 1] = 1.0
        self.kf.F = F

        H = np.zeros((dim_z, dim_x))
        for i in range(self.n_assets):
            H[i, 2 * i] = 1.0
        self.kf.H = H

        self.kf.P = np.eye(dim_x)
        self.kf.Q = np.eye(dim_x) * self.process_noise
        self.kf.R = np.eye(dim_z) * self.observation_noise

    def filter(self, prices: np.ndarray) -> dict:
        """
        Apply multivariate Kalman filter.

        Args:
            prices: (n_samples, n_assets) array of prices

        Returns:
            Dictionary with filtered results for each asset
        """
        n_samples = len(prices)
        self._initialize(prices[0])

        filtered_prices = np.zeros((n_samples, self.n_assets))
        filtered_trends = np.zeros((n_samples, self.n_assets))

        for t in range(n_samples):
            self.kf.predict()
            self.kf.update(prices[t].reshape(-1, 1))

            for i in range(self.n_assets):
                filtered_prices[t, i] = self.kf.x[2 * i, 0]
                filtered_trends[t, i] = self.kf.x[2 * i + 1, 0]

        return {
            "filtered_prices": filtered_prices,
            "filtered_trends": filtered_trends,
        }


class UKFPriceFilter:
    """
    Unscented Kalman Filter for non-linear state estimation.

    Handles non-linear dynamics better than standard Kalman filter.
    Useful for volatility-adjusted price estimation.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
    ):
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.ukf: Optional[UnscentedKalmanFilter] = None

    def _state_transition(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Non-linear state transition function."""
        price, trend, vol = x
        new_price = price * np.exp(trend * dt)
        new_trend = trend * 0.99
        new_vol = vol * 0.95 + 0.05 * abs(trend)
        return np.array([new_price, new_trend, new_vol])

    def _measurement(self, x: np.ndarray) -> np.ndarray:
        """Measurement function (observe log price)."""
        return np.array([np.log(x[0])])

    def _initialize(self, initial_price: float) -> None:
        """Initialize UKF."""
        points = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2.0, kappa=0.0)

        self.ukf = UnscentedKalmanFilter(
            dim_x=3,
            dim_z=1,
            dt=1.0,
            fx=self._state_transition,
            hx=self._measurement,
            points=points,
        )

        self.ukf.x = np.array([initial_price, 0.0, 0.01])
        self.ukf.P = np.diag([1.0, 0.01, 0.001])
        self.ukf.Q = np.diag([self.process_noise, self.process_noise * 0.1, 0.0001])
        self.ukf.R = np.array([[self.observation_noise]])

    def filter(self, prices: np.ndarray) -> dict:
        """Apply UKF to price series."""
        self._initialize(prices[0])

        filtered_prices = np.zeros(len(prices))
        filtered_trends = np.zeros(len(prices))
        filtered_vols = np.zeros(len(prices))

        for i, price in enumerate(prices):
            self.ukf.predict()
            self.ukf.update(np.array([np.log(price)]))

            filtered_prices[i] = self.ukf.x[0]
            filtered_trends[i] = self.ukf.x[1]
            filtered_vols[i] = self.ukf.x[2]

        return {
            "filtered_prices": filtered_prices,
            "filtered_trends": filtered_trends,
            "filtered_volatility": filtered_vols,
        }


def denoise_prices(
    prices: np.ndarray,
    method: str = "standard",
    **kwargs,
) -> np.ndarray:
    """
    Convenience function to denoise price series.

    Args:
        prices: Array of prices
        method: "standard", "adaptive", or "ukf"
        **kwargs: Additional arguments for the filter

    Returns:
        Denoised price series
    """
    if method == "standard":
        kf = PriceKalmanFilter(**kwargs)
        result = kf.filter(prices)
        return result["filtered_prices"]
    elif method == "adaptive":
        kf = AdaptiveKalmanFilter(**kwargs)
        result = kf.filter(prices)
        return result["filtered_prices"]
    elif method == "ukf":
        ukf = UKFPriceFilter(**kwargs)
        result = ukf.filter(prices)
        return result["filtered_prices"]
    else:
        raise ValueError(f"Unknown method: {method}")
