"""SARIMA model for time series forecasting."""

from typing import Optional, Literal
import warnings

import mlflow
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from star_e.models.base import BaseForecaster


class SARIMAForecaster(BaseForecaster):
    """
    SARIMA model for linear time series forecasting.

    Seasonal ARIMA captures:
    - Autoregressive (AR) effects: past values influence current value
    - Moving average (MA) effects: past forecast errors influence current value
    - Integration (I): differencing for stationarity
    - Seasonal patterns: repeating patterns at fixed intervals

    Attributes:
        order: (p, d, q) tuple for ARIMA parameters
        seasonal_order: (P, D, Q, s) tuple for seasonal parameters
        model: Fitted SARIMAX model
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 5),
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
    ):
        """
        Initialize SARIMA forecaster.

        Args:
            order: (p, d, q) - AR order, differencing, MA order
            seasonal_order: (P, D, Q, s) - seasonal AR, diff, MA, period
            enforce_stationarity: Constrain AR parameters to ensure stationarity
            enforce_invertibility: Constrain MA parameters to ensure invertibility
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self.fitted = None
        self._series: Optional[np.ndarray] = None

    def auto_order(
        self,
        series: np.ndarray,
        max_p: int = 3,
        max_q: int = 3,
        max_d: int = 2,
        criterion: Literal["aic", "bic"] = "aic",
    ) -> tuple[int, int, int]:
        """
        Automatically select ARIMA order using information criterion.

        Uses grid search over (p, d, q) combinations to find optimal order.

        Args:
            series: Time series to fit
            max_p: Maximum AR order
            max_q: Maximum MA order
            max_d: Maximum differencing order
            criterion: Selection criterion ("aic" or "bic")

        Returns:
            Optimal (p, d, q) tuple
        """
        best_score = float("inf")
        best_order = (1, 1, 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        if p == 0 and q == 0:
                            continue

                        try:
                            model = SARIMAX(
                                series,
                                order=(p, d, q),
                                enforce_stationarity=self.enforce_stationarity,
                                enforce_invertibility=self.enforce_invertibility,
                            )
                            fitted = model.fit(disp=False, maxiter=100)
                            score = getattr(fitted, criterion)

                            if score < best_score:
                                best_score = score
                                best_order = (p, d, q)
                        except Exception:
                            continue

        if mlflow.active_run():
            mlflow.log_params(
                {
                    f"sarima_auto_{criterion}": best_score,
                    "sarima_auto_order": str(best_order),
                }
            )

        return best_order

    def fit(
        self,
        series: np.ndarray,
        exog: Optional[np.ndarray] = None,
    ) -> "SARIMAForecaster":
        """
        Fit SARIMA model to time series.

        Args:
            series: 1D array of time series values
            exog: Optional exogenous variables

        Returns:
            Self for method chaining
        """
        self._series = series

        self.model = SARIMAX(
            series,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted = self.model.fit(disp=False)

        if mlflow.active_run():
            mlflow.log_params(
                {
                    "sarima_order": str(self.order),
                    "sarima_seasonal_order": str(self.seasonal_order),
                }
            )
            mlflow.log_metrics(
                {
                    "sarima_aic": self.fitted.aic,
                    "sarima_bic": self.fitted.bic,
                    "sarima_loglikelihood": self.fitted.llf,
                }
            )

        return self

    def forecast(
        self,
        steps: int = 1,
        exog_forecast: Optional[np.ndarray] = None,
        return_conf_int: bool = True,
        alpha: float = 0.05,
    ) -> dict:
        """
        Generate forecast with optional confidence intervals.

        Args:
            steps: Number of periods to forecast
            exog_forecast: Exogenous variables for forecast period
            return_conf_int: Whether to include confidence intervals
            alpha: Significance level for confidence intervals

        Returns:
            Dictionary with 'mean' forecast and optionally 'lower', 'upper' bounds
        """
        if self.fitted is None:
            raise RuntimeError("Model must be fitted before forecasting")

        forecast = self.fitted.get_forecast(steps=steps, exog=exog_forecast)
        result = {"mean": forecast.predicted_mean.values}

        if return_conf_int:
            conf_int = forecast.conf_int(alpha=alpha)
            result["lower"] = conf_int.iloc[:, 0].values
            result["upper"] = conf_int.iloc[:, 1].values

        return result

    def predict_in_sample(self) -> np.ndarray:
        """Get in-sample predictions."""
        if self.fitted is None:
            raise RuntimeError("Model must be fitted first")
        return self.fitted.fittedvalues.values

    def residuals(self) -> np.ndarray:
        """Get model residuals."""
        if self.fitted is None:
            raise RuntimeError("Model must be fitted first")
        return self.fitted.resid.values

    def summary(self) -> str:
        """Get model summary."""
        if self.fitted is None:
            raise RuntimeError("Model must be fitted first")
        return str(self.fitted.summary())

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        if self.fitted is None:
            raise RuntimeError("Model must be fitted first")
        return self.fitted.aic

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        if self.fitted is None:
            raise RuntimeError("Model must be fitted first")
        return self.fitted.bic
