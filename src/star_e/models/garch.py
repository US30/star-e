"""GARCH model for volatility forecasting."""

from typing import Literal, Optional

import mlflow
import numpy as np
from arch import arch_model
from arch.univariate import ARCHModelResult


class GARCHModel:
    """
    GARCH(p,q) model for volatility forecasting.

    Models volatility clustering commonly observed in financial returns:
    - High volatility periods tend to cluster together
    - Captures the "leverage effect" where negative returns increase future volatility
    - Uses Student's t-distribution for fat tails

    The GARCH(1,1) model:
        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

    Where:
        - σ²_t is the conditional variance at time t
        - ω is the long-run average variance weight
        - α is the ARCH coefficient (news impact)
        - β is the GARCH coefficient (persistence)

    Attributes:
        p: GARCH order (lagged variance terms)
        q: ARCH order (lagged squared residual terms)
        model: Fitted ARCH model
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        vol: Literal["GARCH", "EGARCH", "TGARCH"] = "GARCH",
        dist: Literal["normal", "t", "skewt"] = "t",
        mean: Literal["Zero", "Constant", "AR"] = "Constant",
    ):
        """
        Initialize GARCH model.

        Args:
            p: GARCH order (number of lagged variance terms)
            q: ARCH order (number of lagged squared residuals)
            vol: Volatility model type
            dist: Error distribution (t for fat tails)
            mean: Mean model specification
        """
        self.p = p
        self.q = q
        self.vol = vol
        self.dist = dist
        self.mean = mean
        self.model = None
        self.fitted: Optional[ARCHModelResult] = None
        self._scale_factor = 100  # GARCH expects percentage returns

    def fit(self, returns: np.ndarray) -> "GARCHModel":
        """
        Fit GARCH model to return series.

        Args:
            returns: Return series (as decimals, e.g., 0.02 for 2%)

        Returns:
            Self for method chaining
        """
        returns_scaled = returns * self._scale_factor

        self.model = arch_model(
            returns_scaled,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist,
            mean=self.mean,
        )

        self.fitted = self.model.fit(disp="off", show_warning=False)

        if mlflow.active_run():
            mlflow.log_params(
                {
                    "garch_p": self.p,
                    "garch_q": self.q,
                    "garch_vol": self.vol,
                    "garch_dist": self.dist,
                }
            )
            mlflow.log_metrics(
                {
                    "garch_aic": self.fitted.aic,
                    "garch_bic": self.fitted.bic,
                    "garch_loglikelihood": self.fitted.loglikelihood,
                }
            )

        return self

    def forecast(self, horizon: int = 21) -> dict:
        """
        Forecast conditional volatility.

        Args:
            horizon: Forecast horizon in periods

        Returns:
            Dictionary with:
                - 'variance': Forecasted variance array
                - 'volatility': Forecasted volatility (std dev) array
        """
        if self.fitted is None:
            raise RuntimeError("Model must be fitted before forecasting")

        forecast = self.fitted.forecast(horizon=horizon)

        # Convert back from percentage scale
        variance = forecast.variance.iloc[-1].values / (self._scale_factor**2)
        volatility = np.sqrt(variance)

        return {
            "variance": variance,
            "volatility": volatility,
        }

    @property
    def conditional_volatility(self) -> np.ndarray:
        """
        Get in-sample conditional volatility.

        Returns:
            Array of conditional volatility values
        """
        if self.fitted is None:
            raise RuntimeError("Model must be fitted first")
        return self.fitted.conditional_volatility / self._scale_factor

    @property
    def standardized_residuals(self) -> np.ndarray:
        """Get standardized residuals (residuals / conditional volatility)."""
        if self.fitted is None:
            raise RuntimeError("Model must be fitted first")
        return self.fitted.std_resid

    def persistence(self) -> float:
        """
        Calculate volatility persistence (α + β).

        Persistence close to 1 indicates highly persistent volatility shocks.
        """
        if self.fitted is None:
            raise RuntimeError("Model must be fitted first")

        params = self.fitted.params

        if self.vol == "GARCH":
            alpha = params.get("alpha[1]", 0)
            beta = params.get("beta[1]", 0)
            return alpha + beta
        else:
            return np.nan

    def half_life(self) -> float:
        """
        Calculate half-life of volatility shocks.

        Number of periods for a volatility shock to decay by half.
        """
        pers = self.persistence()
        if pers >= 1 or pers <= 0:
            return float("inf")
        return np.log(0.5) / np.log(pers)

    def unconditional_volatility(self) -> float:
        """
        Calculate unconditional (long-run) volatility.

        σ² = ω / (1 - α - β)
        """
        if self.fitted is None:
            raise RuntimeError("Model must be fitted first")

        params = self.fitted.params
        omega = params.get("omega", 0)
        pers = self.persistence()

        if pers >= 1:
            return float("inf")

        variance = omega / (1 - pers)
        return np.sqrt(variance) / self._scale_factor

    def summary(self) -> str:
        """Get model summary."""
        if self.fitted is None:
            raise RuntimeError("Model must be fitted first")
        return str(self.fitted.summary())
