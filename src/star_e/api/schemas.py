"""Pydantic schemas for API request/response validation."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class PortfolioRequest(BaseModel):
    """Request for portfolio optimization."""

    tickers: list[str] = Field(
        ...,
        min_length=2,
        max_length=50,
        description="List of ticker symbols",
        examples=[["AAPL", "MSFT", "GOOGL"]],
    )
    start_date: Optional[date] = Field(
        None,
        description="Start date for analysis",
    )
    end_date: Optional[date] = Field(
        None,
        description="End date for analysis",
    )
    method: str = Field(
        default="max_sharpe",
        description="Optimization method: max_sharpe, min_variance, max_sortino",
    )
    risk_tolerance: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Risk tolerance (0=conservative, 1=aggressive)",
    )


class PortfolioResponse(BaseModel):
    """Response from portfolio optimization."""

    weights: dict[str, float] = Field(
        ...,
        description="Optimal portfolio weights by ticker",
    )
    expected_return: float = Field(
        ...,
        description="Annualized expected return",
    )
    volatility: float = Field(
        ...,
        description="Annualized portfolio volatility",
    )
    sharpe_ratio: float = Field(
        ...,
        description="Portfolio Sharpe ratio",
    )
    regime: str = Field(
        ...,
        description="Current market regime (Bear/Sideways/Bull)",
    )
    var_95: float = Field(
        ...,
        description="95% Value at Risk (daily)",
    )
    cvar_95: float = Field(
        ...,
        description="95% Conditional VaR (daily)",
    )


class RegimeResponse(BaseModel):
    """Response with regime detection results."""

    current_state: str = Field(
        ...,
        description="Current market regime",
    )
    state_probabilities: dict[str, float] = Field(
        ...,
        description="Probability of each regime",
    )
    expected_duration: dict[str, float] = Field(
        ...,
        description="Expected duration in each regime (days)",
    )
    transition_matrix: list[list[float]] = Field(
        ...,
        description="State transition probability matrix",
    )


class RiskRequest(BaseModel):
    """Request for risk calculation."""

    tickers: list[str] = Field(
        ...,
        min_length=1,
        description="List of ticker symbols",
    )
    weights: Optional[list[float]] = Field(
        None,
        description="Portfolio weights (equal if not provided)",
    )
    lookback_days: int = Field(
        default=252,
        ge=30,
        le=2520,
        description="Lookback period for risk calculation",
    )


class RiskResponse(BaseModel):
    """Response with risk metrics."""

    var_95: float = Field(..., description="95% VaR (daily)")
    var_99: float = Field(..., description="99% VaR (daily)")
    cvar_95: float = Field(..., description="95% CVaR (daily)")
    cvar_99: float = Field(..., description="99% CVaR (daily)")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    volatility: float = Field(..., description="Daily volatility")
    annualized_volatility: float = Field(..., description="Annualized volatility")


class ForecastRequest(BaseModel):
    """Request for price/return forecast."""

    ticker: str = Field(..., description="Ticker symbol")
    horizon: int = Field(default=21, ge=1, le=252, description="Forecast horizon (days)")
    model: str = Field(default="ensemble", description="Model: sarima, lstm, ensemble")


class ForecastResponse(BaseModel):
    """Response with forecast results."""

    ticker: str
    horizon: int
    model: str
    forecast: list[float] = Field(..., description="Point forecasts")
    lower_bound: Optional[list[float]] = Field(None, description="Lower confidence bound")
    upper_bound: Optional[list[float]] = Field(None, description="Upper confidence bound")
