"""FastAPI application for StAR-E."""

from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from star_e.api.schemas import (
    PortfolioRequest,
    PortfolioResponse,
    RegimeResponse,
    RiskRequest,
    RiskResponse,
    HealthResponse,
)

app = FastAPI(
    title="StAR-E API",
    description="Statistical Arbitrage & Risk Engine - API for portfolio optimization and risk management",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "StAR-E API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.post("/portfolio/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest):
    """
    Optimize portfolio weights based on current regime.

    Returns optimal weights, expected return, volatility, and risk metrics.
    """
    import numpy as np
    from star_e.data import load_from_duckdb
    from star_e.portfolio import PortfolioOptimizer
    from star_e.portfolio.risk import calculate_var, calculate_cvar
    from star_e.models import RegimeHMM

    try:
        df = load_from_duckdb("prices")
    except Exception:
        raise HTTPException(status_code=500, detail="No data available. Run 'star-e ingest' first.")

    df = df.filter(df["ticker"].is_in(request.tickers))

    if request.start_date:
        df = df.filter(df["date"] >= str(request.start_date))
    if request.end_date:
        df = df.filter(df["date"] <= str(request.end_date))

    # Pivot to returns matrix
    pivot_df = df.pivot(index="date", columns="ticker", values="return_1d")
    pivot_df = pivot_df.drop_nulls()

    if pivot_df.height < 60:
        raise HTTPException(status_code=400, detail="Insufficient data for optimization")

    tickers = [t for t in request.tickers if t in pivot_df.columns]
    returns = pivot_df.select(tickers).to_numpy()

    # Detect regime
    market_returns = returns.mean(axis=1)
    X = np.column_stack([market_returns, np.abs(market_returns)])

    hmm = RegimeHMM(n_states=3)
    hmm.fit(X)
    states, probs = hmm.decode(X)
    current_regime = int(states[-1])

    # Optimize
    exp_ret = np.mean(returns[-63:], axis=0)
    cov = np.cov(returns[-252:].T)

    optimizer = PortfolioOptimizer(risk_free_rate=0.04)
    result = optimizer.optimize(
        exp_ret,
        cov,
        current_regime=current_regime,
        method=request.method,
        returns_history=returns if request.method == "max_sortino" else None,
    )

    # Risk metrics
    portfolio_returns = returns @ result["weights"]
    var_95 = calculate_var(portfolio_returns, 0.95)
    cvar_95 = calculate_cvar(portfolio_returns, 0.95)

    regime_names = ["Bear", "Sideways", "Bull"]

    return PortfolioResponse(
        weights={t: float(w) for t, w in zip(tickers, result["weights"])},
        expected_return=float(result["expected_return"]) * 252,
        volatility=float(result["volatility"]) * np.sqrt(252),
        sharpe_ratio=float(result["sharpe"]),
        regime=regime_names[current_regime],
        var_95=float(var_95),
        cvar_95=float(cvar_95),
    )


@app.get("/regime/current", response_model=RegimeResponse)
async def get_current_regime(
    ticker: str = "SPY",
):
    """
    Get current market regime from HMM.

    Returns state probabilities and transition information.
    """
    import numpy as np
    from star_e.data import load_from_duckdb
    from star_e.models import RegimeHMM

    try:
        df = load_from_duckdb("prices")
    except Exception:
        raise HTTPException(status_code=500, detail="No data available")

    df = df.filter(df["ticker"] == ticker).sort("date")
    returns = df["return_1d"].drop_nulls().to_numpy()

    if len(returns) < 100:
        raise HTTPException(status_code=400, detail="Insufficient data for regime detection")

    X = np.column_stack([returns, np.abs(returns)])

    hmm = RegimeHMM(n_states=3)
    hmm.fit(X)

    states, probs = hmm.decode(X)
    current_probs = probs[-1]

    state_names = ["Bear", "Sideways", "Bull"]
    current_state = state_names[states[-1]]

    return RegimeResponse(
        current_state=current_state,
        state_probabilities={
            name: float(prob) for name, prob in zip(state_names, current_probs)
        },
        expected_duration=hmm.expected_duration(),
        transition_matrix=hmm.get_transition_matrix().tolist(),
    )


@app.post("/risk/calculate", response_model=RiskResponse)
async def calculate_risk(request: RiskRequest):
    """Calculate risk metrics for given tickers."""
    import numpy as np
    from star_e.data import load_from_duckdb
    from star_e.portfolio.risk import (
        calculate_var,
        calculate_cvar,
        max_drawdown,
    )

    try:
        df = load_from_duckdb("prices")
    except Exception:
        raise HTTPException(status_code=500, detail="No data available")

    df = df.filter(df["ticker"].is_in(request.tickers))

    pivot_df = df.pivot(index="date", columns="ticker", values="return_1d")
    pivot_df = pivot_df.drop_nulls()

    tickers = [t for t in request.tickers if t in pivot_df.columns]
    returns = pivot_df.select(tickers).to_numpy()

    # Equal weights if not provided
    weights = np.array(request.weights) if request.weights else np.ones(len(tickers)) / len(tickers)

    portfolio_returns = returns @ weights

    dd = max_drawdown(portfolio_returns)

    return RiskResponse(
        var_95=calculate_var(portfolio_returns, 0.95),
        var_99=calculate_var(portfolio_returns, 0.99),
        cvar_95=calculate_cvar(portfolio_returns, 0.95),
        cvar_99=calculate_cvar(portfolio_returns, 0.99),
        max_drawdown=dd["max_drawdown"],
        volatility=float(np.std(portfolio_returns)),
        annualized_volatility=float(np.std(portfolio_returns) * np.sqrt(252)),
    )
