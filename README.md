# StAR-E: Statistical Arbitrage & Risk Engine

A regime-aware statistical arbitrage system combining classical quantitative finance methods (HMM, GARCH, Cointegration) with modern machine learning (LSTM), deployed as a full-stack application with MLOps practices.

## Features

- **Regime Detection**: Hidden Markov Models identify Bull/Bear/Sideways market states
- **Volatility Modeling**: GARCH forecasts volatility clustering and risk premiums
- **Hybrid Forecasting**: SARIMA + LSTM ensemble adapts to market conditions
- **Cointegration Analysis**: Johansen test identifies mean-reverting pairs
- **Risk Management**: VaR, CVaR, and comprehensive risk metrics
- **Portfolio Optimization**: Mean-variance with regime-aware risk scaling
- **Interactive Dashboard**: Streamlit-based visualization
- **REST API**: FastAPI backend for programmatic access
- **MLOps**: MLflow tracking and drift detection

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/star-e.git
cd star-e

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install package
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Ingest data from yFinance
star-e ingest --tickers "AAPL,MSFT,GOOGL,AMZN,META" --start 2020-01-01

# Train HMM regime detector
star-e train --model hmm --tickers "AAPL,MSFT,GOOGL"

# Run backtest
star-e backtest --strategy max_sharpe --tickers "AAPL,MSFT,GOOGL" --rebalance monthly

# Analyze current regime
star-e regime --ticker SPY

# Calculate risk metrics
star-e risk --tickers "AAPL,MSFT,GOOGL"
```

### API Server

```bash
# Start FastAPI server
star-e serve --port 8000

# API documentation available at http://localhost:8000/docs
```

### Dashboard

```bash
# Launch Streamlit dashboard
star-e dashboard --port 8501
```

### Docker

```bash
# Build and run all services
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - MLflow: http://localhost:5000
```

## Project Structure

```
star-e/
├── src/star_e/
│   ├── data/           # Data ingestion, features, validation
│   ├── models/         # HMM, SARIMA, LSTM, GARCH, Ensemble
│   ├── portfolio/      # Optimizer, cointegration, risk, metrics
│   ├── backtesting/    # Backtest engine, walk-forward validation
│   ├── mlops/          # MLflow tracking, drift detection
│   └── api/            # FastAPI application
├── dashboard/          # Streamlit dashboard
├── tests/              # Unit and integration tests
├── notebooks/          # Jupyter notebooks for analysis
└── data/               # Raw and processed data
```

## Models

### Hidden Markov Model (HMM)

Detects latent market regimes from observable returns and volatility:

```python
from star_e.models import RegimeHMM

hmm = RegimeHMM(n_states=3)
hmm.fit(features)  # features: [returns, volatility]

states, probabilities = hmm.decode(features)
transition_matrix = hmm.get_transition_matrix()
expected_duration = hmm.expected_duration()
```

### GARCH Volatility

Models volatility clustering:

```python
from star_e.models import GARCHModel

garch = GARCHModel(p=1, q=1, dist="t")
garch.fit(returns)

forecast = garch.forecast(horizon=21)
conditional_vol = garch.conditional_volatility
```

### Portfolio Optimization

Mean-variance with regime awareness:

```python
from star_e.portfolio import PortfolioOptimizer

optimizer = PortfolioOptimizer(risk_free_rate=0.04)
result = optimizer.optimize(
    expected_returns,
    cov_matrix,
    current_regime=2,  # Bull
    method="max_sharpe",
)
```

## Risk Metrics

```python
from star_e.portfolio.risk import calculate_var, calculate_cvar, max_drawdown

var_95 = calculate_var(returns, confidence=0.95, method="cornish_fisher")
cvar_95 = calculate_cvar(returns, confidence=0.95)
dd = max_drawdown(returns)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/portfolio/optimize` | POST | Optimize portfolio weights |
| `/regime/current` | GET | Get current market regime |
| `/risk/calculate` | POST | Calculate risk metrics |

## Configuration

Environment variables:

```bash
STAR_E_DATA_DIR=/path/to/data
STAR_E_MLFLOW_TRACKING_URI=mlruns
STAR_E_RISK_FREE_RATE=0.04
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=star_e

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/star_e
```

## License

MIT License

## Author

Utkarsh Sinha - M.Tech Data Science
