# StAR-E: Statistical Arbitrage & Risk Engine

A comprehensive regime-aware statistical arbitrage system combining classical quantitative finance methods (HMM, GMM, GARCH, Cointegration) with modern deep learning (LSTM, Temporal Fusion Transformer, Graph Attention Networks), deployed as a full-stack application with React+D3.js dashboard.

## Features

### Data Engineering
- **yFinance Integration**: Historical OHLCV data for stocks and ETFs
- **Binance WebSocket**: Real-time cryptocurrency streaming
- **Kalman Filter**: Advanced noise reduction (Standard, Adaptive, UKF)
- **Fractional Differentiation**: Memory-preserving stationarity transformation

### Regime Detection
- **Hidden Markov Models (HMM)**: Temporal regime detection with Viterbi decoding
- **Gaussian Mixture Models (GMM)**: Clustering-based regime identification
- **HMM + GMM Ensemble**: Robust state detection combining both approaches

### Time Series Forecasting
- **SARIMA**: Classical linear forecasting with auto-order selection
- **LSTM with Attention**: Deep learning for non-linear patterns
- **Temporal Fusion Transformer (TFT)**: State-of-the-art multi-horizon forecasting
- **Sharpe & Sortino Loss Functions**: Custom loss for direct risk-adjusted optimization
- **Regime-Aware Ensemble**: Dynamic model weighting based on market state

### Portfolio Optimization
- **Mean-Variance Optimization**: Efficient frontier construction
- **Regime-Aware Risk Scaling**: Dynamic risk adjustment
- **Johansen Cointegration**: Multi-asset cointegration testing
- **Engle-Granger Cointegration**: Two-step cointegration analysis
- **Granger Causality**: Lead-lag relationship detection
- **Graph Attention Networks (GAT)**: Correlation-based asset clustering

### Risk Management
- **VaR (Historical, Parametric, Cornish-Fisher)**: Multiple VaR methodologies
- **CVaR (Expected Shortfall)**: Coherent tail risk measure
- **Monte Carlo Simulation**: Path-dependent risk analysis
- **Stress Testing**: Scenario-based risk assessment
- **Incremental & Component VaR**: Risk decomposition

### Dashboard & API
- **React + D3.js Dashboard**: Interactive data visualizations
- **FastAPI Backend**: RESTful API endpoints
- **Streamlit Dashboard**: Alternative Python-based dashboard
- **MLflow Integration**: Experiment tracking and model registry

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/US30/star-e.git
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

# Ingest crypto data from Binance
star-e ingest --source binance --symbols "BTCUSDT,ETHUSDT" --interval 1h

# Train HMM + GMM ensemble for regime detection
star-e train --model hmm --tickers "AAPL,MSFT,GOOGL"
star-e train --model gmm --tickers "AAPL,MSFT,GOOGL"

# Train forecasting models
star-e train --model sarima --ticker AAPL
star-e train --model lstm --ticker AAPL --loss sharpe
star-e train --model tft --tickers "AAPL,MSFT,GOOGL"

# Run backtest
star-e backtest --strategy regime_aware --tickers "AAPL,MSFT,GOOGL" --rebalance monthly

# Analyze cointegration
star-e cointegration --tickers "AAPL,MSFT,GOOGL" --method both

# Calculate risk metrics with Monte Carlo
star-e risk --tickers "AAPL,MSFT,GOOGL" --monte-carlo --simulations 10000
```

### API Server

```bash
# Start FastAPI server
star-e serve --port 8000

# API documentation: http://localhost:8000/docs
```

### React Dashboard

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Dashboard available at http://localhost:3000
```

### Docker Deployment

```bash
# Build and run all services
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - React Dashboard: http://localhost:3000
# - Streamlit Dashboard: http://localhost:8501
# - MLflow: http://localhost:5000
```

## Project Structure

```
star-e/
├── src/star_e/
│   ├── data/
│   │   ├── ingestion.py          # yFinance data fetching
│   │   ├── binance_stream.py     # Binance WebSocket streaming
│   │   ├── kalman_filter.py      # Kalman filter implementations
│   │   ├── features.py           # Feature engineering
│   │   ├── validation.py         # Data quality checks
│   │   └── storage.py            # DuckDB storage
│   ├── models/
│   │   ├── hmm.py                # Hidden Markov Model
│   │   ├── gmm.py                # Gaussian Mixture Model
│   │   ├── sarima.py             # SARIMA forecaster
│   │   ├── lstm.py               # LSTM + Attention + Sharpe/Sortino loss
│   │   ├── tft.py                # Temporal Fusion Transformer
│   │   ├── garch.py              # GARCH volatility model
│   │   └── ensemble.py           # Regime-aware ensemble
│   ├── portfolio/
│   │   ├── optimizer.py          # Mean-variance optimization
│   │   ├── cointegration.py      # Johansen + Engle-Granger + Granger Causality
│   │   ├── gat.py                # Graph Attention Networks
│   │   ├── risk.py               # VaR, CVaR, Monte Carlo
│   │   └── metrics.py            # Performance metrics
│   ├── backtesting/
│   │   ├── engine.py             # Backtest runner
│   │   └── walk_forward.py       # Walk-forward validation
│   ├── mlops/
│   │   ├── tracking.py           # MLflow integration
│   │   └── drift.py              # PSI, KS tests
│   └── api/
│       ├── main.py               # FastAPI application
│       └── schemas.py            # Pydantic models
├── frontend/                      # React + D3.js dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── RegimeChart.tsx   # D3.js regime overlay chart
│   │   │   ├── TransitionMatrix.tsx  # HMM transition heatmap
│   │   │   ├── AllocationPie.tsx     # Portfolio allocation
│   │   │   ├── VaRChart.tsx          # VaR distribution
│   │   │   └── CorrelationHeatmap.tsx  # GAT visualization
│   │   └── pages/
│   │       ├── Dashboard.tsx
│   │       ├── RegimeAnalysis.tsx
│   │       ├── Portfolio.tsx
│   │       ├── RiskDashboard.tsx
│   │       └── ModelComparison.tsx
├── dashboard/                     # Streamlit dashboard (alternative)
├── tests/                         # Unit and integration tests
├── notebooks/                     # Jupyter notebooks
└── data/                          # Raw and processed data
```

## Models

### Hidden Markov Model (HMM)

```python
from star_e.models import RegimeHMM

hmm = RegimeHMM(n_states=3)
hmm.fit(features)  # features: [returns, volatility]

states, probabilities = hmm.decode(features)
transition_matrix = hmm.get_transition_matrix()
expected_duration = hmm.expected_duration()
```

### Gaussian Mixture Model (GMM)

```python
from star_e.models import RegimeGMM, EnsembleRegimeDetector

gmm = RegimeGMM(n_components=3)
gmm.fit(features)
states = gmm.predict(features)

# Ensemble detection
ensemble = EnsembleRegimeDetector(n_states=3, hmm_weight=0.6, gmm_weight=0.4)
ensemble.fit(features)
states = ensemble.predict(features)
```

### Temporal Fusion Transformer

```python
from star_e.models import TFTForecaster

tft = TFTForecaster(
    max_encoder_length=60,
    max_prediction_length=21,
    hidden_size=64,
)
tft.fit(df, target_col="return")
predictions = tft.predict(df)
```

### Graph Attention Networks

```python
from star_e.portfolio import GATClusterer

clusterer = GATClusterer(hidden_dim=64, embedding_dim=32)
clusterer.fit(returns)
clusters = clusterer.cluster_assets(n_clusters=5)
embeddings = clusterer.get_embeddings()
```

### Monte Carlo VaR

```python
from star_e.portfolio.risk import monte_carlo_var, monte_carlo_paths

var_result = monte_carlo_var(
    weights, returns,
    confidence=0.95,
    n_simulations=10000,
    distribution="t"
)

paths = monte_carlo_paths(
    initial_value=1000000,
    weights=weights,
    returns=returns,
    n_simulations=1000,
    horizon=252
)
```

## Configuration

Environment variables (`.env`):

```bash
# Data
STAR_E_DATA_DIR=/path/to/data
STAR_E_MLFLOW_TRACKING_URI=mlruns

# Binance API (optional)
STAR_E_BINANCE_API_KEY=your_api_key
STAR_E_BINANCE_API_SECRET=your_api_secret

# Model defaults
STAR_E_HMM_N_STATES=3
STAR_E_TFT_HIDDEN_SIZE=64
STAR_E_GAT_EMBEDDING_DIM=32

# Risk
STAR_E_VAR_CONFIDENCE=0.95
STAR_E_MONTE_CARLO_SIMULATIONS=10000
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

# Frontend development
cd frontend
npm start
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/portfolio/optimize` | POST | Optimize portfolio weights |
| `/regime/current` | GET | Get current market regime |
| `/risk/calculate` | POST | Calculate risk metrics |
| `/risk/monte-carlo` | POST | Monte Carlo VaR simulation |
| `/cointegration/find` | POST | Find cointegrated pairs |
| `/models/comparison` | GET | Compare model performance |

## License

MIT License

## Author

Utkarsh Sinha - M.Tech Data Science  
Aditya Siraskar - M.Tech Data Science
