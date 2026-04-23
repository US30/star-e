# StAR-E: Statistical Arbitrage & Risk Engine
## Comprehensive Project Documentation

---

## Context

**Problem**: M.Tech Data Science students need portfolio projects that demonstrate mastery in statistical theory, non-linear time series, and production deployment. Generic ML projects fail to differentiate candidates in competitive quant/data science roles.

**Solution**: StAR-E is a regime-aware statistical arbitrage system that combines classical quantitative finance methods (HMM, GARCH, Cointegration) with modern machine learning (LSTM), deployed as a full-stack application with proper MLOps practices.

**Outcome**: A deployable, well-documented project demonstrating end-to-end quantitative data science skills.

**Timeline**: 10-12 weeks (achievable scope)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Scope Modifications from Original Spec](#2-scope-modifications-from-original-spec)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Module Specifications](#5-module-specifications)
6. [Data Engineering & Preprocessing](#6-data-engineering--preprocessing)
7. [Statistical Regime Detection](#7-statistical-regime-detection)
8. [Time Series & Volatility Modeling](#8-time-series--volatility-modeling)
9. [Portfolio Optimization](#9-portfolio-optimization)
10. [Risk Management](#10-risk-management)
11. [Dashboard & API](#11-dashboard--api)
12. [MLOps & Deployment](#12-mlops--deployment)
13. [Phased Implementation Plan](#13-phased-implementation-plan)
14. [Testing Strategy](#14-testing-strategy)
15. [Project Structure](#15-project-structure)
16. [Verification & Demo](#16-verification--demo)
17. [Interview Talking Points](#17-interview-talking-points)

---

## 1. Executive Summary

### Project Vision

StAR-E identifies statistical arbitrage opportunities by detecting market regimes (Bull/Bear/Sideways) using Hidden Markov Models, forecasting returns with hybrid SARIMA+LSTM models, and constructing risk-managed portfolios that adapt to the current regime. The system provides real-time risk metrics (VaR, CVaR) and a Streamlit dashboard for visualization.

### Key Differentiators

| Feature | Why It Matters |
|---------|----------------|
| **HMM Regime Detection** | Shows understanding of latent state models, Viterbi decoding, transition dynamics |
| **GARCH Volatility Clustering** | Core quantitative finance skill, models fat-tailed distributions |
| **Cointegration Analysis** | Statistical arbitrage foundation, demonstrates econometric rigor |
| **LSTM + SARIMA Hybrid** | Bridges classical statistics and deep learning |
| **Regime-Aware Portfolio Construction** | Dynamic risk management, not static allocation |
| **MLflow Integration** | Production-ready practices, experiment reproducibility |
| **Full Backtesting Framework** | Validates claims with walk-forward optimization |

### Success Metrics

- Backtest Sharpe Ratio > 1.0 (risk-adjusted)
- Information Ratio > 0.5
- Maximum Drawdown < 20%
- Model drift detection alerts within 1 day
- End-to-end pipeline runtime < 5 minutes
- Test coverage > 60%

---

## 2. Scope Modifications from Original Spec

### Components Retained (Core)

| Original Component | Status | Rationale |
|-------------------|--------|-----------|
| yFinance data ingestion | **Retained** | Reliable, free, sufficient for portfolio demo |
| HMM regime detection | **Retained** | Project differentiator, achievable with hmmlearn |
| GARCH volatility | **Retained** | Industry standard, well-documented |
| SARIMA forecasting | **Retained** | Strong baseline, interpretable |
| Cointegration (Johansen) | **Retained** | Core statistical arbitrage skill |
| VaR/CVaR calculation | **Retained** | Essential risk metrics |
| Backtesting framework | **Retained** | Validates all model claims |
| Docker deployment | **Retained** | Production readiness |

### Components Modified

| Original | Modified To | Rationale |
|----------|-------------|-----------|
| React + D3.js dashboard | **Streamlit** | 10x faster development, adequate for data science portfolio |
| FastAPI + WebSockets | **FastAPI REST only** | WebSockets add complexity without proportional value |
| Kalman Filter + Fractional Differentiation | **Fractional Differentiation only** | Pick one; fractional diff is more novel |
| LSTM + Temporal Fusion Transformer | **LSTM with attention** | TFT requires extensive tuning; LSTM is 80% value at 20% complexity |
| Johansen + Engle-Granger | **Johansen only** | Johansen subsumes two-variable case |
| Sharpe + Sortino loss | **Sortino only** | One custom loss done well > two done superficially |
| HMM + GMM | **HMM primary, GMM exploratory** | Avoid diluting focus |

### Components Deferred (Phase 4 / Stretch)

| Component | Reason for Deferral |
|-----------|---------------------|
| Graph Attention Networks (GAT) | Research-grade complexity, poor ROI |
| Temporal Fusion Transformer | Requires 2+ weeks of tuning alone |
| Real-time Binance WebSocket | Regulatory complexity, API instability |
| Monte Carlo VaR simulation | Nice-to-have after analytical VaR works |
| Granger Causality networks | Interesting but tangential to core value |
| Bayesian posterior estimation | Can add after point estimates work |

### Components Removed

| Component | Reason |
|-----------|--------|
| Binance integration | Regulatory complexity, not needed for portfolio demo |
| Custom D3.js visualizations | Streamlit + Plotly is sufficient |
| Multiple cointegration methods | Redundant; Johansen is comprehensive |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       PRESENTATION LAYER                             │
│  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────┐ │
│  │    Streamlit      │  │     FastAPI       │  │   CLI (typer)   │ │
│  │    Dashboard      │  │    REST API       │  │   Interface     │ │
│  │                   │  │                   │  │                 │ │
│  │  - Performance    │  │  - /portfolio     │  │  - backtest     │ │
│  │  - Regime View    │  │  - /regime        │  │  - train        │ │
│  │  - Risk Metrics   │  │  - /risk          │  │  - ingest       │ │
│  │  - Model Compare  │  │  - /forecast      │  │  - report       │ │
│  └───────────────────┘  └───────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                             │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Pipeline Manager                              ││
│  │         Coordinates: Ingest → Process → Model → Report          ││
│  │                                                                  ││
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ││
│  │   │  Ingest  │──▶│ Feature  │──▶│  Model   │──▶│ Backtest │   ││
│  │   │   Task   │   │   Task   │   │   Task   │   │   Task   │   ││
│  │   └──────────┘   └──────────┘   └──────────┘   └──────────┘   ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         MODEL LAYER                                  │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │   FORECASTERS   │  │ REGIME DETECTOR │  │ VOLATILITY MODELS   │ │
│  │                 │  │                 │  │                     │ │
│  │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────────┐  │ │
│  │  │  SARIMA   │  │  │  │    HMM    │  │  │  │  GARCH(1,1)   │  │ │
│  │  └───────────┘  │  │  │           │  │  │  └───────────────┘  │ │
│  │  ┌───────────┐  │  │  │  States:  │  │  │                     │ │
│  │  │   LSTM    │  │  │  │  - Bull   │  │  │  Outputs:           │ │
│  │  │  + Attn   │  │  │  │  - Bear   │  │  │  - Conditional Vol  │ │
│  │  └───────────┘  │  │  │  - Sideways│ │  │  - Vol Forecast     │ │
│  │  ┌───────────┐  │  │  └───────────┘  │  │                     │ │
│  │  │  Ensemble │  │  │                 │  │                     │ │
│  │  └───────────┘  │  │  Outputs:       │  │                     │ │
│  │                 │  │  - State probs  │  │                     │ │
│  │  Outputs:       │  │  - Transitions  │  │                     │ │
│  │  - Return pred  │  │  - Duration     │  │                     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐                          │
│  │    PORTFOLIO    │  │  RISK MANAGER   │                          │
│  │    OPTIMIZER    │  │                 │                          │
│  │                 │  │  ┌───────────┐  │                          │
│  │  - Mean-Var     │  │  │    VaR    │  │                          │
│  │  - Regime-Aware │  │  │   (95%)   │  │                          │
│  │  - Constraints  │  │  └───────────┘  │                          │
│  │                 │  │  ┌───────────┐  │                          │
│  │  Outputs:       │  │  │   CVaR    │  │                          │
│  │  - Weights      │  │  │   (95%)   │  │                          │
│  │  - Allocations  │  │  └───────────┘  │                          │
│  └─────────────────┘  └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                  │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │    INGESTION    │  │  FEATURE ENGINE │  │      STORAGE        │ │
│  │                 │  │                 │  │                     │ │
│  │  - yFinance API │  │  - Returns      │  │  ┌───────────────┐  │ │
│  │  - Rate Limit   │  │  - Rolling Stats│  │  │    DuckDB     │  │ │
│  │  - Validation   │  │  - Tech Indic.  │  │  │   (Primary)   │  │ │
│  │                 │  │  - Frac. Diff   │  │  └───────────────┘  │ │
│  │  Sources:       │  │  - Stationarity │  │  ┌───────────────┐  │ │
│  │  - S&P 500      │  │                 │  │  │   Parquet     │  │ │
│  │  - Sector ETFs  │  │  Tests:         │  │  │   (Cache)     │  │ │
│  │  - VIX          │  │  - ADF          │  │  └───────────────┘  │ │
│  │                 │  │  - KPSS         │  │                     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         MLOPS LAYER                                  │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │     MLFLOW      │  │  DRIFT MONITOR  │  │       CI/CD         │ │
│  │                 │  │                 │  │                     │ │
│  │  - Experiments  │  │  - PSI Tracking │  │  - GitHub Actions   │ │
│  │  - Parameters   │  │  - KS Tests     │  │  - Linting (ruff)   │ │
│  │  - Metrics      │  │  - Alerts       │  │  - Type Check       │ │
│  │  - Artifacts    │  │                 │  │  - Unit Tests       │ │
│  │  - Registry     │  │  Thresholds:    │  │                     │ │
│  │                 │  │  PSI > 0.1 warn │  │                     │ │
│  │                 │  │  PSI > 0.25 act │  │                     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. INGESTION
   yFinance API ──▶ Raw OHLCV Data ──▶ Validation ──▶ DuckDB

2. FEATURE ENGINEERING
   Raw Data ──▶ Returns ──▶ Rolling Stats ──▶ Fractional Diff ──▶ Features Table

3. REGIME DETECTION
   Features ──▶ HMM Training ──▶ Viterbi Decoding ──▶ Regime Labels

4. FORECASTING
   Features + Regime ──▶ SARIMA ──┐
                                   ├──▶ Ensemble Forecast
   Features + Regime ──▶ LSTM ────┘

5. VOLATILITY
   Returns ──▶ GARCH(1,1) ──▶ Conditional Volatility Forecast

6. PORTFOLIO CONSTRUCTION
   Forecasts + Volatility + Regime ──▶ Mean-Variance Optimizer ──▶ Weights

7. RISK CALCULATION
   Weights + Volatility ──▶ VaR/CVaR ──▶ Risk Report

8. BACKTESTING
   Weights + Historical Returns ──▶ Walk-Forward ──▶ Performance Metrics
```

---

## 4. Technology Stack

### Core Dependencies

```toml
[project]
name = "star-e"
version = "0.1.0"
python = ">=3.11,<3.13"

[project.dependencies]
# Data Engineering
yfinance = ">=0.2.40"
duckdb = ">=1.0.0"
polars = ">=0.20.0"        # Faster than pandas for large datasets
pandas = ">=2.2.0"         # Compatibility layer
pyarrow = ">=15.0.0"

# Statistical Models
statsmodels = ">=0.14.0"   # SARIMA, GARCH, ADF, KPSS, Johansen
hmmlearn = ">=0.3.0"       # Hidden Markov Models
arch = ">=6.0.0"           # Advanced GARCH variants
scipy = ">=1.12.0"

# Machine Learning
torch = ">=2.2.0"          # LSTM
scikit-learn = ">=1.4.0"

# Visualization
plotly = ">=5.20.0"
matplotlib = ">=3.8.0"

# API & Dashboard
fastapi = ">=0.110.0"
uvicorn = ">=0.28.0"
streamlit = ">=1.32.0"
pydantic = ">=2.6.0"
pydantic-settings = ">=2.2.0"

# MLOps
mlflow = ">=2.11.0"

# CLI
typer = ">=0.12.0"
rich = ">=13.7.0"

# Testing
pytest = ">=8.1.0"
pytest-cov = ">=5.0.0"
hypothesis = ">=6.100.0"   # Property-based testing

# Code Quality
ruff = ">=0.3.0"
mypy = ">=1.9.0"
```

### Hardware Optimization

```python
# config.py
import torch

def get_device() -> torch.device:
    """Select best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

---

## 5. Module Specifications

### 5.1 Data Module (`src/star_e/data/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `ingestion.py` | Fetch OHLCV data from yFinance | `fetch_tickers()`, `update_data()` |
| `validation.py` | Data quality checks | `validate_ohlcv()`, `check_gaps()` |
| `features.py` | Feature engineering | `compute_returns()`, `compute_rolling_stats()`, `fractional_diff()` |
| `storage.py` | DuckDB operations | `save_features()`, `load_features()`, `query()` |

### 5.2 Models Module (`src/star_e/models/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `base.py` | Abstract model interface | `BaseForecaster`, `BaseRegimeDetector` |
| `hmm.py` | Regime detection | `RegimeHMM` |
| `sarima.py` | Linear forecasting | `SARIMAForecaster` |
| `lstm.py` | Non-linear forecasting | `LSTMForecaster` |
| `garch.py` | Volatility modeling | `GARCHModel` |
| `ensemble.py` | Model combination | `EnsembleForecaster` |

### 5.3 Portfolio Module (`src/star_e/portfolio/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `optimizer.py` | Portfolio construction | `mean_variance_optimize()`, `regime_aware_weights()` |
| `cointegration.py` | Pairs analysis | `johansen_test()`, `find_cointegrated_pairs()` |
| `risk.py` | Risk metrics | `calculate_var()`, `calculate_cvar()`, `max_drawdown()` |
| `metrics.py` | Performance metrics | `sharpe_ratio()`, `sortino_ratio()`, `information_ratio()` |

### 5.4 Backtesting Module (`src/star_e/backtesting/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `engine.py` | Backtest execution | `BacktestEngine` |
| `walk_forward.py` | Walk-forward optimization | `WalkForwardValidator` |
| `report.py` | Results reporting | `BacktestReport` |

### 5.5 API Module (`src/star_e/api/`)

| File | Purpose | Key Endpoints |
|------|---------|---------------|
| `main.py` | FastAPI application | `/portfolio`, `/regime`, `/risk`, `/forecast` |
| `schemas.py` | Pydantic models | `PortfolioRequest`, `RiskResponse`, etc. |

---

## 6. Data Engineering & Preprocessing

### 6.1 Data Ingestion

**Universe**: S&P 500 subset (50 most liquid tickers) + Sector ETFs (XLK, XLF, XLE, etc.) + VIX

```python
# src/star_e/data/ingestion.py

import yfinance as yf
import polars as pl
from pydantic import BaseModel, field_validator
from datetime import date

class TickerData(BaseModel):
    """Validated OHLCV data for a single ticker."""
    ticker: str
    dates: list[date]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[int]
    adj_close: list[float]
    
    @field_validator('close', 'adj_close')
    @classmethod
    def no_negative_prices(cls, v):
        if any(x <= 0 for x in v):
            raise ValueError("Prices must be positive")
        return v

def fetch_tickers(
    tickers: list[str],
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pl.DataFrame:
    """
    Fetch OHLCV data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (1d, 1h, etc.)
    
    Returns:
        Polars DataFrame with multi-ticker OHLCV data
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        group_by='ticker',
        auto_adjust=False,
        progress=True
    )
    
    # Convert to long format
    frames = []
    for ticker in tickers:
        if ticker in data.columns.get_level_values(0):
            ticker_df = data[ticker].copy()
            ticker_df['ticker'] = ticker
            frames.append(ticker_df)
    
    combined = pd.concat(frames, axis=0)
    return pl.from_pandas(combined.reset_index())
```

### 6.2 Feature Engineering

```python
# src/star_e/data/features.py

import polars as pl
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

def compute_returns(df: pl.DataFrame, periods: list[int] = [1, 5, 21]) -> pl.DataFrame:
    """Compute log returns over multiple periods."""
    result = df.clone()
    for period in periods:
        result = result.with_columns(
            (pl.col("adj_close").log() - pl.col("adj_close").shift(period).log())
            .over("ticker")
            .alias(f"return_{period}d")
        )
    return result

def compute_rolling_stats(
    df: pl.DataFrame, 
    windows: list[int] = [5, 10, 21, 63]
) -> pl.DataFrame:
    """Compute rolling mean, std, skewness, kurtosis."""
    result = df.clone()
    for window in windows:
        result = result.with_columns([
            pl.col("return_1d").rolling_mean(window).over("ticker").alias(f"mean_{window}d"),
            pl.col("return_1d").rolling_std(window).over("ticker").alias(f"vol_{window}d"),
        ])
    return result

def fractional_diff(series: np.ndarray, d: float = 0.4, threshold: float = 1e-5) -> np.ndarray:
    """
    Apply fractional differentiation to preserve memory while achieving stationarity.
    
    Implements Marcos López de Prado's method from "Advances in Financial Machine Learning".
    
    Args:
        series: Time series to differentiate
        d: Differentiation order (0 < d < 1)
        threshold: Weight threshold for truncation
        
    Returns:
        Fractionally differentiated series
    """
    def get_weights(d: float, size: int, threshold: float) -> np.ndarray:
        weights = [1.0]
        for k in range(1, size):
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < threshold:
                break
            weights.append(w)
        return np.array(weights[::-1])
    
    weights = get_weights(d, len(series), threshold)
    width = len(weights)
    
    result = np.full(len(series), np.nan)
    for i in range(width - 1, len(series)):
        result[i] = np.dot(weights, series[i - width + 1:i + 1])
    
    return result

def test_stationarity(series: np.ndarray) -> dict:
    """Run ADF and KPSS tests for stationarity."""
    adf_result = adfuller(series, autolag='AIC')
    kpss_result = kpss(series, regression='c', nlags='auto')
    
    return {
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'adf_is_stationary': adf_result[1] < 0.05,
        'kpss_statistic': kpss_result[0],
        'kpss_pvalue': kpss_result[1],
        'kpss_is_stationary': kpss_result[1] > 0.05,
    }
```

### 6.3 Outlier Detection

```python
def detect_outliers(
    df: pl.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 3.0
) -> pl.DataFrame:
    """
    Detect and flag outliers using IQR or Z-score method.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        method: "iqr" or "zscore"
        threshold: IQR multiplier or Z-score threshold
        
    Returns:
        DataFrame with is_outlier column added
    """
    if method == "zscore":
        return df.with_columns(
            ((pl.col(column) - pl.col(column).mean()).abs() / pl.col(column).std() > threshold)
            .alias("is_outlier")
        )
    elif method == "iqr":
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return df.with_columns(
            ((pl.col(column) < lower) | (pl.col(column) > upper)).alias("is_outlier")
        )
```

---

## 7. Statistical Regime Detection

### 7.1 Hidden Markov Model Implementation

```python
# src/star_e/models/hmm.py

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import mlflow
from typing import Literal

class RegimeHMM:
    """
    Hidden Markov Model for market regime detection.
    
    Identifies latent market states (Bull, Bear, Sideways) from observable
    features like returns and volatility.
    
    Attributes:
        n_states: Number of hidden states (default: 3)
        model: Trained GaussianHMM model
        scaler: Feature scaler
        state_labels: Human-readable state names
    """
    
    def __init__(
        self,
        n_states: int = 3,
        covariance_type: Literal["full", "diag", "spherical", "tied"] = "full",
        n_iter: int = 100,
        random_state: int = 42
    ):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.state_labels = ["Bear", "Sideways", "Bull"]  # Will be reordered after fitting
        
    def fit(self, features: np.ndarray) -> "RegimeHMM":
        """
        Fit HMM to feature matrix.
        
        Args:
            features: (n_samples, n_features) array of returns, volatility, etc.
            
        Returns:
            Self for chaining
        """
        X_scaled = self.scaler.fit_transform(features)
        self.model.fit(X_scaled)
        
        # Reorder states by mean return (ascending: Bear < Sideways < Bull)
        state_means = self.model.means_[:, 0]  # First feature is return
        self._state_order = np.argsort(state_means)
        
        # Log to MLflow
        mlflow.log_params({
            "hmm_n_states": self.n_states,
            "hmm_covariance_type": self.covariance_type,
            "hmm_n_iter": self.n_iter
        })
        
        return self
    
    def decode(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode most likely state sequence using Viterbi algorithm.
        
        Args:
            features: Feature matrix
            
        Returns:
            Tuple of (state_sequence, state_probabilities)
        """
        X_scaled = self.scaler.transform(features)
        log_prob, states = self.model.decode(X_scaled, algorithm="viterbi")
        
        # Remap to ordered states
        ordered_states = np.array([np.where(self._state_order == s)[0][0] for s in states])
        state_probs = self.model.predict_proba(X_scaled)
        
        return ordered_states, state_probs
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get state transition probability matrix."""
        # Reorder according to Bear < Sideways < Bull
        trans = self.model.transmat_[self._state_order][:, self._state_order]
        return trans
    
    def expected_duration(self) -> dict[str, float]:
        """
        Calculate expected duration in each state.
        
        E[duration] = 1 / (1 - P(stay in same state))
        """
        trans = self.get_transition_matrix()
        durations = {}
        for i, label in enumerate(self.state_labels):
            p_stay = trans[i, i]
            durations[label] = 1 / (1 - p_stay) if p_stay < 1 else float('inf')
        return durations
    
    def select_n_states(
        self, 
        features: np.ndarray, 
        state_range: range = range(2, 6),
        criterion: Literal["aic", "bic"] = "bic"
    ) -> int:
        """
        Select optimal number of states using information criteria.
        
        Args:
            features: Feature matrix
            state_range: Range of states to test
            criterion: "aic" or "bic"
            
        Returns:
            Optimal number of states
        """
        X_scaled = self.scaler.fit_transform(features)
        scores = {}
        
        for n in state_range:
            model = hmm.GaussianHMM(
                n_components=n,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state
            )
            model.fit(X_scaled)
            
            log_likelihood = model.score(X_scaled)
            n_params = n * n + 2 * n * X_scaled.shape[1]  # Transition + emissions
            n_samples = X_scaled.shape[0]
            
            if criterion == "aic":
                scores[n] = 2 * n_params - 2 * log_likelihood
            else:  # bic
                scores[n] = n_params * np.log(n_samples) - 2 * log_likelihood
        
        optimal = min(scores, key=scores.get)
        
        mlflow.log_metrics({f"hmm_{criterion}_{n}_states": score for n, score in scores.items()})
        mlflow.log_metric(f"hmm_optimal_states_{criterion}", optimal)
        
        return optimal
```

### 7.2 Regime Visualization

```python
# src/star_e/visualization/regime.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_regime_overlay(
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    states: np.ndarray,
    state_labels: list[str] = ["Bear", "Sideways", "Bull"]
) -> go.Figure:
    """
    Create price chart with regime state overlay.
    
    Args:
        dates: Date index
        prices: Price series
        states: State sequence (0, 1, 2)
        state_labels: Labels for states
        
    Returns:
        Plotly figure
    """
    colors = {"Bear": "rgba(255, 0, 0, 0.3)", 
              "Sideways": "rgba(128, 128, 128, 0.3)", 
              "Bull": "rgba(0, 255, 0, 0.3)"}
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.05)
    
    # Price chart
    fig.add_trace(go.Scatter(x=dates, y=prices, name="Price", 
                             line=dict(color="black", width=1)), row=1, col=1)
    
    # Add regime backgrounds
    current_state = states[0]
    start_idx = 0
    
    for i in range(1, len(states)):
        if states[i] != current_state or i == len(states) - 1:
            label = state_labels[current_state]
            fig.add_vrect(
                x0=dates[start_idx], x1=dates[i-1],
                fillcolor=colors[label],
                layer="below", line_width=0,
                row=1, col=1
            )
            current_state = states[i]
            start_idx = i
    
    # State probability over time
    for i, label in enumerate(state_labels):
        state_mask = states == i
        fig.add_trace(go.Bar(x=dates, y=state_mask.astype(int), name=label), row=2, col=1)
    
    fig.update_layout(
        title="Price with Regime Overlay",
        xaxis2_title="Date",
        yaxis_title="Price",
        yaxis2_title="Regime",
        barmode='stack',
        showlegend=True
    )
    
    return fig
```

---

## 8. Time Series & Volatility Modeling

### 8.1 SARIMA Forecaster

```python
# src/star_e/models/sarima.py

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
import mlflow
from typing import Optional

class SARIMAForecaster:
    """
    SARIMA model for linear time series forecasting.
    
    Seasonal ARIMA captures:
    - Autoregressive (AR) effects
    - Moving average (MA) effects
    - Differencing for stationarity
    - Seasonal patterns
    """
    
    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 5),
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self.fitted = None
        
    def auto_order(
        self,
        series: np.ndarray,
        max_p: int = 5,
        max_q: int = 5,
        criterion: str = "aic"
    ) -> tuple[int, int, int]:
        """
        Automatically select ARIMA order using information criterion.
        
        Uses grid search over (p, d, q) combinations.
        """
        best_score = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                for d in [0, 1]:
                    try:
                        model = SARIMAX(series, order=(p, d, q))
                        fitted = model.fit(disp=False)
                        score = getattr(fitted, criterion)
                        if score < best_score:
                            best_score = score
                            best_order = (p, d, q)
                    except:
                        continue
        
        mlflow.log_params({
            f"sarima_auto_{criterion}": best_score,
            "sarima_auto_order": str(best_order)
        })
        
        return best_order
    
    def fit(self, series: np.ndarray) -> "SARIMAForecaster":
        """Fit SARIMA model to time series."""
        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )
        self.fitted = self.model.fit(disp=False)
        
        mlflow.log_params({
            "sarima_order": str(self.order),
            "sarima_seasonal_order": str(self.seasonal_order)
        })
        mlflow.log_metrics({
            "sarima_aic": self.fitted.aic,
            "sarima_bic": self.fitted.bic
        })
        
        return self
    
    def forecast(
        self,
        steps: int,
        return_conf_int: bool = True,
        alpha: float = 0.05
    ) -> dict:
        """
        Generate forecast with confidence intervals.
        
        Args:
            steps: Number of steps to forecast
            return_conf_int: Include confidence intervals
            alpha: Significance level for intervals
            
        Returns:
            Dict with 'mean', 'lower', 'upper' forecasts
        """
        forecast = self.fitted.get_forecast(steps=steps)
        result = {'mean': forecast.predicted_mean.values}
        
        if return_conf_int:
            conf_int = forecast.conf_int(alpha=alpha)
            result['lower'] = conf_int.iloc[:, 0].values
            result['upper'] = conf_int.iloc[:, 1].values
        
        return result
```

### 8.2 LSTM Forecaster

```python
# src/star_e/models/lstm.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from typing import Optional

class AttentionLSTM(nn.Module):
    """LSTM with self-attention for sequence modeling."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        attention_heads: int = 4
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_hidden = attn_out[:, -1, :]
        return self.fc(last_hidden)


class LSTMForecaster:
    """
    LSTM-based forecaster with attention mechanism.
    
    Captures non-linear patterns in time series that SARIMA cannot model.
    """
    
    def __init__(
        self,
        sequence_length: int = 21,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        device: Optional[torch.device] = None
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or torch.device("cpu")
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
    def _create_sequences(
        self, 
        features: np.ndarray, 
        targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for LSTM."""
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(targets[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        val_split: float = 0.2
    ) -> "LSTMForecaster":
        """
        Train LSTM model.
        
        Args:
            features: (n_samples, n_features) input features
            targets: (n_samples,) target values
            val_split: Validation set fraction
        """
        # Create sequences
        X, y = self._create_sequences(features, targets)
        
        # Train/val split (respecting time order)
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Initialize model
        self.model = AttentionLSTM(
            input_size=features.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        mlflow.log_params({
            "lstm_hidden_size": self.hidden_size,
            "lstm_num_layers": self.num_layers,
            "lstm_sequence_length": self.sequence_length
        })
        mlflow.log_metric("lstm_best_val_loss", best_val_loss)
        
        return self
    
    def forecast(self, features: np.ndarray) -> np.ndarray:
        """Generate forecast from feature sequence."""
        self.model.eval()
        X = torch.FloatTensor(features[-self.sequence_length:]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(X)
        return pred.cpu().numpy().flatten()
```

### 8.3 GARCH Volatility Model

```python
# src/star_e/models/garch.py

import numpy as np
from arch import arch_model
import mlflow

class GARCHModel:
    """
    GARCH(1,1) model for volatility forecasting.
    
    Models volatility clustering commonly observed in financial returns:
    - High volatility periods tend to cluster
    - Captures "leverage effect" (negative returns increase future volatility)
    """
    
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        vol: str = "GARCH",
        dist: str = "t"  # Student's t for fat tails
    ):
        self.p = p
        self.q = q
        self.vol = vol
        self.dist = dist
        self.model = None
        self.fitted = None
        
    def fit(self, returns: np.ndarray) -> "GARCHModel":
        """
        Fit GARCH model to return series.
        
        Args:
            returns: Return series (should be demeaned or percentage returns)
        """
        # Scale returns to percentage (GARCH expects ~0-100 scale)
        returns_scaled = returns * 100
        
        self.model = arch_model(
            returns_scaled,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist
        )
        self.fitted = self.model.fit(disp="off")
        
        mlflow.log_params({
            "garch_p": self.p,
            "garch_q": self.q,
            "garch_dist": self.dist
        })
        mlflow.log_metrics({
            "garch_aic": self.fitted.aic,
            "garch_bic": self.fitted.bic
        })
        
        return self
    
    def forecast(self, horizon: int = 21) -> dict:
        """
        Forecast conditional volatility.
        
        Args:
            horizon: Forecast horizon in periods
            
        Returns:
            Dict with 'variance' and 'volatility' forecasts
        """
        forecast = self.fitted.forecast(horizon=horizon)
        
        # Convert back from percentage scale
        variance = forecast.variance.iloc[-1].values / 10000
        volatility = np.sqrt(variance)
        
        return {
            "variance": variance,
            "volatility": volatility
        }
    
    @property
    def conditional_volatility(self) -> np.ndarray:
        """Get in-sample conditional volatility."""
        return self.fitted.conditional_volatility / 100
```

### 8.4 Ensemble Forecaster

```python
# src/star_e/models/ensemble.py

import numpy as np
from typing import Dict, List

class EnsembleForecaster:
    """
    Regime-aware ensemble of forecasters.
    
    Combines SARIMA and LSTM predictions with weights that vary by regime.
    """
    
    def __init__(
        self,
        forecasters: Dict[str, object],
        regime_weights: Dict[int, Dict[str, float]] = None
    ):
        """
        Args:
            forecasters: Dict mapping name to forecaster instance
            regime_weights: Dict mapping regime (0,1,2) to forecaster weights
        """
        self.forecasters = forecasters
        
        # Default: equal weighting, but favor SARIMA in sideways markets
        self.regime_weights = regime_weights or {
            0: {"sarima": 0.3, "lstm": 0.7},  # Bear: trust LSTM more
            1: {"sarima": 0.6, "lstm": 0.4},  # Sideways: trust SARIMA more
            2: {"sarima": 0.4, "lstm": 0.6},  # Bull: slight LSTM edge
        }
    
    def forecast(
        self,
        features: np.ndarray,
        current_regime: int,
        steps: int = 1
    ) -> np.ndarray:
        """
        Generate ensemble forecast based on current regime.
        
        Args:
            features: Input features for forecasters
            current_regime: Current HMM state (0, 1, or 2)
            steps: Forecast horizon
            
        Returns:
            Weighted ensemble forecast
        """
        predictions = {}
        weights = self.regime_weights[current_regime]
        
        for name, forecaster in self.forecasters.items():
            if name == "sarima":
                pred = forecaster.forecast(steps=steps)["mean"]
            else:  # lstm
                pred = forecaster.forecast(features)
            predictions[name] = pred
        
        ensemble = sum(
            weights[name] * predictions[name]
            for name in predictions
        )
        
        return ensemble
```

---

## 9. Portfolio Optimization

### 9.1 Mean-Variance Optimizer

```python
# src/star_e/portfolio/optimizer.py

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Dict

class PortfolioOptimizer:
    """
    Mean-Variance portfolio optimizer with regime-aware adjustments.
    
    Constructs portfolios on the efficient frontier with optional
    regime-based risk aversion scaling.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.04,
        min_weight: float = 0.0,
        max_weight: float = 0.3,
        regime_risk_scaling: Dict[int, float] = None
    ):
        """
        Args:
            risk_free_rate: Annual risk-free rate
            min_weight: Minimum allocation per asset
            max_weight: Maximum allocation per asset
            regime_risk_scaling: Multiplier for risk aversion by regime
        """
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Higher risk aversion in Bear markets
        self.regime_risk_scaling = regime_risk_scaling or {
            0: 2.0,   # Bear: double risk aversion
            1: 1.0,   # Sideways: normal
            2: 0.7,   # Bull: reduce risk aversion
        }
    
    def _portfolio_return(
        self, 
        weights: np.ndarray, 
        expected_returns: np.ndarray
    ) -> float:
        """Calculate portfolio expected return."""
        return np.dot(weights, expected_returns)
    
    def _portfolio_volatility(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """Calculate portfolio volatility."""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def _sortino_loss(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        returns_history: np.ndarray,
        target_return: float = 0
    ) -> float:
        """
        Negative Sortino ratio (for minimization).
        
        Sortino focuses on downside risk only, making it more appropriate
        for asymmetric return distributions.
        """
        port_return = self._portfolio_return(weights, expected_returns)
        
        # Calculate downside deviation
        portfolio_returns = returns_history @ weights
        downside_returns = np.minimum(portfolio_returns - target_return, 0)
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_std == 0:
            return -1e10  # Perfect score if no downside
        
        sortino = (port_return - self.risk_free_rate / 252) / downside_std
        return -sortino  # Negative for minimization
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_regime: int,
        method: str = "max_sharpe",
        returns_history: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Optimize portfolio weights.
        
        Args:
            expected_returns: (n_assets,) expected returns
            cov_matrix: (n_assets, n_assets) covariance matrix
            current_regime: Current HMM state
            method: "max_sharpe", "min_variance", or "max_sortino"
            returns_history: Historical returns for Sortino calculation
            
        Returns:
            Dict with 'weights', 'expected_return', 'volatility', 'sharpe'
        """
        n_assets = len(expected_returns)
        
        # Initial guess: equal weights
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Constraints: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        
        # Bounds: min/max per asset
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
        
        # Regime-adjusted risk aversion
        risk_scale = self.regime_risk_scaling[current_regime]
        
        if method == "max_sharpe":
            def neg_sharpe(w):
                ret = self._portfolio_return(w, expected_returns)
                vol = self._portfolio_volatility(w, cov_matrix) * risk_scale
                return -(ret - self.risk_free_rate / 252) / vol if vol > 0 else 0
            
            objective = neg_sharpe
            
        elif method == "min_variance":
            def variance(w):
                return self._portfolio_volatility(w, cov_matrix) ** 2 * risk_scale
            
            objective = variance
            
        elif method == "max_sortino":
            if returns_history is None:
                raise ValueError("returns_history required for Sortino optimization")
            
            objective = lambda w: self._sortino_loss(w, expected_returns, returns_history)
        
        result = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000}
        )
        
        optimal_weights = result.x
        port_return = self._portfolio_return(optimal_weights, expected_returns)
        port_vol = self._portfolio_volatility(optimal_weights, cov_matrix)
        sharpe = (port_return - self.risk_free_rate / 252) / port_vol if port_vol > 0 else 0
        
        return {
            "weights": optimal_weights,
            "expected_return": port_return,
            "volatility": port_vol,
            "sharpe": sharpe,
            "regime": current_regime,
            "success": result.success
        }
```

### 9.2 Cointegration Analysis

```python
# src/star_e/portfolio/cointegration.py

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
from typing import List, Tuple, Dict

def johansen_test(
    prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1
) -> Dict:
    """
    Perform Johansen cointegration test.
    
    Args:
        prices: DataFrame with price series as columns
        det_order: Deterministic trend order (-1=no constant, 0=constant, 1=trend)
        k_ar_diff: Number of lagged differences
        
    Returns:
        Dict with test statistics and critical values
    """
    result = coint_johansen(prices.values, det_order=det_order, k_ar_diff=k_ar_diff)
    
    return {
        "trace_stat": result.lr1,
        "trace_crit_95": result.cvt[:, 1],
        "max_eigen_stat": result.lr2,
        "max_eigen_crit_95": result.cvm[:, 1],
        "eigenvectors": result.evec,
        "eigenvalues": result.eig,
        "n_cointegrating": sum(result.lr1 > result.cvt[:, 1])
    }


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    significance: float = 0.05
) -> List[Tuple[str, str, Dict]]:
    """
    Find all cointegrated pairs in universe.
    
    Args:
        prices: DataFrame with price series as columns
        significance: Significance level for test
        
    Returns:
        List of (ticker1, ticker2, test_results) tuples
    """
    tickers = prices.columns.tolist()
    cointegrated = []
    
    for t1, t2 in combinations(tickers, 2):
        pair_prices = prices[[t1, t2]].dropna()
        if len(pair_prices) < 100:
            continue
        
        try:
            result = johansen_test(pair_prices)
            
            # Check if at least one cointegrating relationship
            if result["trace_stat"][0] > result["trace_crit_95"][0]:
                cointegrated.append((t1, t2, result))
        except Exception:
            continue
    
    return cointegrated


def calculate_spread(
    prices: pd.DataFrame,
    hedge_ratio: np.ndarray
) -> pd.Series:
    """
    Calculate cointegration spread (error correction term).
    
    Args:
        prices: Price DataFrame
        hedge_ratio: Eigenvector from Johansen test
        
    Returns:
        Spread series
    """
    return (prices.values @ hedge_ratio).flatten()
```

---

## 10. Risk Management

### 10.1 VaR and CVaR

```python
# src/star_e/portfolio/risk.py

import numpy as np
from scipy import stats
from typing import Dict

def calculate_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Calculate Value at Risk.
    
    Args:
        returns: Return series
        confidence: Confidence level (0.95 = 95%)
        method: "historical", "parametric", or "cornish_fisher"
        
    Returns:
        VaR (positive number representing potential loss)
    """
    alpha = 1 - confidence
    
    if method == "historical":
        var = np.percentile(returns, alpha * 100)
        
    elif method == "parametric":
        mu = np.mean(returns)
        sigma = np.std(returns)
        var = mu + sigma * stats.norm.ppf(alpha)
        
    elif method == "cornish_fisher":
        # Cornish-Fisher expansion for non-normal distributions
        mu = np.mean(returns)
        sigma = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        z = stats.norm.ppf(alpha)
        z_cf = (z + 
                (z**2 - 1) * skew / 6 + 
                (z**3 - 3*z) * (kurt - 3) / 24 - 
                (2*z**3 - 5*z) * skew**2 / 36)
        
        var = mu + sigma * z_cf
    
    return -var  # Return positive VaR


def calculate_cvar(
    returns: np.ndarray,
    confidence: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    
    CVaR is the expected loss given that the loss exceeds VaR.
    More coherent risk measure than VaR.
    
    Args:
        returns: Return series
        confidence: Confidence level
        
    Returns:
        CVaR (positive number)
    """
    var = calculate_var(returns, confidence, method="historical")
    tail_returns = returns[returns <= -var]
    
    if len(tail_returns) == 0:
        return var
    
    return -np.mean(tail_returns)


def max_drawdown(cumulative_returns: np.ndarray) -> Dict:
    """
    Calculate maximum drawdown and duration.
    
    Args:
        cumulative_returns: Cumulative return series (1 + r).cumprod()
        
    Returns:
        Dict with 'max_drawdown', 'peak_idx', 'trough_idx', 'duration'
    """
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    
    max_dd = np.min(drawdowns)
    trough_idx = np.argmin(drawdowns)
    peak_idx = np.argmax(cumulative_returns[:trough_idx + 1])
    
    # Recovery (if any)
    recovery_idx = trough_idx
    for i in range(trough_idx, len(cumulative_returns)):
        if cumulative_returns[i] >= cumulative_returns[peak_idx]:
            recovery_idx = i
            break
    
    return {
        "max_drawdown": -max_dd,  # Positive percentage
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "recovery_idx": recovery_idx,
        "duration": trough_idx - peak_idx,
        "recovery_duration": recovery_idx - trough_idx
    }
```

### 10.2 Performance Metrics

```python
# src/star_e/portfolio/metrics.py

import numpy as np
from typing import Optional

def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Sharpe = (E[R] - Rf) / σ(R) * √(periods_per_year)
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    if np.std(returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    target_return: float = 0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio.
    
    Uses downside deviation instead of total standard deviation.
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    
    downside_returns = np.minimum(returns - target_return, 0)
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_std == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0
    
    return np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)


def information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio.
    
    IR = (Active Return) / (Tracking Error)
    """
    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns)
    
    if tracking_error == 0:
        return 0
    
    return np.mean(active_returns) / tracking_error * np.sqrt(periods_per_year)


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio.
    
    Calmar = Annualized Return / Max Drawdown
    """
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = abs(np.min(drawdowns))
    
    annualized_return = (cumulative[-1] ** (periods_per_year / len(returns))) - 1
    
    if max_dd == 0:
        return np.inf if annualized_return > 0 else 0
    
    return annualized_return / max_dd
```

---

## 11. Dashboard & API

### 11.1 Streamlit Dashboard

```python
# dashboard/app.py

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="StAR-E Dashboard",
    page_icon="📈",
    layout="wide"
)

# Sidebar
st.sidebar.title("StAR-E")
st.sidebar.markdown("Statistical Arbitrage & Risk Engine")

page = st.sidebar.radio(
    "Navigation",
    ["Portfolio Overview", "Regime Analysis", "Risk Dashboard", "Model Comparison"]
)

# Load data (cached)
@st.cache_data(ttl=3600)
def load_portfolio_data():
    # Load from API or database
    pass

if page == "Portfolio Overview":
    st.title("Portfolio Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", "+23.4%", "+2.1%")
    with col2:
        st.metric("Sharpe Ratio", "1.42", "+0.12")
    with col3:
        st.metric("Max Drawdown", "-8.2%", "-1.3%")
    with col4:
        st.metric("Current Regime", "Bull", delta=None)
    
    # Cumulative returns chart
    fig = go.Figure()
    # ... add traces
    st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio allocation
    st.subheader("Current Allocation")
    # ... allocation pie chart

elif page == "Regime Analysis":
    st.title("Market Regime Detection")
    
    # HMM state visualization
    # Transition matrix heatmap
    # Expected duration chart
    
elif page == "Risk Dashboard":
    st.title("Risk Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("95% VaR (Daily)", "2.3%")
        st.metric("95% CVaR (Daily)", "3.1%")
    
    with col2:
        st.metric("Current Volatility", "18.2%")
        st.metric("Volatility Forecast (21d)", "16.8%")
    
    # Risk gauges
    # Drawdown chart

elif page == "Model Comparison":
    st.title("Model Backtest Comparison")
    
    # SARIMA vs LSTM vs Ensemble comparison table
    # Performance over time chart
```

### 11.2 FastAPI Backend

```python
# src/star_e/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

app = FastAPI(
    title="StAR-E API",
    description="Statistical Arbitrage & Risk Engine",
    version="0.1.0"
)

class PortfolioRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=2, max_length=50)
    start_date: date
    end_date: date
    risk_tolerance: float = Field(default=0.5, ge=0, le=1)

class PortfolioResponse(BaseModel):
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    regime: str
    var_95: float
    cvar_95: float

class RegimeResponse(BaseModel):
    current_state: str
    state_probabilities: dict[str, float]
    expected_duration: dict[str, float]
    transition_matrix: List[List[float]]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/portfolio/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest):
    """Optimize portfolio weights based on current regime."""
    # Implementation
    pass

@app.get("/regime/current", response_model=RegimeResponse)
async def get_current_regime():
    """Get current market regime from HMM."""
    pass

@app.get("/risk/metrics")
async def get_risk_metrics(
    tickers: List[str],
    lookback_days: int = 252
):
    """Calculate risk metrics for given portfolio."""
    pass

@app.get("/forecast/{ticker}")
async def get_forecast(
    ticker: str,
    horizon: int = 21,
    model: str = "ensemble"
):
    """Get return forecast for a ticker."""
    pass
```

---

## 12. MLOps & Deployment

### 12.1 MLflow Integration

```python
# src/star_e/mlops/tracking.py

import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow(
    experiment_name: str = "star-e",
    tracking_uri: str = "mlruns"
):
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return MlflowClient()

def log_model_run(
    model_name: str,
    params: dict,
    metrics: dict,
    artifacts: dict = None
):
    """Log a model training run."""
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, name)
```

### 12.2 Drift Detection

```python
# src/star_e/mlops/drift.py

import numpy as np
from scipy import stats

def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10
) -> float:
    """
    Calculate Population Stability Index.
    
    PSI Interpretation:
    - < 0.1: No significant change
    - 0.1-0.25: Moderate shift, monitor
    - > 0.25: Significant shift, investigate/retrain
    """
    # Bin the distributions
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.clip(expected_percents, 0.0001, None)
    actual_percents = np.clip(actual_percents, 0.0001, None)
    
    psi = np.sum(
        (actual_percents - expected_percents) * 
        np.log(actual_percents / expected_percents)
    )
    
    return psi


def ks_test_drift(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.05
) -> dict:
    """
    Kolmogorov-Smirnov test for distribution drift.
    """
    statistic, pvalue = stats.ks_2samp(reference, current)
    
    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "is_drift": pvalue < threshold
    }
```

### 12.3 Docker Configuration

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source code
COPY src/ src/
COPY dashboard/ dashboard/

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["uvicorn", "star_e.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./mlruns:/app/mlruns
    environment:
      - MLFLOW_TRACKING_URI=/app/mlruns
    command: uvicorn star_e.api.main:app --host 0.0.0.0 --port 8000

  dashboard:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - API_URL=http://api:8000
    command: streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /mlruns
```

---

## 13. Phased Implementation Plan

### Phase 1: Foundation (Weeks 1-4)

**Goal**: Working end-to-end pipeline with basic models

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | - Set up project structure<br>- Initialize git, pyproject.toml<br>- yFinance data ingestion<br>- DuckDB storage layer | `src/star_e/data/` module, unit tests |
| 2 | - Feature engineering (returns, rolling stats)<br>- Data validation (Pydantic schemas)<br>- Stationarity tests (ADF/KPSS) | Feature pipeline, validation tests |
| 3 | - SARIMA implementation<br>- Basic GARCH(1,1)<br>- MLflow setup | Working forecasters, logged experiments |
| 4 | - Simple mean-variance optimizer<br>- Basic backtesting framework<br>- CLI interface (typer) | First backtest results, `star-e backtest` CLI |

**Phase 1 Milestone**: `star-e backtest --tickers AAPL,MSFT,GOOGL --start 2020-01-01` produces Sharpe ratio, returns, drawdown.

### Phase 2: Statistical Intelligence (Weeks 5-8)

**Goal**: Add regime awareness and improved forecasting

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 5 | - HMM implementation with hmmlearn<br>- State selection (AIC/BIC)<br>- Regime visualization | `RegimeHMM` class, regime overlays |
| 6 | - Regime-aware portfolio weighting<br>- Position sizing based on state<br>- Updated backtest with regimes | Improved risk-adjusted returns |
| 7 | - LSTM + attention implementation<br>- Training pipeline<br>- Apple Silicon optimization (MPS) | `LSTMForecaster` class |
| 8 | - Ensemble forecaster<br>- Fractional differentiation<br>- Johansen cointegration | Ensemble predictions, pairs list |

**Phase 2 Milestone**: HMM correctly identifies major regime shifts (COVID crash, 2022 bear market). Ensemble outperforms individual models.

### Phase 3: Production & Interface (Weeks 9-12)

**Goal**: Portfolio-ready presentation layer

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 9 | - FastAPI backend<br>- REST endpoints<br>- Pydantic schemas | `/portfolio`, `/regime`, `/risk` APIs |
| 10 | - Streamlit dashboard<br>- Performance visualizations<br>- Regime display | Interactive dashboard |
| 11 | - VaR/CVaR calculations<br>- Drift detection (PSI)<br>- Docker containerization | Risk metrics panel, Docker images |
| 12 | - Documentation<br>- Demo video<br>- GitHub Actions CI<br>- Deploy to Railway/Render | Live demo, README, CI pipeline |

**Phase 3 Milestone**: Publicly accessible demo at https://star-e-demo.railway.app with documentation.

---

## 14. Testing Strategy

### Unit Tests

```python
# tests/test_data.py

import pytest
import numpy as np
from star_e.data.features import compute_returns, fractional_diff, test_stationarity

def test_compute_returns():
    prices = np.array([100, 102, 101, 103, 105])
    returns = compute_returns(prices, period=1)
    
    assert len(returns) == len(prices)
    assert np.isnan(returns[0])
    assert np.isclose(returns[1], np.log(102/100))

def test_fractional_diff_preserves_length():
    series = np.random.randn(100)
    diff = fractional_diff(series, d=0.4)
    
    assert len(diff) == len(series)

def test_stationarity_on_random_walk():
    random_walk = np.cumsum(np.random.randn(500))
    result = test_stationarity(random_walk)
    
    assert result['adf_is_stationary'] == False  # Random walk is non-stationary
```

```python
# tests/test_models.py

import pytest
import numpy as np
from star_e.models.hmm import RegimeHMM

def test_hmm_fit_decode():
    np.random.seed(42)
    
    # Generate synthetic regime data
    n_samples = 500
    states = np.repeat([0, 1, 2, 1, 0], 100)
    features = np.column_stack([
        states * 0.02 + np.random.randn(n_samples) * 0.01,  # Return
        np.abs(states - 1) * 0.1 + np.random.randn(n_samples) * 0.02  # Vol
    ])
    
    hmm = RegimeHMM(n_states=3)
    hmm.fit(features)
    decoded_states, probs = hmm.decode(features)
    
    assert len(decoded_states) == n_samples
    assert probs.shape == (n_samples, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_hmm_transition_matrix_rows_sum_to_one():
    hmm = RegimeHMM(n_states=3)
    hmm.fit(np.random.randn(200, 2))
    
    trans = hmm.get_transition_matrix()
    
    assert trans.shape == (3, 3)
    assert np.allclose(trans.sum(axis=1), 1.0)
```

### Integration Tests

```python
# tests/test_integration.py

import pytest
from star_e.backtesting.engine import BacktestEngine

def test_full_backtest_pipeline():
    engine = BacktestEngine(
        tickers=["AAPL", "MSFT"],
        start_date="2022-01-01",
        end_date="2023-01-01"
    )
    
    results = engine.run()
    
    assert "sharpe_ratio" in results
    assert "max_drawdown" in results
    assert results["sharpe_ratio"] is not None
```

### Test Coverage Target

```bash
pytest --cov=star_e --cov-report=html tests/
# Target: >60% coverage
```

---

## 15. Project Structure

```
star-e/
├── README.md                           # Project overview, setup, demo link
├── pyproject.toml                      # Dependencies (uv/poetry)
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml                      # Lint, type-check, test on PR
├── data/
│   ├── raw/                            # Downloaded OHLCV data
│   └── processed/                      # Features, cleaned data
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_regime_detection.ipynb       # HMM experiments
│   ├── 03_cointegration.ipynb          # Pairs analysis
│   └── 04_backtesting.ipynb            # Strategy validation
├── src/
│   └── star_e/
│       ├── __init__.py
│       ├── config.py                   # Pydantic settings
│       ├── cli.py                      # Typer CLI
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingestion.py            # yFinance fetcher
│       │   ├── features.py             # Feature engineering
│       │   ├── validation.py           # Data quality checks
│       │   └── storage.py              # DuckDB operations
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py                 # Abstract interfaces
│       │   ├── sarima.py               # SARIMA forecaster
│       │   ├── lstm.py                 # LSTM with attention
│       │   ├── garch.py                # GARCH volatility
│       │   ├── hmm.py                  # Regime detection
│       │   └── ensemble.py             # Model combination
│       ├── portfolio/
│       │   ├── __init__.py
│       │   ├── optimizer.py            # Mean-variance
│       │   ├── cointegration.py        # Johansen test
│       │   ├── risk.py                 # VaR, CVaR, drawdown
│       │   └── metrics.py              # Sharpe, Sortino, IR
│       ├── backtesting/
│       │   ├── __init__.py
│       │   ├── engine.py               # Backtest runner
│       │   ├── walk_forward.py         # Walk-forward validation
│       │   └── report.py               # Results generation
│       ├── mlops/
│       │   ├── __init__.py
│       │   ├── tracking.py             # MLflow integration
│       │   └── drift.py                # PSI, KS tests
│       └── api/
│           ├── __init__.py
│           ├── main.py                 # FastAPI app
│           └── schemas.py              # Request/response models
├── dashboard/
│   └── app.py                          # Streamlit dashboard
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Fixtures
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_portfolio.py
│   └── test_integration.py
└── mlruns/                             # MLflow artifacts (gitignored)
```

---

## 16. Verification & Demo

### End-to-End Verification Checklist

| Step | Command | Expected Output |
|------|---------|-----------------|
| 1. Install | `pip install -e .` | No errors |
| 2. Ingest | `star-e ingest --tickers AAPL,MSFT,GOOGL --start 2020-01-01` | Data saved to DuckDB |
| 3. Train | `star-e train --model hmm` | MLflow run logged |
| 4. Backtest | `star-e backtest --strategy regime_aware` | Sharpe, returns, drawdown printed |
| 5. API | `uvicorn star_e.api.main:app` | Swagger UI at localhost:8000/docs |
| 6. Dashboard | `streamlit run dashboard/app.py` | Interactive dashboard |
| 7. Docker | `docker-compose up` | All services running |
| 8. Tests | `pytest tests/` | >60% coverage, all pass |

### Demo Script (2-minute walkthrough)

1. **Open Dashboard** → Show portfolio performance chart
2. **Navigate to Regime Analysis** → Point out Bull/Bear/Sideways states
3. **Show Risk Metrics** → Explain VaR/CVaR in context
4. **API Documentation** → Quick Swagger UI tour
5. **MLflow UI** → Show experiment comparison
6. **Terminal** → Run `star-e backtest` to show CLI

---

## 17. Interview Talking Points

### Technical Depth Questions

**Q: Why Hidden Markov Models for regime detection?**
> HMMs are ideal because markets exhibit latent states (Bull/Bear/Sideways) that aren't directly observable but influence returns and volatility. The Markov assumption (next state depends only on current state) is reasonable for market regimes. I used the Viterbi algorithm for most-likely state sequence and AIC/BIC for model selection.

**Q: How does fractional differentiation preserve memory?**
> Standard differencing (d=1) removes all serial correlation, destroying predictive signal. Fractional differentiation with d≈0.4 achieves stationarity while preserving long-term dependencies. I implemented López de Prado's weight truncation method for computational efficiency.

**Q: Why combine SARIMA and LSTM?**
> SARIMA excels at capturing linear autoregressive patterns and seasonality, while LSTM captures non-linear dependencies. By ensembling them with regime-aware weights (favoring SARIMA in sideways markets, LSTM in trending markets), we get robust predictions across market conditions.

**Q: How do you handle fat-tailed return distributions?**
> I use Student's t-distribution in GARCH (rather than Normal), Cornish-Fisher expansion for VaR to account for skewness/kurtosis, and CVaR (Expected Shortfall) as the primary risk metric since it's coherent and captures tail risk better than VaR.

### Architecture & Decisions

**Q: Why Streamlit instead of React?**
> For a data science portfolio project, Streamlit provides 10x faster development while adequately demonstrating the system. React would take 3+ weeks of frontend development time better spent on the statistical models. The dashboard serves as a demonstration interface, not a production trading platform.

**Q: How would you scale this for production?**
> Current architecture handles portfolio-scale data. For production: (1) Replace DuckDB with TimescaleDB for real-time ingestion, (2) Add Redis for caching regime states, (3) Deploy models via TorchServe, (4) Add Kubernetes for horizontal scaling, (5) Implement proper event sourcing for audit trails.

---

## Critical Files to Create

1. **`src/star_e/data/ingestion.py`** - Foundation; everything depends on clean data
2. **`src/star_e/models/hmm.py`** - Project's key differentiator
3. **`src/star_e/portfolio/optimizer.py`** - Core portfolio construction
4. **`src/star_e/backtesting/engine.py`** - Validates all claims
5. **`dashboard/app.py`** - Primary presentation layer

---

## Verification Plan

1. **Data Pipeline**: Run ingestion for S&P 500 subset, verify no NaN values, check date coverage
2. **Regime Detection**: Train HMM, verify it identifies COVID crash (Mar 2020) as Bear regime
3. **Forecasters**: Compare SARIMA vs LSTM vs Ensemble MAE on held-out test set
4. **Portfolio**: Verify regime-aware portfolio has better Sharpe than static allocation
5. **Risk**: Validate VaR by checking historical violation rate ≈ 5%
6. **Dashboard**: Manual testing of all interactive elements
7. **CI Pipeline**: All tests pass, coverage >60%, no linting errors
