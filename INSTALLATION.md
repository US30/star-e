# Installation Guide

This guide covers various installation methods for StAR-E.

## Prerequisites

- **Python**: 3.11 or 3.12
- **Node.js**: 20+ (for React frontend)
- **Docker**: Latest version (optional, for containerized deployment)
- **Git**: For cloning the repository

## Method 1: Local Installation (Recommended for Development)

### Step 1: Clone the Repository

```bash
git clone https://github.com/US30/star-e.git
cd star-e
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
# Install the package with all dependencies
pip install -e ".[dev]"

# This installs:
# - Core dependencies (yfinance, pandas, torch, etc.)
# - Development dependencies (pytest, ruff, mypy)
# - All optional dependencies
```

### Step 4: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 5: Set Up Configuration (Optional)

Create a `.env` file in the project root:

```bash
# Data settings
STAR_E_DATA_DIR=./data
STAR_E_MLFLOW_TRACKING_URI=mlruns

# Binance API (optional - only if using crypto features)
STAR_E_BINANCE_API_KEY=your_api_key_here
STAR_E_BINANCE_API_SECRET=your_api_secret_here
STAR_E_BINANCE_TESTNET=true

# Model settings
STAR_E_HMM_N_STATES=3
STAR_E_TFT_HIDDEN_SIZE=64
STAR_E_GAT_EMBEDDING_DIM=32

# Risk settings
STAR_E_VAR_CONFIDENCE=0.95
STAR_E_MONTE_CARLO_SIMULATIONS=10000
```

### Step 6: Verify Installation

```bash
# Check Python package
star-e --help

# Start API server
star-e serve --port 8000
# Visit http://localhost:8000/docs

# In another terminal, start frontend
cd frontend
npm start
# Visit http://localhost:3000
```

## Method 2: Docker Installation (Recommended for Production)

### Step 1: Clone and Build

```bash
git clone https://github.com/US30/star-e.git
cd star-e
```

### Step 2: Configure Environment

Create `.env` file with your settings (see Step 5 above).

### Step 3: Build and Run

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 4: Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **React Dashboard**: http://localhost:3000
- **Streamlit Dashboard**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

### Stopping Services

```bash
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v
```

## Method 3: Minimal Installation (CLI Only)

For users who only need the CLI tools:

```bash
git clone https://github.com/US30/star-e.git
cd star-e

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install core dependencies only
pip install -e .

# Verify
star-e --help
```

## GPU Support (Optional)

### NVIDIA CUDA

For NVIDIA GPUs:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Apple Silicon (M1/M2/M3)

PyTorch automatically detects and uses Apple's MPS (Metal Performance Shaders):

```bash
# No additional installation needed
# The code automatically detects MPS availability
```

Verify GPU detection:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

## Dependency Installation Details

### Core Dependencies

```bash
# Data engineering
yfinance>=0.2.40          # Stock market data
python-binance>=1.0.19    # Cryptocurrency data
duckdb>=1.0.0             # Fast analytical database
polars>=0.20.0            # High-performance DataFrame library

# Signal processing
filterpy>=1.4.5           # Kalman filters

# Statistical models
statsmodels>=0.14.0       # SARIMA, GARCH, cointegration
hmmlearn>=0.3.0           # Hidden Markov Models
arch>=6.0.0               # Advanced GARCH models

# Machine learning
torch>=2.2.0              # Deep learning
scikit-learn>=1.4.0       # Classical ML
pytorch-forecasting>=1.0.0  # TFT
torch-geometric>=2.5.0    # Graph neural networks

# Visualization
plotly>=5.20.0            # Interactive plots
matplotlib>=3.8.0         # Static plots
d3>=7.8.5                 # D3.js (frontend)

# API & Dashboard
fastapi>=0.110.0          # REST API
streamlit>=1.32.0         # Python dashboard
react>=18.2.0             # Frontend framework

# MLOps
mlflow>=2.11.0            # Experiment tracking
```

## Troubleshooting

### Issue: `torch-geometric` installation fails

**Solution:**
```bash
# Install dependencies first
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-geometric
```

### Issue: `filterpy` not found

**Solution:**
```bash
pip install filterpy
```

### Issue: Frontend build fails

**Solution:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Issue: Permission denied on Docker

**Solution:**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Restart Docker Desktop (macOS/Windows)
```

### Issue: Port already in use

**Solution:**
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
star-e serve --port 8001
```

## Platform-Specific Notes

### macOS

- Use Homebrew to install Python: `brew install python@3.11`
- Install Command Line Tools: `xcode-select --install`

### Windows

- Install Python from python.org
- Use PowerShell or Git Bash
- May need to install Microsoft C++ Build Tools for some packages

### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Fedora/RHEL
sudo dnf install python3.11 python3-pip
```

## Next Steps

After installation:

1. **Try the Quick Start**: See README.md
2. **Run Tests**: `pytest tests/ -v`
3. **Explore Examples**: Check `notebooks/` directory
4. **Read Documentation**: Visit the wiki

## Getting Help

- **Issues**: https://github.com/US30/star-e/issues
- **Discussions**: https://github.com/US30/star-e/discussions
- **Email**: sinha.utkarshsinha30@gmail.com
