"""Command-line interface for StAR-E."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="star-e",
    help="StAR-E: Statistical Arbitrage & Risk Engine",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    tickers: str = typer.Option(
        "AAPL,MSFT,GOOGL,AMZN,META",
        "--tickers", "-t",
        help="Comma-separated list of tickers",
    ),
    start_date: str = typer.Option(
        "2020-01-01",
        "--start", "-s",
        help="Start date (YYYY-MM-DD)",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end", "-e",
        help="End date (YYYY-MM-DD), defaults to today",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for CSV (optional)",
    ),
):
    """Ingest price data from yFinance."""
    from star_e.data import fetch_tickers, save_to_duckdb
    from star_e.data.features import compute_returns, compute_rolling_stats

    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Fetching data from yFinance...", total=None)
        df = fetch_tickers(ticker_list, start_date, end_date)

    console.print(f"[green]Fetched {len(df)} rows for {len(ticker_list)} tickers[/green]")

    # Compute features
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Computing features...", total=None)
        df = compute_returns(df)
        df = compute_rolling_stats(df)

    # Save
    save_to_duckdb(df, "prices")
    console.print("[green]Data saved to DuckDB[/green]")

    if output:
        df.write_csv(output)
        console.print(f"[green]Data exported to {output}[/green]")


@app.command()
def train(
    model: str = typer.Option(
        "hmm",
        "--model", "-m",
        help="Model to train: hmm, sarima, lstm, garch, ensemble",
    ),
    tickers: str = typer.Option(
        "AAPL,MSFT,GOOGL",
        "--tickers", "-t",
        help="Comma-separated list of tickers",
    ),
):
    """Train a model on historical data."""
    import numpy as np
    from star_e.data import load_from_duckdb
    from star_e.mlops import setup_mlflow
    import mlflow

    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    console.print(f"[blue]Training {model} model...[/blue]")

    # Load data
    df = load_from_duckdb("prices")
    df = df.filter(df["ticker"].is_in(ticker_list))

    setup_mlflow()

    if model == "hmm":
        from star_e.models import RegimeHMM

        # Prepare features (returns and volatility)
        features = []
        for ticker in ticker_list:
            ticker_df = df.filter(df["ticker"] == ticker).sort("date")
            returns = ticker_df["return_1d"].drop_nulls().to_numpy()
            if len(returns) > 0:
                features.append(returns)

        if not features:
            console.print("[red]No data available for training[/red]")
            raise typer.Exit(1)

        # Use first ticker's features for now
        X = np.column_stack([
            features[0],
            np.abs(features[0]),  # Volatility proxy
        ])
        X = X[~np.isnan(X).any(axis=1)]

        with mlflow.start_run(run_name="hmm_training"):
            hmm = RegimeHMM(n_states=3)
            hmm.fit(X)

            states, probs = hmm.decode(X)
            durations = hmm.expected_duration()

            console.print("\n[green]HMM Training Complete![/green]")
            console.print(f"States detected: {len(np.unique(states))}")
            console.print(f"Expected durations: {durations}")

    elif model == "garch":
        from star_e.models import GARCHModel

        ticker_df = df.filter(df["ticker"] == ticker_list[0]).sort("date")
        returns = ticker_df["return_1d"].drop_nulls().to_numpy()

        with mlflow.start_run(run_name="garch_training"):
            garch = GARCHModel(p=1, q=1)
            garch.fit(returns)

            console.print("\n[green]GARCH Training Complete![/green]")
            console.print(f"Persistence: {garch.persistence():.4f}")
            console.print(f"Unconditional Vol: {garch.unconditional_volatility():.4%}")

    else:
        console.print(f"[red]Model '{model}' not implemented in CLI yet[/red]")
        raise typer.Exit(1)


@app.command()
def backtest(
    strategy: str = typer.Option(
        "equal_weight",
        "--strategy", "-s",
        help="Strategy: equal_weight, min_variance, max_sharpe, regime_aware",
    ),
    tickers: str = typer.Option(
        "AAPL,MSFT,GOOGL,AMZN,META",
        "--tickers", "-t",
        help="Comma-separated list of tickers",
    ),
    start_date: str = typer.Option(
        "2020-01-01",
        "--start",
        help="Backtest start date",
    ),
    rebalance: str = typer.Option(
        "monthly",
        "--rebalance", "-r",
        help="Rebalancing frequency: daily, weekly, monthly, quarterly",
    ),
):
    """Run a backtest on historical data."""
    import numpy as np
    import pandas as pd
    from star_e.data import load_from_duckdb
    from star_e.backtesting import BacktestEngine
    from star_e.backtesting.engine import print_backtest_report
    from star_e.portfolio import PortfolioOptimizer

    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    console.print(f"[blue]Running {strategy} backtest...[/blue]")

    # Load data
    df = load_from_duckdb("prices")
    df = df.filter(df["ticker"].is_in(ticker_list))
    df = df.filter(df["date"] >= start_date)

    # Pivot to get returns matrix
    pivot_df = df.pivot(index="date", columns="ticker", values="return_1d")
    pivot_df = pivot_df.drop_nulls()

    dates = pd.DatetimeIndex(pivot_df["date"].to_list())
    returns = pivot_df.select(ticker_list).to_numpy()

    # Define strategy
    optimizer = PortfolioOptimizer()
    n_assets = len(ticker_list)

    if strategy == "equal_weight":
        def weight_func(returns_history, t):
            return np.ones(n_assets) / n_assets

    elif strategy == "min_variance":
        def weight_func(returns_history, t):
            if len(returns_history) < 60:
                return np.ones(n_assets) / n_assets
            cov = np.cov(returns_history[-252:].T)
            exp_ret = np.mean(returns_history[-63:], axis=0)
            result = optimizer.optimize(exp_ret, cov, method="min_variance")
            return result["weights"]

    elif strategy == "max_sharpe":
        def weight_func(returns_history, t):
            if len(returns_history) < 60:
                return np.ones(n_assets) / n_assets
            cov = np.cov(returns_history[-252:].T)
            exp_ret = np.mean(returns_history[-63:], axis=0)
            result = optimizer.optimize(exp_ret, cov, method="max_sharpe")
            return result["weights"]

    else:
        console.print(f"[red]Unknown strategy: {strategy}[/red]")
        raise typer.Exit(1)

    # Run backtest
    engine = BacktestEngine(rebalance_frequency=rebalance)
    result = engine.run(returns, weight_func, dates, lookback=63)

    print_backtest_report(result, strategy.replace("_", " ").title())


@app.command()
def regime(
    ticker: str = typer.Option(
        "SPY",
        "--ticker", "-t",
        help="Ticker to analyze",
    ),
):
    """Detect current market regime using HMM."""
    import numpy as np
    from star_e.data import load_from_duckdb
    from star_e.models import RegimeHMM

    console.print(f"[blue]Analyzing regime for {ticker}...[/blue]")

    df = load_from_duckdb("prices")
    df = df.filter(df["ticker"] == ticker).sort("date")

    returns = df["return_1d"].drop_nulls().to_numpy()

    if len(returns) < 100:
        console.print("[red]Insufficient data for regime detection[/red]")
        raise typer.Exit(1)

    X = np.column_stack([returns, np.abs(returns)])

    hmm = RegimeHMM(n_states=3)
    hmm.fit(X)

    states, probs = hmm.decode(X)
    current_state = states[-1]
    current_probs = probs[-1]

    state_names = ["Bear", "Sideways", "Bull"]

    table = Table(title=f"Regime Analysis for {ticker}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Current Regime", state_names[current_state])
    table.add_row("Bear Probability", f"{current_probs[0]:.1%}")
    table.add_row("Sideways Probability", f"{current_probs[1]:.1%}")
    table.add_row("Bull Probability", f"{current_probs[2]:.1%}")

    durations = hmm.expected_duration()
    for state, dur in durations.items():
        table.add_row(f"Expected {state} Duration", f"{dur:.1f} days")

    console.print(table)


@app.command()
def risk(
    tickers: str = typer.Option(
        "AAPL,MSFT,GOOGL",
        "--tickers", "-t",
        help="Comma-separated list of tickers",
    ),
    confidence: float = typer.Option(
        0.95,
        "--confidence", "-c",
        help="Confidence level for VaR/CVaR",
    ),
):
    """Calculate risk metrics for a portfolio."""
    import numpy as np
    from star_e.data import load_from_duckdb
    from star_e.portfolio.risk import calculate_var, calculate_cvar, max_drawdown

    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    df = load_from_duckdb("prices")
    df = df.filter(df["ticker"].is_in(ticker_list))

    # Equal-weighted portfolio returns
    pivot_df = df.pivot(index="date", columns="ticker", values="return_1d")
    pivot_df = pivot_df.drop_nulls()

    returns_matrix = pivot_df.select(ticker_list).to_numpy()
    weights = np.ones(len(ticker_list)) / len(ticker_list)
    portfolio_returns = returns_matrix @ weights

    var = calculate_var(portfolio_returns, confidence)
    cvar = calculate_cvar(portfolio_returns, confidence)
    dd = max_drawdown(portfolio_returns)

    table = Table(title="Portfolio Risk Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row(f"VaR ({confidence:.0%})", f"{var:.2%}")
    table.add_row(f"CVaR ({confidence:.0%})", f"{cvar:.2%}")
    table.add_row("Max Drawdown", f"{dd['max_drawdown']:.2%}")
    table.add_row("Drawdown Duration", f"{dd['duration']} days")
    table.add_row("Volatility (annualized)", f"{np.std(portfolio_returns) * np.sqrt(252):.2%}")

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
):
    """Start the FastAPI server."""
    import uvicorn

    console.print(f"[green]Starting API server at http://{host}:{port}[/green]")
    uvicorn.run("star_e.api.main:app", host=host, port=port, reload=True)


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Port for Streamlit"),
):
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys

    dashboard_path = Path(__file__).parent.parent.parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        console.print("[red]Dashboard not found. Creating placeholder...[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Starting dashboard at http://localhost:{port}[/green]")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
    ])


@app.command()
def version():
    """Show version information."""
    from star_e import __version__

    console.print(f"StAR-E version {__version__}")


if __name__ == "__main__":
    app()
