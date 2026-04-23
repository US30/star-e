"""MLflow experiment tracking utilities."""

from pathlib import Path
from typing import Optional, Any

import mlflow
from mlflow.tracking import MlflowClient

from star_e.config import settings


def setup_mlflow(
    experiment_name: str = "star-e",
    tracking_uri: Optional[str] = None,
) -> MlflowClient:
    """
    Initialize MLflow tracking.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI (default: local mlruns)

    Returns:
        MlflowClient instance
    """
    uri = tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)

    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)

    return MlflowClient()


def log_model_run(
    model_name: str,
    params: dict,
    metrics: dict,
    artifacts: Optional[dict[str, str]] = None,
    tags: Optional[dict[str, str]] = None,
    model: Optional[Any] = None,
) -> str:
    """
    Log a model training run to MLflow.

    Args:
        model_name: Name/type of the model
        params: Hyperparameters
        metrics: Performance metrics
        artifacts: Dict mapping artifact names to file paths
        tags: Additional tags
        model: Model object to log (optional)

    Returns:
        Run ID
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log artifacts
        if artifacts:
            for name, path in artifacts.items():
                if Path(path).exists():
                    mlflow.log_artifact(path, name)

        # Log model
        if model is not None:
            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception:
                pass  # Model type not supported

        return run.info.run_id


def log_backtest_results(
    strategy_name: str,
    result: Any,  # BacktestResult
    additional_params: Optional[dict] = None,
) -> str:
    """
    Log backtest results to MLflow.

    Args:
        strategy_name: Name of the strategy
        result: BacktestResult object
        additional_params: Extra parameters to log

    Returns:
        Run ID
    """
    params = {
        "strategy": strategy_name,
        "n_trades": result.n_trades,
    }
    if additional_params:
        params.update(additional_params)

    metrics = {
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "volatility": result.volatility,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "max_drawdown": result.max_drawdown,
        "calmar_ratio": result.calmar_ratio,
        "var_95": result.var_95,
        "cvar_95": result.cvar_95,
        "turnover": result.turnover,
    }

    tags = {
        "type": "backtest",
    }
    if result.start_date:
        tags["start_date"] = result.start_date.strftime("%Y-%m-%d")
    if result.end_date:
        tags["end_date"] = result.end_date.strftime("%Y-%m-%d")

    return log_model_run(
        model_name=f"backtest_{strategy_name}",
        params=params,
        metrics=metrics,
        tags=tags,
    )


def get_best_run(
    experiment_name: str,
    metric: str = "sharpe_ratio",
    ascending: bool = False,
) -> Optional[dict]:
    """
    Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize
        ascending: If True, lower is better

    Returns:
        Dict with run info or None if no runs
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return None

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        return None

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "tags": run.data.tags,
    }


def compare_runs(
    experiment_name: str,
    metrics: list[str],
    max_runs: int = 10,
) -> list[dict]:
    """
    Compare recent runs in an experiment.

    Args:
        experiment_name: Name of the experiment
        metrics: List of metrics to compare
        max_runs: Maximum number of runs to return

    Returns:
        List of run summaries
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_runs,
    )

    summaries = []
    for run in runs:
        summary = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
        }
        for metric in metrics:
            summary[metric] = run.data.metrics.get(metric)
        summaries.append(summary)

    return summaries
