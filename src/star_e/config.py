"""Configuration settings for StAR-E."""

from pathlib import Path
from typing import Literal

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="STAR_E_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    mlflow_tracking_uri: str = "mlruns"

    # Data settings
    default_tickers: list[str] = Field(
        default=[
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "V", "JNJ",
            "XLK", "XLF", "XLE", "XLV", "XLI",
            "SPY", "QQQ", "IWM",
        ]
    )
    default_start_date: str = "2018-01-01"
    risk_free_rate: float = 0.04

    # Model settings
    hmm_n_states: int = 3
    hmm_covariance_type: Literal["full", "diag", "spherical", "tied"] = "full"
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_sequence_length: int = 21
    garch_p: int = 1
    garch_q: int = 1

    # Portfolio settings
    min_weight: float = 0.0
    max_weight: float = 0.3

    # Risk settings
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Dashboard settings
    dashboard_port: int = 8501


def get_device() -> torch.device:
    """Select best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


settings = Settings()
