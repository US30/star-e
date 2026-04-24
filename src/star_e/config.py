"""Configuration settings for StAR-E."""

from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="STAR_E_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    mlflow_tracking_uri: str = "mlruns"

    # Binance API settings
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True

    # Data settings
    default_tickers: list[str] | str = Field(
        default=[
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "V", "JNJ",
            "XLK", "XLF", "XLE", "XLV", "XLI",
            "SPY", "QQQ", "IWM",
        ]
    )
    default_crypto_pairs: list[str] | str = Field(
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    )
    default_start_date: str = "2018-01-01"
    risk_free_rate: float = 0.04

    # Model settings - HMM
    hmm_n_states: int = 3
    hmm_covariance_type: Literal["full", "diag", "spherical", "tied"] = "full"

    # Model settings - GMM
    gmm_n_components: int = 3
    gmm_covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"

    # Model settings - LSTM
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_sequence_length: int = 21

    # Model settings - TFT
    tft_hidden_size: int = 64
    tft_attention_head_size: int = 4
    tft_max_encoder_length: int = 60
    tft_max_prediction_length: int = 21

    # Model settings - GAT
    gat_hidden_dim: int = 64
    gat_embedding_dim: int = 32
    gat_num_heads: int = 4

    # Model settings - GARCH
    garch_p: int = 1
    garch_q: int = 1

    # Kalman Filter settings
    kalman_process_noise: float = 0.01
    kalman_observation_noise: float = 0.1

    # Portfolio settings
    min_weight: float = 0.0
    max_weight: float = 0.3

    # Risk settings
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95
    monte_carlo_simulations: int = 10000

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Dashboard settings
    dashboard_port: int = 8501

    @field_validator("default_tickers", "default_crypto_pairs", mode="before")
    @classmethod
    def parse_csv(cls, v: Any):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v


def get_device() -> torch.device:
    """Select best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


settings = Settings()
