"""Statistical and ML models for StAR-E."""

from star_e.models.hmm import RegimeHMM
from star_e.models.gmm import RegimeGMM, BayesianRegimeGMM, EnsembleRegimeDetector
from star_e.models.sarima import SARIMAForecaster
from star_e.models.lstm import (
    LSTMForecaster,
    AttentionLSTM,
    SharpeLoss,
    SortinoLoss,
    CombinedRiskLoss,
)
from star_e.models.garch import GARCHModel
from star_e.models.ensemble import EnsembleForecaster
from star_e.models.tft import TFTForecaster, SimpleTFT, TFTTrainer

__all__ = [
    # Regime Detection
    "RegimeHMM",
    "RegimeGMM",
    "BayesianRegimeGMM",
    "EnsembleRegimeDetector",
    # Forecasters
    "SARIMAForecaster",
    "LSTMForecaster",
    "AttentionLSTM",
    "TFTForecaster",
    "SimpleTFT",
    "TFTTrainer",
    "GARCHModel",
    "EnsembleForecaster",
    # Loss Functions
    "SharpeLoss",
    "SortinoLoss",
    "CombinedRiskLoss",
]
