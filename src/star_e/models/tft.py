"""
Temporal Fusion Transformer for multi-horizon time series forecasting.

TFT combines high-performance multi-horizon forecasting with interpretable
insights into temporal dynamics via attention mechanisms.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
    QuantileLoss,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE
from typing import Optional, Dict, List, Tuple
import mlflow


class TFTForecaster:
    """
    Temporal Fusion Transformer wrapper for financial forecasting.

    TFT provides:
    - Multi-horizon probabilistic forecasting
    - Variable selection networks
    - Temporal attention for interpretability
    - Static covariate handling

    Attributes:
        max_encoder_length: Number of historical time steps
        max_prediction_length: Forecast horizon
        hidden_size: Size of hidden layers
        attention_head_size: Size of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        max_encoder_length: int = 60,
        max_prediction_length: int = 21,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        hidden_continuous_size: int = 32,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 100,
        device: Optional[str] = None,
    ):
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.hidden_continuous_size = hidden_continuous_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model: Optional[TemporalFusionTransformer] = None
        self.training_dataset: Optional[TimeSeriesDataSet] = None
        self.trainer: Optional[L.Trainer] = None

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "return",
        time_idx_col: str = "time_idx",
        group_col: str = "ticker",
        known_reals: List[str] = None,
        unknown_reals: List[str] = None,
        static_categoricals: List[str] = None,
    ) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Prepare data for TFT training.

        Args:
            df: DataFrame with time series data
            target_col: Column to forecast
            time_idx_col: Time index column
            group_col: Group identifier column
            known_reals: Known future real variables
            unknown_reals: Unknown future real variables
            static_categoricals: Static categorical variables

        Returns:
            Training and validation datasets
        """
        known_reals = known_reals or []
        unknown_reals = unknown_reals or ["volatility", "momentum"]
        static_categoricals = static_categoricals or [group_col]

        max_time_idx = df[time_idx_col].max()
        training_cutoff = max_time_idx - self.max_prediction_length

        self.training_dataset = TimeSeriesDataSet(
            df[df[time_idx_col] <= training_cutoff],
            time_idx=time_idx_col,
            target=target_col,
            group_ids=[group_col],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=static_categoricals,
            time_varying_known_reals=known_reals + [time_idx_col],
            time_varying_unknown_reals=[target_col] + unknown_reals,
            target_normalizer=GroupNormalizer(groups=[group_col]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df,
            predict=True,
            stop_randomization=True,
        )

        return self.training_dataset, validation_dataset

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "return",
        time_idx_col: str = "time_idx",
        group_col: str = "ticker",
        known_reals: List[str] = None,
        unknown_reals: List[str] = None,
        val_split: float = 0.2,
    ) -> "TFTForecaster":
        """
        Train TFT model.

        Args:
            df: Training DataFrame
            target_col: Target column
            time_idx_col: Time index column
            group_col: Group column
            known_reals: Known future variables
            unknown_reals: Unknown future variables
            val_split: Validation split ratio

        Returns:
            Self for chaining
        """
        training_dataset, validation_dataset = self._prepare_data(
            df, target_col, time_idx_col, group_col, known_reals, unknown_reals
        )

        train_dataloader = training_dataset.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )
        val_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )

        self.model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            hidden_continuous_size=self.hidden_continuous_size,
            dropout=self.dropout,
            output_size=7,
            loss=QuantileLoss(),
            learning_rate=self.learning_rate,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
            ),
        ]

        self.trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.device if self.device != "mps" else "cpu",
            callbacks=callbacks,
            enable_progress_bar=True,
            gradient_clip_val=0.1,
        )

        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        mlflow.log_params({
            "tft_hidden_size": self.hidden_size,
            "tft_attention_heads": self.attention_head_size,
            "tft_encoder_length": self.max_encoder_length,
            "tft_prediction_length": self.max_prediction_length,
            "tft_dropout": self.dropout,
        })

        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            self.model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        return self

    def predict(
        self,
        df: pd.DataFrame,
        return_x: bool = False,
    ) -> Dict:
        """
        Generate forecasts.

        Args:
            df: DataFrame with input features
            return_x: Whether to return input data

        Returns:
            Dictionary with predictions and metadata
        """
        if self.model is None or self.training_dataset is None:
            raise ValueError("Model must be fitted before prediction")

        prediction_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df,
            predict=True,
            stop_randomization=True,
        )

        prediction_dataloader = prediction_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )

        predictions = self.model.predict(
            prediction_dataloader,
            mode="prediction",
            return_x=return_x,
        )

        return {
            "predictions": predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions,
        }

    def get_attention_weights(self, df: pd.DataFrame) -> Dict:
        """
        Get attention weights for interpretability.

        Returns:
            Dictionary with attention weights
        """
        if self.model is None or self.training_dataset is None:
            raise ValueError("Model must be fitted")

        prediction_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df,
            predict=True,
            stop_randomization=True,
        )

        prediction_dataloader = prediction_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )

        interpretation = self.model.interpret_output(
            self.model.predict(prediction_dataloader, mode="raw", return_x=True)
        )

        return {
            "attention": interpretation["attention"],
            "static_variables": interpretation.get("static_variables"),
            "encoder_variables": interpretation.get("encoder_variables"),
            "decoder_variables": interpretation.get("decoder_variables"),
        }

    def get_variable_importance(self) -> pd.DataFrame:
        """Get variable importance from trained model."""
        if self.model is None:
            raise ValueError("Model must be fitted")

        interpretation = self.model.interpret_output(
            self.model.predict(
                self.training_dataset.to_dataloader(train=False, batch_size=64),
                mode="raw",
                return_x=True,
            )
        )

        importance = pd.DataFrame({
            "variable": list(interpretation.get("encoder_variables", {}).keys()),
            "importance": list(interpretation.get("encoder_variables", {}).values()),
        })

        return importance.sort_values("importance", ascending=False)


class SimpleTFT(nn.Module):
    """
    Simplified TFT implementation for educational purposes.

    A lightweight version that demonstrates key TFT concepts:
    - Variable selection
    - Gated residual networks
    - Multi-head attention
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_horizon: int = 21,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forecast_horizon = forecast_horizon

        self.input_embedding = nn.Linear(input_size, hidden_size)

        self.variable_selection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1),
        )

        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GLU(dim=-1),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_size)

        Returns:
            Predictions (batch, forecast_horizon)
        """
        var_weights = self.variable_selection(
            self.input_embedding(x).mean(dim=1)
        )
        x_weighted = x * var_weights.unsqueeze(1)

        x_embed = self.input_embedding(x_weighted)

        lstm_out, (h_n, _) = self.encoder(x_embed)

        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        gated = self.gate(torch.cat([lstm_out, attn_out], dim=-1))

        residual = self.layer_norm(gated + lstm_out)

        last_hidden = residual[:, -1, :]

        return self.output_layer(last_hidden)


class TFTTrainer:
    """Training wrapper for SimpleTFT."""

    def __init__(
        self,
        model: SimpleTFT,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x.to(self.device))
        return predictions.cpu().numpy()
