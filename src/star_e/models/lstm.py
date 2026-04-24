"""LSTM model with attention for time series forecasting."""

from typing import Optional, Literal

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from star_e.config import get_device
from star_e.models.base import BaseForecaster


class SharpeLoss(nn.Module):
    """
    Sharpe Ratio based loss function.

    Maximizes the Sharpe ratio of predicted returns.
    Loss = -Sharpe = -(mean(returns) / std(returns))
    """

    def __init__(self, risk_free_rate: float = 0.0, eps: float = 1e-8):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.eps = eps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative Sharpe ratio.

        Args:
            predictions: Predicted returns
            targets: Actual returns (used to compute realized PnL)

        Returns:
            Negative Sharpe ratio (for minimization)
        """
        pnl = predictions * targets

        excess_returns = pnl - self.risk_free_rate
        mean_return = torch.mean(excess_returns)
        std_return = torch.std(excess_returns) + self.eps

        sharpe = mean_return / std_return

        return -sharpe


class SortinoLoss(nn.Module):
    """
    Sortino Ratio based loss function.

    Similar to Sharpe but uses downside deviation instead of
    standard deviation, penalizing only negative returns.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.eps = eps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative Sortino ratio.

        Args:
            predictions: Predicted returns
            targets: Actual returns

        Returns:
            Negative Sortino ratio (for minimization)
        """
        pnl = predictions * targets

        excess_returns = pnl - self.risk_free_rate
        mean_return = torch.mean(excess_returns)

        downside = torch.clamp(pnl - self.target_return, max=0)
        downside_deviation = torch.sqrt(torch.mean(downside ** 2) + self.eps)

        sortino = mean_return / downside_deviation

        return -sortino


class CombinedRiskLoss(nn.Module):
    """
    Combined loss using both Sharpe and Sortino ratios.

    Loss = alpha * (-Sharpe) + (1-alpha) * (-Sortino) + beta * MSE
    """

    def __init__(
        self,
        sharpe_weight: float = 0.3,
        sortino_weight: float = 0.3,
        mse_weight: float = 0.4,
        risk_free_rate: float = 0.0,
    ):
        super().__init__()
        self.sharpe_weight = sharpe_weight
        self.sortino_weight = sortino_weight
        self.mse_weight = mse_weight

        self.sharpe_loss = SharpeLoss(risk_free_rate)
        self.sortino_loss = SortinoLoss(risk_free_rate)
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss."""
        sharpe = self.sharpe_loss(predictions, targets)
        sortino = self.sortino_loss(predictions, targets)
        mse = self.mse_loss(predictions, targets)

        return (
            self.sharpe_weight * sharpe +
            self.sortino_weight * sortino +
            self.mse_weight * mse
        )


class AttentionLSTM(nn.Module):
    """
    LSTM with self-attention mechanism for sequence modeling.

    Architecture:
    1. Multi-layer LSTM processes the sequence
    2. Multi-head self-attention weighs different time steps
    3. Fully connected layers produce the forecast
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        attention_heads: int = 4,
    ):
        """
        Initialize AttentionLSTM.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            attention_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, sequence_length, input_size) input tensor

        Returns:
            (batch_size, 1) predictions
        """
        lstm_out, _ = self.lstm(x)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(lstm_out + attn_out)

        last_hidden = attn_out[:, -1, :]

        return self.fc(last_hidden)


class LSTMForecaster(BaseForecaster):
    """
    LSTM-based forecaster with attention mechanism.

    Captures non-linear patterns in time series that linear models
    like SARIMA cannot model.

    Features:
    - Automatic sequence creation from time series
    - Early stopping with validation loss
    - Apple Silicon (MPS) optimization
    - MLflow experiment tracking
    """

    def __init__(
        self,
        sequence_length: int = 21,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        attention_heads: int = 4,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        loss_type: Literal["mse", "sharpe", "sortino", "combined"] = "mse",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize LSTMForecaster.

        Args:
            sequence_length: Number of time steps in input sequence
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            attention_heads: Number of attention heads
            learning_rate: Adam learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            patience: Early stopping patience
            loss_type: Loss function type:
                - "mse": Mean Squared Error
                - "sharpe": Sharpe Ratio based loss
                - "sortino": Sortino Ratio based loss
                - "combined": Weighted combination of all
            device: Torch device (auto-detected if None)
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_heads = attention_heads
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_type = loss_type
        self.device = device or get_device()

        self.model: Optional[AttentionLSTM] = None
        self._input_size: Optional[int] = None
        self._last_sequence: Optional[np.ndarray] = None

    def _create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM training.

        Args:
            features: (n_samples, n_features) input features
            targets: (n_samples,) target values

        Returns:
            Tuple of (X, y) arrays with shape:
                X: (n_sequences, sequence_length, n_features)
                y: (n_sequences,)
        """
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i : i + self.sequence_length])
            y.append(targets[i + self.sequence_length])
        return np.array(X), np.array(y)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        val_split: float = 0.2,
    ) -> "LSTMForecaster":
        """
        Train LSTM model.

        Args:
            features: (n_samples, n_features) input features
            targets: (n_samples,) target values
            val_split: Fraction of data for validation

        Returns:
            Self for method chaining
        """
        self._input_size = features.shape[1]

        X, y = self._create_sequences(features, targets)

        # Store last sequence for forecasting
        self._last_sequence = features[-self.sequence_length :]

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
            shuffle=True,
        )

        # Initialize model
        self.model = AttentionLSTM(
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            attention_heads=self.attention_heads,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        if self.loss_type == "mse":
            criterion = nn.MSELoss()
        elif self.loss_type == "sharpe":
            criterion = SharpeLoss()
        elif self.loss_type == "sortino":
            criterion = SortinoLoss()
        elif self.loss_type == "combined":
            criterion = CombinedRiskLoss()
        else:
            criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_params(
                {
                    "lstm_hidden_size": self.hidden_size,
                    "lstm_num_layers": self.num_layers,
                    "lstm_sequence_length": self.sequence_length,
                    "lstm_attention_heads": self.attention_heads,
                    "lstm_loss_type": self.loss_type,
                    "lstm_epochs_trained": epoch + 1,
                }
            )
            mlflow.log_metrics(
                {
                    "lstm_best_val_loss": best_val_loss,
                    "lstm_final_train_loss": train_loss,
                }
            )

        return self

    def forecast(
        self,
        features: Optional[np.ndarray] = None,
        steps: int = 1,
    ) -> np.ndarray:
        """
        Generate forecast from feature sequence.

        Args:
            features: (sequence_length, n_features) input features.
                     If None, uses the last sequence from training.
            steps: Number of steps to forecast (currently only 1 supported)

        Returns:
            (steps,) array of predictions
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before forecasting")

        if features is None:
            if self._last_sequence is None:
                raise ValueError("No features provided and no stored sequence")
            features = self._last_sequence

        self.model.eval()

        if features.ndim == 2:
            features = features[-self.sequence_length :]
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        else:
            X = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            pred = self.model(X)

        return pred.cpu().numpy().flatten()

    def predict_sequence(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """
        Generate predictions for entire sequence (walk-forward).

        Args:
            features: (n_samples, n_features) input features
            targets: (n_samples,) targets (for sequence creation)

        Returns:
            (n_samples - sequence_length,) array of predictions
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before predicting")

        X, _ = self._create_sequences(features, targets)

        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_t)

        return predictions.cpu().numpy().flatten()
