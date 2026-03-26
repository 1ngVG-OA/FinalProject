"""Torch model definitions for the neural forecasting step."""

from __future__ import annotations

import torch
from torch import nn


class MLPForecaster(nn.Module):
    """Time-lagged feedforward neural network for one-step regression."""

    def __init__(self, input_size: int, hidden_size: int, activation: str = "relu", dropout: float = 0.0) -> None:
        super().__init__()

        if activation == "tanh":
            activation_layer: nn.Module = nn.Tanh()
        else:
            activation_layer = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LSTMForecaster(nn.Module):
    """Many-to-one LSTM forecaster for one-step regression."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_state = out[:, -1, :]
        return self.output(last_state)