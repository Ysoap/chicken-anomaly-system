"""Stage 3 model: an LSTM autoencoder for temporal anomaly scoring.

The autoencoder is trained only on sequences of normal chicken behavior.
At run time it tries to reconstruct an input feature sequence; sequences
that do not look like normal behavior reconstruct poorly, and the mean
squared reconstruction error is the anomaly score.

Architecture
------------
Encoder: an LSTM reads the (window, input_dim) feature sequence and its
final hidden state is projected to a small latent vector (the bottleneck).
Decoder: the latent vector is repeated at every timestep and a second LSTM
plus a linear output layer reconstruct the original feature sequence.

The bottleneck forces the network to learn the regular structure of normal
motion (steady walking, pecking rhythms, calm flock movement) instead of
memorising individual sequences, which is what makes the reconstruction
error useful as an anomaly signal.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """LSTM encoder-decoder autoencoder for fixed-dimension sequences."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim, hidden_dim and latent_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sequences to latent vectors, shape (batch, latent_dim)."""
        _, (h_n, _c_n) = self.encoder(x)
        return self.to_latent(h_n[-1])

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent vectors back to sequences of length ``seq_len``."""
        start = self.from_latent(latent)  # (batch, hidden_dim)
        decoder_input = start.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoder_input)
        return self.output_layer(decoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input. Input and output: (batch, time, input_dim)."""
        latent = self.encode(x)
        return self.decode(latent, seq_len=x.size(1))

    @torch.no_grad()
    def score(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Per-sequence anomaly score: mean squared reconstruction error.

        Args:
            x: Tensor or NumPy array of shape (batch, time, input_dim).

        Returns:
            A 1-D tensor of shape (batch,) on the same device as the model.
            Higher means more anomalous.
        """
        if isinstance(x, np.ndarray):
            device = next(self.parameters()).device
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(dtype=torch.float32)
        was_training = self.training
        self.eval()
        try:
            reconstruction = self.forward(x)
            errors = (reconstruction - x) ** 2
            return errors.mean(dim=(1, 2))
        finally:
            if was_training:
                self.train()

    def config_dict(self) -> dict:
        """Constructor arguments, stored inside checkpoints."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
        }


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters, for logging."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
