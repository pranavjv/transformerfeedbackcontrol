from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class RNNConfig:
    context_len: int = 60
    hidden_dim: int = 128
    num_layers: int = 1
    cell: str = "gru"  # "rnn" or "gru"
    dropout: float = 0.0
    include_time: bool = True


class MeasurementRNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_bins: int,
                 *,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 cell: str = "gru",
                 dropout: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_bins = int(num_bins)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.cell = str(cell).lower()

        if self.cell == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=float(dropout) if self.num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif self.cell == "rnn":
            self.rnn = nn.RNN(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                nonlinearity="tanh",
                dropout=float(dropout) if self.num_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown cell type: {cell}")

        self.head = nn.Linear(self.hidden_dim, self.num_bins)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            x: (B,L,input_dim)
            h0: optional initial hidden state (num_layers,B,hidden_dim)
        Returns:
            logits: (B,num_bins) for the *last* time step
            hN: final hidden state
        """
        out, hN = self.rnn(x, h0)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits, hN


def build_window_input(rho0: np.ndarray,
                       r_hist: np.ndarray,
                       *,
                       t: int,
                       T: int,
                       context_len: int,
                       include_time: bool = True) -> np.ndarray:
    """Construct an (L,D) input window for decision time t."""
    rho0 = np.asarray(rho0)
    rho_feat = np.concatenate([rho0.real.reshape(-1), rho0.imag.reshape(-1)], axis=0).astype(np.float32)
    if rho_feat.shape[0] != 8:
        raise ValueError("Expected 2x2 rho0")

    L = int(context_len)
    meas = np.zeros((L,), dtype=np.float32)
    # Past measurements up to dr_{t-1}
    for j in range(L):
        k = t - L + j  # measurement index
        if 0 <= k < len(r_hist):
            meas[j] = float(r_hist[k])

    if include_time:
        tau = float(t / max(1, T - 1))
        time_feat = np.full((L, 1), tau, dtype=np.float32)
        x = np.concatenate([
            meas.reshape(L, 1),
            np.repeat(rho_feat.reshape(1, -1), L, axis=0),
            time_feat,
        ], axis=1)
    else:
        x = np.concatenate([
            meas.reshape(L, 1),
            np.repeat(rho_feat.reshape(1, -1), L, axis=0),
        ], axis=1)
    return x
