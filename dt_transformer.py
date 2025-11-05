from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.0, max_len: int=8192):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div)
        pe[:,1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        T = x.size(1)
        return self.dropout(x + self.pe[:, :T, :])

def generate_causal_mask(sz: int, device=None) -> torch.Tensor:
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

class DecisionTransformer(nn.Module):
    def __init__(self,
                 state_dim: int,
                 num_bins: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 ff_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, d_model)
        self.rtgs_proj  = nn.Linear(1, d_model)
        self.act_embed  = nn.Embedding(num_bins + 1, d_model)  # +1 for <SOS> action
        self.pos_enc = PositionalEncoding(d_model, dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=ff_dim, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_bins)

    def forward(self,
                rtgs: torch.Tensor,      # (B,T,1)
                states: torch.Tensor,    # (B,T,state_dim)
                act_tokens: torch.Tensor # (B,T) in [0..num_bins]
                ) -> torch.Tensor:
        xs = self.state_proj(states)
        xr = self.rtgs_proj(rtgs)
        xa = self.act_embed(act_tokens)
        x = xs + xr + xa
        x = self.pos_enc(x)
        mask = generate_causal_mask(x.size(1), device=x.device)
        h = self.encoder(x, mask)
        logits = self.head(h)  # (B,T,num_bins)
        return logits

    @torch.no_grad()
    def greedy(self, rtgs: torch.Tensor, states: torch.Tensor, T: int, sos_id: int = 0):
        B = states.size(0)
        device = states.device
        actions = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        outs = []
        for t in range(T):
            logits = self.forward(rtgs[:, :t+1], states[:, :t+1], actions)  # (B,t+1,C)
            pred = torch.argmax(logits[:,-1,:], dim=-1) + 1  # shift off <SOS>
            outs.append(pred.unsqueeze(1))
            actions = torch.cat([actions, pred.unsqueeze(1)], dim=1)
        return torch.cat(outs, dim=1)
