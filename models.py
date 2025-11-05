
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (batch_first).
    """
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (T, E)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, E)
        self.register_buffer('pe', pe)  # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, E)
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


def generate_causal_mask(sz: int, device=None) -> torch.Tensor:
    """
    Generate a causal (square subsequent) mask for target sequence of length sz.
    """
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    return mask


class QuantumTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for quantum feedback control.
    - Encoder ingest sequence of measurement record tokens + an initial-state token.
    - Decoder autoregressively predicts discretized control bins.

    Inputs:
      src: (B, S, enc_in_dim)
      src_key_padding_mask: (B, S) True at PAD positions
      tgt_tokens: (B, T) integer token ids in [0..num_bins] where 0 is <SOS> (teacher forcing input)
      tgt_key_padding_mask: (B, T)
    Outputs:
      logits: (B, T, num_bins) over action bins (excluding <SOS>)
    """
    def __init__(self,
                 enc_in_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_bins: int = 51,
                 pad_id: int = -100,
                 sos_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.num_bins = num_bins
        self.pad_id = pad_id
        self.sos_id = sos_id

        self.src_proj = nn.Linear(enc_in_dim, d_model)
        self.tok_embed = nn.Embedding(num_bins + 1, d_model)  # +1 for <SOS>
        self.pos_src = PositionalEncoding(d_model, dropout=dropout)
        self.pos_tgt = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.out_proj = nn.Linear(d_model, num_bins)  # logits over real bins only (exclude <SOS>)

    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        x = self.src_proj(src) * math.sqrt(self.d_model)
        x = self.pos_src(x)
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self,
               tgt_tokens: torch.Tensor,
               memory: torch.Tensor,
               tgt_key_padding_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None):
        """
        tgt_tokens includes <SOS> token ids. We embed and apply causal mask.
        """
        B, T = tgt_tokens.shape
        tgt_emb = self.tok_embed(tgt_tokens) * math.sqrt(self.d_model)
        tgt_emb = self.pos_tgt(tgt_emb)
        tgt_mask = generate_causal_mask(T, device=tgt_tokens.device)
        out = self.decoder(tgt=tgt_emb, memory=memory,
                           tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        logits = self.out_proj(out)  # (B, T, num_bins)
        return logits

    def forward(self,
                src: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor],
                tgt_tokens: torch.Tensor,
                tgt_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        logits = self.decode(tgt_tokens, memory,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=src_key_padding_mask)
        return logits

    @torch.no_grad()
    def greedy_decode(self,
                      src: torch.Tensor,
                      src_key_padding_mask: Optional[torch.Tensor],
                      max_len: int,
                      sos_id: int = 0) -> torch.Tensor:
        """
        Autoregressive decoding with greedy argmax.
        Returns: (B, max_len) predicted action-bin token ids in [1..num_bins]
        """
        device = src.device
        memory = self.encode(src, src_key_padding_mask)
        B = src.size(0)
        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        outputs = []
        for t in range(max_len):
            logits = self.decode(ys, memory,
                                 tgt_key_padding_mask=None,
                                 memory_key_padding_mask=src_key_padding_mask)  # (B, t+1, num_bins)
            step_logits = logits[:, -1, :]  # (B, num_bins)
            pred = torch.argmax(step_logits, dim=-1) + 1  # shift to [1..num_bins]
            outputs.append(pred.unsqueeze(1))
            ys = torch.cat([ys, pred.unsqueeze(1)], dim=1)
        return torch.cat(outputs, dim=1)  # (B, max_len)
