
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


def generate_causal_memory_mask(tgt_len: int,
                               src_len: int,
                               *,
                               delay: int = 1,
                               src_has_init_token: bool = True,
                               device=None) -> torch.Tensor:
    """Generate a causal mask for encoder->decoder cross-attention.

    We assume the encoder source sequence is arranged as:
      [init_token, meas_0, meas_1, ...]
    and decoder position t corresponds to control step t.

    With a discrete feedback delay of `delay` steps, control at step t may only
    depend on measurements up to meas_{t-delay}. (For delay=1, control at step t
    can use meas_{t-1} but not meas_t.)

    Returns a boolean tensor of shape (tgt_len, src_len) with True entries
    indicating masked (disallowed) attention weights.
    """
    if delay < 0:
        raise ValueError(f"delay must be >= 0, got {delay}")

    mask = torch.ones((tgt_len, src_len), dtype=torch.bool, device=device)
    for t in range(tgt_len):
        max_meas = t - delay
        if src_has_init_token:
            # init token at position 0 always allowed
            allowed_max = 0
            if max_meas >= 0:
                allowed_max = min(src_len - 1, 1 + max_meas)
            mask[t, :allowed_max + 1] = False
        else:
            if max_meas >= 0:
                allowed_max = min(src_len - 1, max_meas)
                mask[t, :allowed_max + 1] = False
            # else: leave fully masked
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
        self.enc_in_dim = int(enc_in_dim)
        self.d_model = d_model
        self.num_bins = num_bins
        self.pad_id = pad_id
        self.sos_id = sos_id

        self.src_proj = nn.Linear(enc_in_dim, d_model)
        self.tok_embed = nn.Embedding(num_bins + 1, d_model)  # +1 for <SOS>
        # Optional measurement features aligned with decoder time steps.
        # This helps match the paper description that the decoder receives the
        # measurement record directly (in addition to cross-attending to the encoder).
        self.tgt_meas_proj = nn.Linear(1, d_model)
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

        # Causal masking configuration.
        # delay=1 means control at step t may depend on measurements up to t-1.
        self.default_memory_delay = 1
        self.src_has_init_token = True

        # If False, do not add tgt_meas features to decoder embeddings.
        self.use_tgt_meas = True

    def encode(self,
               src: torch.Tensor,
               src_key_padding_mask: Optional[torch.Tensor] = None,
               *,
               enforce_causal_encoder: bool = True,
               src_mask: Optional[torch.Tensor] = None):
        """Encode measurement record + initial state.

        If enforce_causal_encoder=True (default), applies a causal self-attention
        mask to the encoder so that *no* encoder position can incorporate
        information from future measurements when the full record is provided.

        This is important for faithful online feedback: even if the decoder
        cross-attention is masked, a bidirectional encoder can leak future
        measurements into past memory positions.
        """
        x = self.src_proj(src) * math.sqrt(self.d_model)
        x = self.pos_src(x)

        if src_mask is None and enforce_causal_encoder:
            src_mask = generate_causal_mask(src.size(1), device=src.device)

        memory = self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self,
               tgt_tokens: torch.Tensor,
               memory: torch.Tensor,
               tgt_key_padding_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None,
               *,
               tgt_meas: Optional[torch.Tensor] = None,
               enforce_causal_memory: bool = True,
               memory_delay: Optional[int] = None,
               memory_mask: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None):
        """Decode.

        Args:
            tgt_tokens: (B,T) token ids in [0..num_bins], where 0 is <SOS>.
            memory: (B,S,d_model) encoder output.
            tgt_meas: optional (B,T,1) measurement features aligned with decoder time.
            enforce_causal_memory: if True, applies a causal cross-attention mask.
            memory_delay: overrides self.default_memory_delay if provided.
            memory_mask: optional manual (T,S) cross-attention mask.
            tgt_mask: optional manual (T,T) causal self-attention mask.
        """
        B, T = tgt_tokens.shape
        tgt_emb = self.tok_embed(tgt_tokens) * math.sqrt(self.d_model)
        if self.use_tgt_meas and (tgt_meas is not None):
            if tgt_meas.ndim != 3 or tgt_meas.shape[:2] != (B, T) or tgt_meas.size(-1) != 1:
                raise ValueError(f"tgt_meas must have shape (B,T,1); got {tuple(tgt_meas.shape)}")
            tgt_emb = tgt_emb + self.tgt_meas_proj(tgt_meas)
        tgt_emb = self.pos_tgt(tgt_emb)

        if tgt_mask is None:
            tgt_mask = generate_causal_mask(T, device=tgt_tokens.device)

        if memory_mask is None and enforce_causal_memory:
            delay = self.default_memory_delay if memory_delay is None else int(memory_delay)
            memory_mask = generate_causal_memory_mask(
                tgt_len=T,
                src_len=memory.size(1),
                delay=delay,
                src_has_init_token=self.src_has_init_token,
                device=tgt_tokens.device,
            )

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.out_proj(out)  # (B, T, num_bins)
        return logits

    def forward(self,
                src: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor],
                tgt_tokens: torch.Tensor,
                tgt_key_padding_mask: Optional[torch.Tensor],
                *,
                tgt_meas: Optional[torch.Tensor] = None,
                enforce_causal_memory: bool = True,
                memory_delay: Optional[int] = None) -> torch.Tensor:
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        logits = self.decode(tgt_tokens, memory,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=src_key_padding_mask,
                             tgt_meas=tgt_meas,
                             enforce_causal_memory=enforce_causal_memory,
                             memory_delay=memory_delay)
        return logits

    @torch.no_grad()
    def greedy_decode(self,
                      src: torch.Tensor,
                      src_key_padding_mask: Optional[torch.Tensor],
                      max_len: int,
                      sos_id: int = 0,
                      *,
                      tgt_meas: Optional[torch.Tensor] = None,
                      enforce_causal_memory: bool = True,
                      memory_delay: Optional[int] = None) -> torch.Tensor:
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
            meas_now = None
            if tgt_meas is not None:
                meas_now = tgt_meas[:, :ys.size(1), :]
            logits = self.decode(
                ys,
                memory,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_key_padding_mask,
                tgt_meas=meas_now,
                enforce_causal_memory=enforce_causal_memory,
                memory_delay=memory_delay,
            )  # (B, t+1, num_bins)
            step_logits = logits[:, -1, :]  # (B, num_bins)
            pred = torch.argmax(step_logits, dim=-1) + 1  # shift to [1..num_bins]
            outputs.append(pred.unsqueeze(1))
            ys = torch.cat([ys, pred.unsqueeze(1)], dim=1)
        return torch.cat(outputs, dim=1)  # (B, max_len)

    @torch.no_grad()
    def predict_bins_from_measurements(self,
                                      src: torch.Tensor,
                                      src_key_padding_mask: Optional[torch.Tensor],
                                      tgt_meas: torch.Tensor,
                                      *,
                                      enforce_causal_memory: bool = True,
                                      memory_delay: Optional[int] = None) -> torch.Tensor:
        """Non-autoregressive inference: predict λ bins directly from measurement inputs.

        This matches the architecture description in the paper where the decoder
        consumes the measurement record with a causal mask.

        Args:
            src: (B,S,D) encoder inputs (ρ0 token + measurement tokens).
            tgt_meas: (B,T,1) measurement features aligned with decoder time steps
                      (e.g., tgt_meas[:,0]=0 and tgt_meas[:,t]=r_{t-1}).

        Returns:
            (B,T) integer bin indices in [0..num_bins-1].
        """
        if tgt_meas.ndim != 3 or tgt_meas.size(-1) != 1:
            raise ValueError(f"tgt_meas must have shape (B,T,1); got {tuple(tgt_meas.shape)}")
        B, T, _ = tgt_meas.shape
        device = src.device
        # All-SOS token stream; all information is carried in tgt_meas + positional encoding.
        tgt_tokens = torch.zeros((B, T), dtype=torch.long, device=device)
        logits = self.forward(
            src,
            src_key_padding_mask,
            tgt_tokens,
            tgt_key_padding_mask=None,
            tgt_meas=tgt_meas,
            enforce_causal_memory=enforce_causal_memory,
            memory_delay=memory_delay,
        )
        return torch.argmax(logits, dim=-1)
