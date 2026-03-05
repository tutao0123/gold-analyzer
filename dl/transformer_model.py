"""
PriceTransformer: Transformer Encoder-based time-series price prediction model.
Pure PyTorch implementation, no extra dependencies.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # handle odd d_model
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class PriceTransformer(nn.Module):
    """
    Transformer Encoder time-series prediction model.

    Architecture:
      Input (seq_len, input_size)
        → Linear Projection → d_model
        → Positional Encoding
        → N × Transformer Encoder Layer (Multi-Head Attention + FFN)
        → Global Average Pooling
        → MLP Head → 1 (next-day close price)
    """

    def __init__(self, input_size=7, seq_length=60, d_model=64, nhead=4,
                 num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.input_size = input_size
        self.seq_length = seq_length
        self.d_model = d_model

        # input projection: (batch, seq, input_size) → (batch, seq, d_model)
        self.input_projection = nn.Linear(input_size, d_model)

        # positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 10)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier weight initialisation."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        x:      (batch, seq_len, input_size)  e.g. (32, 60, 7)
        output: (batch, 1)                    next-day close price prediction
        """
        # project to d_model dimensions
        x = self.input_projection(x)          # (batch, seq, d_model)

        # add positional encoding
        x = self.pos_encoder(x)               # (batch, seq, d_model)

        # Transformer Encoder
        x = self.transformer_encoder(x)       # (batch, seq, d_model)

        # global average pooling over all time steps
        x = x.mean(dim=1)                     # (batch, d_model)

        # prediction head
        out = self.head(x)                    # (batch, 1)
        return out


if __name__ == "__main__":
    # quick shape sanity check
    model = PriceTransformer(input_size=7, seq_length=60)
    dummy = torch.randn(4, 60, 7)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
