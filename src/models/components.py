"""Neural network components for Graph-Text CLIP model."""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, width: int, max_seq_length: int):
        super().__init__()
        pe = torch.zeros(max_seq_length, width)
        position = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AttentionHead(nn.Module):
    """Single attention head with optional masking."""

    def __init__(self, width: int, head_size: int):
        super().__init__()
        self.query = nn.Linear(width, head_size)
        self.key = nn.Linear(width, head_size)
        self.value = nn.Linear(width, head_size)
        self.scale = head_size ** -0.5

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = (Q @ K.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1)  # [B, 1, L]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        A = torch.softmax(scores, dim=-1)
        return A @ V


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, width: int, n_heads: int):
        super().__init__()
        assert width % n_heads == 0, f"width {width} must be divisible by n_heads {n_heads}"

        head_size = width // n_heads
        self.heads = nn.ModuleList([AttentionHead(width, head_size) for _ in range(n_heads)])
        self.W_o = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = torch.cat([h(x, attn_mask=attn_mask) for h in self.heads], dim=-1)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm architecture."""

    def __init__(self, width: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.mha = MultiHeadAttention(width, n_heads)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * mlp_ratio, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.mha(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x
