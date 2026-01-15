"""Encoder models for graph and text modalities."""

import torch
import torch.nn as nn
from typing import Optional

from .components import PositionalEmbedding, TransformerEncoderLayer


class GraphEncoder(nn.Module):
    """Linear projection encoder for graph embeddings."""

    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_emb: [B, in_dim] pre-computed graph embeddings

        Returns:
            [B, out_dim] normalized embeddings
        """
        z = self.proj(graph_emb)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z


class TextEncoder(nn.Module):
    """Transformer-based text encoder with byte-level tokenization."""

    def __init__(
        self,
        vocab_size: int = 256,
        width: int = 512,
        max_len: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        emb_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, width)
        self.pos_emb = PositionalEmbedding(width, max_len)
        self.blocks = nn.ModuleList(
            [TransformerEncoderLayer(width, n_heads, mlp_ratio=4, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(width)
        self.proj = nn.Linear(width, emb_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] tokenized text
            attention_mask: [B, L] attention mask (1 for valid, 0 for padding)

        Returns:
            [B, emb_dim] normalized text embeddings
        """
        x = self.tok_emb(input_ids)
        x = self.pos_emb(x)

        for blk in self.blocks:
            x = blk(x, attn_mask=attention_mask)

        x = self.ln_f(x)

        # Mean pooling over valid tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        z = self.proj(pooled)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z
