"""Encoder models for graph and text modalities."""

import torch
import torch.nn as nn
from typing import Optional, Literal
from abc import ABC, abstractmethod

from .components import PositionalEmbedding, TransformerEncoderLayer

try:
    from transformers import AutoModel, T5EncoderModel
except ImportError as e:
    raise ImportError("Please install transformers to use pretrained models.") from e


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


class BaseTextEncoder(nn.Module, ABC):
    """Abstract base class for text encoders."""

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] tokenized text
            attention_mask: [B, L] attention mask (1 for valid, 0 for padding)

        Returns:
            [B, embed_dim] normalized text embeddings
        """
        pass


class CustomTextEncoder(BaseTextEncoder):
    """Transformer-based text encoder with byte-level tokenization."""

    def __init__(
        self,
        vocab_size: int = 256,
        width: int = 512,
        max_len: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        embed_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, width)
        self.pos_emb = PositionalEmbedding(width, max_len)
        self.blocks = nn.ModuleList(
            [TransformerEncoderLayer(width, n_heads, mlp_ratio=4, dropout=dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(width)
        self.proj = nn.Linear(width, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] tokenized text
            attention_mask: [B, L] attention mask (1 for valid, 0 for padding)

        Returns:
            [B, embed_dim] normalized text embeddings
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

def _masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over valid tokens based on attention mask."""
    mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1) + 1e-8
    return summed / denom


class HuggingFaceTextEncoder(BaseTextEncoder):
    """Generic HuggingFace model encoder (T5, BERT, RoBERTa, etc.)."""

    def __init__(
        self,
        model_name: str = "t5-base",
        embed_dim: int = 512,
        pooling: Literal["mean", "first", "last"] = "mean",
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        train_layernorm_only: bool = False,
    ):
        super().__init__()

        # Use T5EncoderModel for T5 models, AutoModel for others
        if "t5" in model_name.lower():
            self.backbone = T5EncoderModel.from_pretrained(model_name, local_files_only=True)
        else:
            self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)

        self.hidden_dim = self.backbone.config.hidden_size if hasattr(self.backbone.config, 'hidden_size') else self.backbone.config.d_model
        self.pooling = pooling

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(self.hidden_dim, embed_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif train_layernorm_only:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for m in self.backbone.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] tokenized text
            attention_mask: [B, L] attention mask

        Returns:
            [B, embed_dim] normalized text embeddings
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state  # [B, L, H]

        if self.pooling == "mean":
            pooled = _masked_mean_pool(x, attention_mask)
        elif self.pooling == "first":
            pooled = x[:, 0, :]
        elif self.pooling == "last":
            idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1
            pooled = x[torch.arange(x.size(0), device=x.device), idx, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        pooled = self.dropout(pooled)
        z = self.proj(pooled)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z

