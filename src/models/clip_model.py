"""CLIP-style contrastive learning model for graph-text pairs."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Literal

from .encoders import GraphEncoder, CustomTextEncoder, HuggingFaceTextEncoder


class GraphTextCLIP(nn.Module):
    """Contrastive learning model aligning graph and text embeddings."""

    def __init__(
        self,
        graph_in_dim: int,
        clip_dim: int,
        text_backend: Literal["huggingface", "custom"] = "huggingface",
        # For custom backend
        text_width: int = 512,
        max_len: int = 256,
        text_layers: int = 6,
        text_heads: int = 8,
        vocab_size: int = 256,
        dropout: float = 0.0,
        # For HuggingFace backend
        text_model_name: str = "t5-base",
        text_pooling: Literal["mean", "first", "last"] = "mean",
        freeze_text_backbone: bool = False,
        train_text_layernorm_only: bool = False,
        text_dropout: float = 0.0,
    ):
        super().__init__()
        self.graph_encoder = GraphEncoder(graph_in_dim, clip_dim)

        if text_backend == "custom":
            self.text_encoder = CustomTextEncoder(
                vocab_size=vocab_size,
                width=text_width,
                max_len=max_len,
                n_layers=text_layers,
                n_heads=text_heads,
                embed_dim=clip_dim,
                dropout=dropout,
            )
        elif text_backend == "huggingface":
            self.text_encoder = HuggingFaceTextEncoder(
                model_name=text_model_name,
                embed_dim=clip_dim,
                pooling=text_pooling,
                freeze_backbone=freeze_text_backbone,
                train_layernorm_only=train_text_layernorm_only,
                dropout=text_dropout,
            )
        else:
            raise ValueError(f"Unsupported text backend: {text_backend}.")

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        graph: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph: [B, graph_in_dim]
            input_ids: [B, L]
            attention_mask: [B, L]

        Returns:
            loss: contrastive loss
            logits: [B, B] similarity matrix
        """
        g_emb = self.graph_encoder(graph)
        t_emb = self.text_encoder(input_ids, attention_mask)

        logits = self.logit_scale.exp() * (g_emb @ t_emb.t())

        labels = torch.arange(logits.size(0), device=logits.device)
        loss_g2t = F.cross_entropy(logits, labels)
        loss_t2g = F.cross_entropy(logits.t(), labels)
        loss = (loss_g2t + loss_t2g) / 2.0

        return loss, logits

    @torch.no_grad()
    def encode_graph(self, graph: torch.Tensor) -> torch.Tensor:
        """Encode graph embeddings."""
        return self.graph_encoder(graph)

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text."""
        return self.text_encoder(input_ids, attention_mask)

    @staticmethod
    def compute_retrieval_accuracy(logits: torch.Tensor) -> Tuple[float, float]:
        """
        Compute retrieval accuracy from similarity matrix.

        Args:
            logits: [B, B] similarity matrix

        Returns:
            acc_g2t: graph-to-text retrieval accuracy
            acc_t2g: text-to-graph retrieval accuracy
        """
        gt = torch.arange(logits.size(0), device=logits.device)
        acc_g2t = (logits.argmax(1) == gt).float().mean().item()
        acc_t2g = (logits.argmax(0) == gt).float().mean().item()
        return acc_g2t, acc_t2g
