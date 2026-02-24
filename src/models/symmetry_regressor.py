"""
Symmetry-aware tensor regression model.

Combines CLIP text encoder with dual-head architecture for:
    1. Tensor prediction (6 components)
    2. Crystal system classification (mixture weights)
"""

import torch
import torch.nn as nn
from typing import Tuple

from .clip_model import GraphTextCLIP
from .symmetry_head import TensorAndSymmetryHead


class TextTensorRegressorWithSymmetry(nn.Module):
    """
    Symmetry-aware tensor regression using pre-trained CLIP text encoder.

    Outputs both tensor prediction and crystal system classification.
    """

    def __init__(
        self,
        clip_model: GraphTextCLIP,
        tensor_dim: int = 6,  # For symmetric6 mode
        head_hidden_dim: int = None,
        head_dropout: float = 0.1,
        freeze_clip: bool = True,
        num_crystal_systems: int = 4,
        crystal_systems: list = None,
    ):
        super().__init__()

        self.clip_model = clip_model
        self.tensor_mode = "symmetric6"  # Always use 6-parameter mode

        # Dual-head: predicts tensor + symmetry mixture
        self.regression_head = TensorAndSymmetryHead(
            in_dim=clip_model.text_encoder.proj.out_features,
            num_crystal_systems=num_crystal_systems,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            crystal_systems=crystal_systems,
        )

        self.freeze_clip = freeze_clip
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def set_freeze_clip(self, freeze: bool):
        """Toggle CLIP encoder freezing."""
        self.freeze_clip = freeze
        for param in self.clip_model.parameters():
            param.requires_grad = not freeze

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embedding: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            return_embedding: If True, return (pred6, alpha, text_emb)

        Returns:
            pred6: [B, 6] predicted tensor components
            alpha: [B, K] symmetry mixture weights
        """
        # Get text embedding from CLIP encoder
        with torch.set_grad_enabled(not self.freeze_clip):
            text_emb = self.clip_model.text_encoder(input_ids, attention_mask)

        # Predict tensor and symmetry
        pred6, alpha = self.regression_head(text_emb)

        if return_embedding:
            return pred6, alpha, text_emb
        return pred6, alpha

    def compute_clip_loss(
        self,
        graph: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CLIP contrastive loss (for Phase 2 joint training)."""
        clip_loss, _ = self.clip_model(graph, input_ids, attention_mask)
        return clip_loss
