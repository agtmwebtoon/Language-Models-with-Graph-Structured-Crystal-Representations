"""
Complete 2D elastic tensor regressor with symmetry awareness.

Model: CLIP Text Encoder → Dual Head → (pred6, α)

Supports:
    - Phase 1: Freeze α-head, train tensor head
    - Phase 2: Unfreeze α-head, joint training
    - Robo-based symmetry priors
    - Scheduled λ_sym
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from .clip_model import GraphTextCLIP
from .elastic_head_2d import ElasticTensorHead


class TextElasticRegressor2D(nn.Module):
    """
    2D Elastic tensor regression with symmetry awareness.

    Architecture:
        Text (Robo) → CLIP → ElasticTensorHead → (pred6, α)

    Supports phased training:
        Phase 1: α-head frozen, stabilize tensor prediction
        Phase 2: α-head active, learn effective symmetry
    """

    def __init__(
        self,
        clip_model: GraphTextCLIP,
        head_hidden_dim: int = None,
        head_dropout: float = 0.1,
        beta_prior: float = 1.0,  # Prior strength
        freeze_clip: bool = True,
        freeze_alpha_head: bool = False,  # For Phase 1
    ):
        """
        Args:
            clip_model: Pre-trained CLIP model
            head_hidden_dim: Hidden size for regression heads
            head_dropout: Dropout probability
            beta_prior: Weight for Robo prior (β)
            freeze_clip: Whether to freeze CLIP encoder
            freeze_alpha_head: Whether to freeze α-head (Phase 1)
        """
        super().__init__()

        self.clip_model = clip_model
        self.freeze_clip = freeze_clip

        # Dual-head architecture
        self.regression_head = ElasticTensorHead(
            in_dim=clip_model.text_encoder.proj.out_features,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            beta_prior=beta_prior,
            freeze_alpha_head=freeze_alpha_head,
        )

        # Freeze CLIP if specified
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def set_freeze_clip(self, freeze: bool):
        """Toggle CLIP encoder freezing."""
        self.freeze_clip = freeze
        for param in self.clip_model.parameters():
            param.requires_grad = not freeze

    def set_freeze_alpha_head(self, freeze: bool):
        """Toggle α-head freezing (for Phase 1 → Phase 2 transition)."""
        self.regression_head.set_freeze_alpha_head(freeze)

    def set_beta_prior(self, beta: float):
        """Update prior strength (for scheduled training)."""
        self.regression_head.set_beta_prior(beta)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        crystal_systems: Optional[List[str]] = None,
        return_embedding: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (B, L) tokenized text
            attention_mask: (B, L) attention mask
            crystal_systems: list of B crystal system names (from Robo)
            return_embedding: If True, also return text embedding

        Returns:
            pred6: (B, 6) elastic tensor [C11, C22, C12, C66, C16, C26]
            alpha: (B, 4) symmetry mixture [tri, rect, tetra, hexa]
            z_logits: (B, 4) raw logits (before prior)
            text_emb: (B, clip_dim) if return_embedding=True
        """
        # Get text embedding from CLIP
        with torch.set_grad_enabled(not self.freeze_clip):
            text_emb = self.clip_model.text_encoder(input_ids, attention_mask)

        # Predict tensor and symmetry
        pred6, alpha, z_logits = self.regression_head(
            text_emb, crystal_systems=crystal_systems
        )

        if return_embedding:
            return pred6, alpha, z_logits, text_emb
        return pred6, alpha, z_logits

    def compute_clip_loss(
        self,
        graph: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP contrastive loss (for Phase 2 joint training).

        Args:
            graph: (B, graph_dim) graph embeddings
            input_ids: (B, L) text tokens
            attention_mask: (B, L) mask

        Returns:
            clip_loss: scalar
        """
        clip_loss, _ = self.clip_model(graph, input_ids, attention_mask)
        return clip_loss

    def get_phase_info(self) -> dict:
        """Get current training phase information."""
        return {
            "clip_frozen": self.freeze_clip,
            "alpha_head_frozen": self.regression_head.freeze_alpha_head,
            "beta_prior": self.regression_head.beta_prior,
        }


class TextElasticRegressor2DSimple(nn.Module):
    """
    Baseline without symmetry awareness (for comparison).

    Just predicts 6 components directly without α-head.
    """

    def __init__(
        self,
        clip_model: GraphTextCLIP,
        head_hidden_dims: list = None,
        head_dropout: float = 0.1,
        freeze_clip: bool = True,
    ):
        super().__init__()

        self.clip_model = clip_model
        self.freeze_clip = freeze_clip

        if head_hidden_dims is None:
            head_hidden_dims = [512, 256]

        # Simple MLP
        layers = []
        prev_dim = clip_model.text_encoder.proj.out_features
        for hidden_dim in head_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 6))  # [C11, C22, C12, C66, C16, C26]
        self.regression_head = nn.Sequential(*layers)

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        crystal_systems: Optional[List[str]] = None,  # Ignored
    ):
        """
        Forward pass (simple regression).

        Returns:
            pred6: (B, 6)
            None: (for API compatibility)
            None: (for API compatibility)
        """
        with torch.set_grad_enabled(not self.freeze_clip):
            text_emb = self.clip_model.text_encoder(input_ids, attention_mask)

        pred6 = self.regression_head(text_emb)
        return pred6, None, None
