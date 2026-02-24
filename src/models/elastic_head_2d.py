"""
Dual-head architecture for 2D elastic tensor prediction with symmetry awareness.

Outputs:
    - pred6: (B, 6) = [C11, C22, C12, C66, C16, C26]
    - alpha: (B, 4) = softmax([tri, rect, tetra, hexa])

Key: alpha incorporates learnable z + fixed Robo prior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .elastic_projection_2d import get_symmetry_prior


class ElasticTensorHead(nn.Module):
    """
    Dual-head for 2D elastic tensor + symmetry mixture.

    Architecture:
        Text Embedding (512)
            ↓
        ┌─────────┴─────────┐
        ↓                   ↓
    Tensor Head       Symmetry Head
        ↓                   ↓
    pred6 (6)          z logits (4)
                            ↓
                     α = softmax(z + β·log p_prior)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
        beta_prior: float = 1.0,  # Prior strength (0 = ignore prior)
        freeze_alpha_head: bool = False,  # For Phase 1
    ):
        """
        Args:
            in_dim: Input feature dimension (from CLIP)
            hidden_dim: Hidden layer size (default: same as in_dim)
            dropout: Dropout probability
            beta_prior: Weight for Robo prior (β in formula)
            freeze_alpha_head: If True, freeze symmetry head (Phase 1)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_dim

        self.beta_prior = beta_prior
        self.freeze_alpha_head = freeze_alpha_head

        # Tensor prediction head (always active)
        self.tensor_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),  # [C11, C22, C12, C66, C16, C26]
        )

        # Symmetry classification head (z logits)
        self.alpha_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),  # [tri, rect, tetra, hexa]
        )

        # Optionally freeze alpha head
        if freeze_alpha_head:
            for param in self.alpha_head.parameters():
                param.requires_grad = False

    def set_freeze_alpha_head(self, freeze: bool):
        """Toggle alpha head freezing (for phased training)."""
        self.freeze_alpha_head = freeze
        for param in self.alpha_head.parameters():
            param.requires_grad = not freeze

    def set_beta_prior(self, beta: float):
        """Update prior strength (for scheduled training)."""
        self.beta_prior = beta

    def forward(
        self,
        features: torch.Tensor,
        crystal_systems: Optional[List[str]] = None,
    ):
        """
        Args:
            features: (B, in_dim) encoded features from CLIP
            crystal_systems: list of B crystal system names (from Robo)

        Returns:
            pred6: (B, 6) predicted elastic tensor
            alpha: (B, 4) symmetry mixture weights
            z_logits: (B, 4) raw logits (before prior)
        """
        # Tensor prediction
        pred6 = self.tensor_head(features)

        # Symmetry logits (learnable)
        z_logits = self.alpha_head(features)

        # Incorporate Robo prior
        if crystal_systems is not None and self.beta_prior > 0:
            alpha = self._compute_alpha_with_prior(z_logits, crystal_systems)
        else:
            # No prior: just softmax
            alpha = F.softmax(z_logits, dim=-1)

        return pred6, alpha, z_logits

    def _compute_alpha_with_prior(
        self,
        z_logits: torch.Tensor,
        crystal_systems: List[str],
    ) -> torch.Tensor:
        """
        Compute α = softmax(z + β·log p_prior).

        Args:
            z_logits: (B, 4) learnable logits
            crystal_systems: list of B system names

        Returns:
            alpha: (B, 4) softmax probabilities
        """
        B = z_logits.shape[0]
        device = z_logits.device

        # Get prior for each sample
        prior_probs = []
        for system in crystal_systems:
            prior = get_symmetry_prior(system)
            prior_probs.append(prior)

        # Convert to tensor: (B, 4)
        prior_probs = torch.tensor(
            prior_probs, dtype=z_logits.dtype, device=device
        )

        # Compute log prior (with epsilon for stability)
        log_prior = torch.log(prior_probs + 1e-12)

        # Combine: α = softmax(z + β·log p_prior)
        combined_logits = z_logits + self.beta_prior * log_prior
        alpha = F.softmax(combined_logits, dim=-1)

        return alpha
